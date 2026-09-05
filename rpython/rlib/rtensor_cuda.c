#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifndef RPY_EXPORTED
#  define RPY_EXPORTED extern __attribute__((visibility("default")))
#endif

static CUcontext ctx;
static int inited;
typedef struct { CUdeviceptr p; long n; } buf_t;
static buf_t *allocs, *freed;
static long nallocs, capallocs, nfreed, capfreed;

static int rt_init(void)
{
    CUdevice dev;
    if (inited) return inited > 0;
    inited = -1;
    if (cuInit(0) != CUDA_SUCCESS) return 0;
    if (cuDeviceGet(&dev, 0) != CUDA_SUCCESS) return 0;
    if (cuDevicePrimaryCtxRetain(&ctx, dev) != CUDA_SUCCESS) return 0;
    if (cuCtxSetCurrent(ctx) != CUDA_SUCCESS) return 0;
    inited = 1;
    return 1;
}

static void push(buf_t **arr, long *n, long *cap, CUdeviceptr p, long size)
{
    if (*n == *cap) {
        *cap = *cap ? *cap * 2 : 1024;
        *arr = realloc(*arr, *cap * sizeof(buf_t));
    }
    (*arr)[*n].p = p;
    (*arr)[*n].n = size;
    (*n)++;
}

RPY_EXPORTED int rt_cuda_available(void)
{
    return rt_init();
}

RPY_EXPORTED long rt_cuda_load(const char *ptx, const char *name)
{
    CUmodule mod;
    CUfunction fn;
    if (!rt_init()) return 0;
    if (cuModuleLoadData(&mod, ptx) != CUDA_SUCCESS) return 0;
    if (cuModuleGetFunction(&fn, mod, name) != CUDA_SUCCESS) return 0;
    return (long)fn;
}

RPY_EXPORTED long rt_cuda_alloc(long n)
{
    CUdeviceptr p = 0;
    long i;
    if (!rt_init()) return 0;
    for (i = nfreed - 1; i >= 0; i--) {
        if (freed[i].n == n) {
            p = freed[i].p;
            freed[i] = freed[--nfreed];
            break;
        }
    }
    if (!p && cuMemAlloc(&p, n * sizeof(double)) != CUDA_SUCCESS) return 0;
    cuMemsetD8(p, 0, n * sizeof(double));
    push(&allocs, &nallocs, &capallocs, p, n);
    return (long)p;
}

RPY_EXPORTED long rt_cuda_upload(double *host, long n)
{
    long p = rt_cuda_alloc(n);
    if (p) cuMemcpyHtoD((CUdeviceptr)p, host, n * sizeof(double));
    return p;
}

RPY_EXPORTED int rt_cuda_download(long dptr, double *host, long n)
{
    return cuMemcpyDtoH(host, (CUdeviceptr)dptr, n * sizeof(double)) == CUDA_SUCCESS;
}

RPY_EXPORTED void rt_cuda_reset(void)
{
    long i;
    for (i = 0; i < nallocs; i++) cuMemFree(allocs[i].p);
    for (i = 0; i < nfreed; i++) cuMemFree(freed[i].p);
    nallocs = nfreed = 0;
}

RPY_EXPORTED long rt_cuda_mark(void)
{
    return nallocs;
}

RPY_EXPORTED void rt_cuda_release_since(long mark)
{
    long i;
    for (i = mark; i < nallocs; i++)
        push(&freed, &nfreed, &capfreed, allocs[i].p, allocs[i].n);
    if (mark < nallocs) nallocs = mark;
}

RPY_EXPORTED void rt_cuda_release_range(long from, long to)
{
    long i;
    if (from < 0 || to > nallocs || from >= to) return;
    for (i = from; i < to; i++)
        push(&freed, &nfreed, &capfreed, allocs[i].p, allocs[i].n);
    memmove(&allocs[from], &allocs[to], (nallocs - to) * sizeof(buf_t));
    nallocs -= to - from;
}

RPY_EXPORTED void rt_cuda_sync(void)
{
    cuCtxSynchronize();
}

RPY_EXPORTED int rt_cuda_launch(long fn, long *inputs, int ninputs, long n,
                                long out, int threads, long elems_per_block,
                                int shared, int nextra)
{
    void *params[16];
    void *null = 0;
    long argn = n;
    int i;
    unsigned blocks = (unsigned)((n + elems_per_block - 1) / elems_per_block);
    if (!rt_init() || ninputs > 7 || nextra > 6) return 0;
    for (i = 0; i < ninputs; i++) params[i] = &inputs[i];
    params[ninputs] = &out;
    params[ninputs + 1] = &argn;
    for (i = 0; i < nextra; i++) params[ninputs + 2 + i] = &null;
    return cuLaunchKernel((CUfunction)fn, blocks ? blocks : 1, 1, 1,
                          threads, 1, 1, shared, 0, params, 0) == CUDA_SUCCESS;
}
