#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <dlfcn.h>

#ifndef RPY_EXPORTED
#  define RPY_EXPORTED extern __attribute__((visibility("default")))
#endif

#define RTENSOR_CUBLAS_DEFAULT \
    "/home/yusuke/.venvs/triton/lib/python3.14/site-packages/nvidia/cu13/lib/libcublas.so.13"

typedef int (*cublasCreate_v2_t)(void **handle);
typedef int (*cublasDgemm_v2_t)(void *handle, int transa, int transb,
    int m, int n, int k, const double *alpha, const double *A, int lda,
    const double *B, int ldb, const double *beta, double *C, int ldc);
typedef int (*cublasDgemmStridedBatched_t)(void *handle, int transa, int transb,
    int m, int n, int k, const double *alpha, const double *A, int lda,
    long long strideA, const double *B, int ldb, long long strideB,
    const double *beta, double *C, int ldc, long long strideC, int batchCount);

static void *cublas_lib;
static void *cublas_handle;
static cublasCreate_v2_t p_cublasCreate_v2;
static cublasDgemm_v2_t p_cublasDgemm_v2;
static cublasDgemmStridedBatched_t p_cublasDgemmStridedBatched;
static int cublas_inited;

static CUcontext ctx;
static int inited;
typedef struct { CUdeviceptr p; long n; } buf_t;
static buf_t *allocs, *freed;
static long nallocs, capallocs, nfreed, capfreed;
static long live_bytes, budget_bytes = 64L << 20, launches;

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

static int rt_cublas_init(void)
{
    const char *path;
    if (cublas_inited) return cublas_inited > 0;
    cublas_inited = -1;
    if (!rt_init()) return 0;
    path = getenv("RTENSOR_CUBLAS");
    if (!path) path = RTENSOR_CUBLAS_DEFAULT;
    cublas_lib = dlopen(path, RTLD_NOW | RTLD_GLOBAL);
    if (!cublas_lib) return 0;
    p_cublasCreate_v2 = (cublasCreate_v2_t)dlsym(cublas_lib, "cublasCreate_v2");
    p_cublasDgemm_v2 = (cublasDgemm_v2_t)dlsym(cublas_lib, "cublasDgemm_v2");
    p_cublasDgemmStridedBatched = (cublasDgemmStridedBatched_t)dlsym(
        cublas_lib, "cublasDgemmStridedBatched");
    if (!p_cublasCreate_v2 || !p_cublasDgemm_v2) return 0;
    if (p_cublasCreate_v2(&cublas_handle) != 0) return 0;
    cublas_inited = 1;
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

RPY_EXPORTED long rt_cuda_alloc(long n, long zero)
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
    if (!p) {
        if (cuMemAlloc(&p, n * sizeof(double)) != CUDA_SUCCESS) return 0;
        push(&allocs, &nallocs, &capallocs, p, n);
    }
    live_bytes += n * sizeof(double);
    if (zero) cuMemsetD8(p, 0, n * sizeof(double));
    return (long)p;
}

RPY_EXPORTED void rt_cuda_free(long dptr, long n)
{
    live_bytes -= n * sizeof(double);
    push(&freed, &nfreed, &capfreed, (CUdeviceptr)dptr, n);
}

RPY_EXPORTED long rt_cuda_launch_count(void)
{
    return launches;
}

RPY_EXPORTED void rt_cuda_set_budget(long bytes)
{
    budget_bytes = bytes;
}

RPY_EXPORTED int rt_cuda_needs_gc(long n)
{
    long i;
    if (live_bytes <= budget_bytes) return 0;
    for (i = 0; i < nfreed; i++)
        if (freed[i].n == n) return 0;
    return 1;
}

RPY_EXPORTED long rt_cuda_upload(double *host, long n)
{
    long p = rt_cuda_alloc(n, 0);
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
    nallocs = nfreed = 0;
    live_bytes = 0;
}

RPY_EXPORTED void rt_cuda_sync(void)
{
    cuCtxSynchronize();
}

RPY_EXPORTED int rt_cuda_launch(long fn, long *inputs, int ninputs, long n,
                                long *outs, int nouts, int threads,
                                long elems_per_block, int shared, int nextra,
                                long cols)
{
    void *params[24];
    void *null = 0;
    long argn = n, argc = cols;
    int i, k = 0;
    unsigned blocks = (unsigned)((n + elems_per_block - 1) / elems_per_block);
    if (!rt_init() || ninputs > 7 || nouts > 8 || nextra > 6) return 0;
    for (i = 0; i < ninputs; i++) params[k++] = &inputs[i];
    for (i = 0; i < nouts; i++) params[k++] = &outs[i];
    params[k++] = &argn;
    params[k++] = &argc;
    for (i = 0; i < nextra; i++) params[k++] = &null;
    launches++;
    return cuLaunchKernel((CUfunction)fn, blocks ? blocks : 1, 1, 1,
                          threads, 1, 1, shared, 0, params, 0) == CUDA_SUCCESS;
}

RPY_EXPORTED int rt_cuda_matmul(long a, long b, long c, long rows,
                                long inner, long cols, long ta, long tb)
{
    double alpha = 1.0, beta = 0.0;
    int ldb = tb ? (int)inner : (int)cols;
    int lda = ta ? (int)rows : (int)inner;
    if (!rt_cublas_init()) return 0;
    return p_cublasDgemm_v2(cublas_handle, tb ? 1 : 0, ta ? 1 : 0,
                            (int)cols, (int)rows, (int)inner, &alpha,
                            (const double *)b, ldb, (const double *)a, lda,
                            &beta, (double *)c, (int)cols) == 0;
}

RPY_EXPORTED int rt_cuda_bmm(long a, long b, long c, long batch, long rows,
                             long inner, long cols, long tb)
{
    double alpha = 1.0, beta = 0.0;
    int ldb = tb ? (int)inner : (int)cols;
    if (!rt_cublas_init() || !p_cublasDgemmStridedBatched) return 0;
    return p_cublasDgemmStridedBatched(cublas_handle, tb ? 1 : 0, 0,
                                       (int)cols, (int)rows, (int)inner, &alpha,
                                       (const double *)b, ldb,
                                       (long long)(inner * cols),
                                       (const double *)a, (int)inner,
                                       (long long)(rows * inner),
                                       &beta, (double *)c, (int)cols,
                                       (long long)(rows * cols),
                                       (int)batch) == 0;
}
