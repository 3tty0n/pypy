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
typedef int (*cublasSgemm_v2_t)(void *handle, int transa, int transb,
    int m, int n, int k, const float *alpha, const float *A, int lda,
    const float *B, int ldb, const float *beta, float *C, int ldc);
typedef int (*cublasSgemmStridedBatched_t)(void *handle, int transa, int transb,
    int m, int n, int k, const float *alpha, const float *A, int lda,
    long long strideA, const float *B, int ldb, long long strideB,
    const float *beta, float *C, int ldc, long long strideC, int batchCount);
typedef int (*cublasHgemm_t)(void *handle, int transa, int transb,
    int m, int n, int k, const unsigned short *alpha,
    const unsigned short *A, int lda, const unsigned short *B, int ldb,
    const unsigned short *beta, unsigned short *C, int ldc);
typedef int (*cublasHgemmStridedBatched_t)(void *handle, int transa, int transb,
    int m, int n, int k, const unsigned short *alpha,
    const unsigned short *A, int lda, long long strideA,
    const unsigned short *B, int ldb, long long strideB,
    const unsigned short *beta, unsigned short *C, int ldc,
    long long strideC, int batchCount);

static void *cublas_lib;
static void *cublas_handle;
static cublasCreate_v2_t p_cublasCreate_v2;
static cublasDgemm_v2_t p_cublasDgemm_v2;
static cublasDgemmStridedBatched_t p_cublasDgemmStridedBatched;
static cublasSgemm_v2_t p_cublasSgemm_v2;
static cublasSgemmStridedBatched_t p_cublasSgemmStridedBatched;
static cublasHgemm_t p_cublasHgemm;
static cublasHgemmStridedBatched_t p_cublasHgemmStridedBatched;
static int cublas_inited;

#if defined(__FLT16_MANT_DIG__)
typedef _Float16 rt_half;
static unsigned short rt_f2h(double v)
{
    rt_half h = (rt_half)v;
    unsigned short r;
    memcpy(&r, &h, 2);
    return r;
}
static double rt_h2f(unsigned short u)
{
    rt_half h;
    memcpy(&h, &u, 2);
    return (double)h;
}
#else
static unsigned short rt_f2h(double v)
{
    float f = (float)v;
    unsigned int b, sign, exp, man, r;
    long shift, round;
    memcpy(&b, &f, 4);
    sign = (b >> 16) & 0x8000u;
    exp = (b >> 23) & 0xffu;
    man = b & 0x7fffffu;
    if (exp == 0xff) return (unsigned short)(sign | 0x7c00u | (man ? 0x200u : 0u));
    if (exp > 142) return (unsigned short)(sign | 0x7c00u);
    if (exp < 100) return (unsigned short)sign;
    if (exp < 113) {
        man |= 0x800000u;
        shift = 126 - exp;
        round = man & ((1L << shift) - 1);
        r = (unsigned int)(man >> shift);
        if (round > (1L << (shift - 1)) ||
            (round == (1L << (shift - 1)) && (r & 1u))) r++;
        return (unsigned short)(sign | r);
    }
    r = ((exp - 112) << 10) | (man >> 13);
    round = man & 0x1fffu;
    if (round > 0x1000u || (round == 0x1000u && (r & 1u))) r++;
    return (unsigned short)(sign | r);
}
static double rt_h2f(unsigned short u)
{
    unsigned int sign = (unsigned int)(u & 0x8000u) << 16;
    unsigned int exp = (u >> 10) & 0x1fu;
    unsigned int man = u & 0x3ffu;
    unsigned int b;
    float f;
    if (exp == 0) {
        if (!man) { memcpy(&f, &sign, 4); return (double)f; }
        exp = 1;
        while (!(man & 0x400u)) { man <<= 1; exp--; }
        man &= 0x3ffu;
    } else if (exp == 0x1f) {
        b = sign | 0x7f800000u | (man << 13);
        memcpy(&f, &b, 4);
        return (double)f;
    }
    b = sign | ((exp + 112) << 23) | (man << 13);
    memcpy(&f, &b, 4);
    return (double)f;
}
#endif

static CUcontext ctx;
static int inited;
typedef struct { CUdeviceptr p; long n; } buf_t;
static buf_t *allocs, *freed;
static long nallocs, capallocs, nfreed, capfreed;
static long live_bytes, budget_bytes = 64L << 20, launches, fresh_since_gc;

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
    p_cublasSgemm_v2 = (cublasSgemm_v2_t)dlsym(cublas_lib, "cublasSgemm_v2");
    p_cublasSgemmStridedBatched = (cublasSgemmStridedBatched_t)dlsym(
        cublas_lib, "cublasSgemmStridedBatched");
    p_cublasHgemm = (cublasHgemm_t)dlsym(cublas_lib, "cublasHgemm");
    p_cublasHgemmStridedBatched = (cublasHgemmStridedBatched_t)dlsym(
        cublas_lib, "cublasHgemmStridedBatched");
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

RPY_EXPORTED long rt_cuda_alloc(long nbytes, long zero)
{
    CUdeviceptr p = 0;
    long i;
    if (!rt_init()) return 0;
    for (i = nfreed - 1; i >= 0; i--) {
        if (freed[i].n == nbytes) {
            p = freed[i].p;
            freed[i] = freed[--nfreed];
            break;
        }
    }
    if (!p) {
        if (cuMemAlloc(&p, nbytes) != CUDA_SUCCESS) return 0;
        push(&allocs, &nallocs, &capallocs, p, nbytes);
        fresh_since_gc++;
    }
    live_bytes += nbytes;
    if (zero) cuMemsetD8(p, 0, nbytes);
    return (long)p;
}

RPY_EXPORTED void rt_cuda_free(long dptr, long nbytes)
{
    live_bytes -= nbytes;
    push(&freed, &nfreed, &capfreed, (CUdeviceptr)dptr, nbytes);
}

RPY_EXPORTED long rt_cuda_launch_count(void)
{
    return launches;
}

RPY_EXPORTED void rt_cuda_set_budget(long bytes)
{
    budget_bytes = bytes;
}

RPY_EXPORTED int rt_cuda_needs_gc(long nbytes)
{
    long i;
    if (live_bytes <= budget_bytes && fresh_since_gc < 64) return 0;
    for (i = 0; i < nfreed; i++)
        if (freed[i].n == nbytes) return 0;
    fresh_since_gc = 0;
    return 1;
}

RPY_EXPORTED long rt_cuda_upload(double *host, long n, long dtype)
{
    long i, p;
    void *staging;
    if (dtype == 0) {
        p = rt_cuda_alloc(n * 8, 0);
        if (p) cuMemcpyHtoD((CUdeviceptr)p, host, n * 8);
        return p;
    }
    p = rt_cuda_alloc(n * (dtype == 1 ? 4 : 2), 0);
    if (!p) return 0;
    staging = malloc(n * (size_t)(dtype == 1 ? 4 : 2));
    if (!staging) return 0;
    if (dtype == 1)
        for (i = 0; i < n; i++) ((float *)staging)[i] = (float)host[i];
    else
        for (i = 0; i < n; i++) ((unsigned short *)staging)[i] = rt_f2h(host[i]);
    cuMemcpyHtoD((CUdeviceptr)p, staging, n * (size_t)(dtype == 1 ? 4 : 2));
    free(staging);
    return p;
}

RPY_EXPORTED int rt_cuda_download(long dptr, double *host, long n, long dtype)
{
    long i;
    void *staging;
    int ok;
    if (dtype == 0)
        return cuMemcpyDtoH(host, (CUdeviceptr)dptr, n * 8) == CUDA_SUCCESS;
    staging = malloc(n * (size_t)(dtype == 1 ? 4 : 2));
    if (!staging) return 0;
    ok = cuMemcpyDtoH(staging, (CUdeviceptr)dptr,
                      n * (size_t)(dtype == 1 ? 4 : 2)) == CUDA_SUCCESS;
    if (ok) {
        if (dtype == 1)
            for (i = 0; i < n; i++) host[i] = (double)((float *)staging)[i];
        else
            for (i = 0; i < n; i++) host[i] = rt_h2f(((unsigned short *)staging)[i]);
    }
    free(staging);
    return ok;
}

RPY_EXPORTED int rt_cuda_copy(long dst, long src, long nbytes)
{
    if (!rt_init()) return 0;
    return cuMemcpyDtoD_v2((CUdeviceptr)dst, (CUdeviceptr)src, nbytes) ==
           CUDA_SUCCESS;
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
                                long inner, long cols, long ta, long tb,
                                long dtype)
{
    double alpha = 1.0, beta = 0.0;
    float alphaf = 1.0f, betaf = 0.0f;
    unsigned short alphah = 0x3c00, betah = 0;
    int ldb = tb ? (int)inner : (int)cols;
    int lda = ta ? (int)rows : (int)inner;
    if (!rt_cublas_init()) return 0;
    if (dtype == 1) {
        if (!p_cublasSgemm_v2) return 0;
        return p_cublasSgemm_v2(cublas_handle, tb ? 1 : 0, ta ? 1 : 0,
                                (int)cols, (int)rows, (int)inner, &alphaf,
                                (const float *)b, ldb, (const float *)a, lda,
                                &betaf, (float *)c, (int)cols) == 0;
    }
    if (dtype == 2) {
        if (!p_cublasHgemm) return 0;
        return p_cublasHgemm(cublas_handle, tb ? 1 : 0, ta ? 1 : 0,
                             (int)cols, (int)rows, (int)inner, &alphah,
                             (const unsigned short *)b, ldb,
                             (const unsigned short *)a, lda,
                             &betah, (unsigned short *)c, (int)cols) == 0;
    }
    return p_cublasDgemm_v2(cublas_handle, tb ? 1 : 0, ta ? 1 : 0,
                            (int)cols, (int)rows, (int)inner, &alpha,
                            (const double *)b, ldb, (const double *)a, lda,
                            &beta, (double *)c, (int)cols) == 0;
}

RPY_EXPORTED int rt_cuda_bmm(long a, long b, long c, long batch, long rows,
                             long inner, long cols, long ta, long tb,
                             long dtype)
{
    double alpha = 1.0, beta = 0.0;
    float alphaf = 1.0f, betaf = 0.0f;
    unsigned short alphah = 0x3c00, betah = 0;
    int ldb = tb ? (int)inner : (int)cols;
    int lda = ta ? (int)rows : (int)inner;
    if (!rt_cublas_init()) return 0;
    if (dtype == 1) {
        if (!p_cublasSgemmStridedBatched) return 0;
        return p_cublasSgemmStridedBatched(
            cublas_handle, tb ? 1 : 0, ta ? 1 : 0,
            (int)cols, (int)rows, (int)inner, &alphaf,
            (const float *)b, ldb, (long long)(inner * cols),
            (const float *)a, lda, (long long)(rows * inner),
            &betaf, (float *)c, (int)cols, (long long)(rows * cols),
            (int)batch) == 0;
    }
    if (dtype == 2) {
        if (!p_cublasHgemmStridedBatched) return 0;
        return p_cublasHgemmStridedBatched(
            cublas_handle, tb ? 1 : 0, ta ? 1 : 0,
            (int)cols, (int)rows, (int)inner, &alphah,
            (const unsigned short *)b, ldb, (long long)(inner * cols),
            (const unsigned short *)a, lda, (long long)(rows * inner),
            &betah, (unsigned short *)c, (int)cols, (long long)(rows * cols),
            (int)batch) == 0;
    }
    if (!p_cublasDgemmStridedBatched) return 0;
    return p_cublasDgemmStridedBatched(cublas_handle, tb ? 1 : 0, ta ? 1 : 0,
                                       (int)cols, (int)rows, (int)inner, &alpha,
                                       (const double *)b, ldb,
                                       (long long)(inner * cols),
                                       (const double *)a, lda,
                                       (long long)(rows * inner),
                                       &beta, (double *)c, (int)cols,
                                       (long long)(rows * cols),
                                       (int)batch) == 0;
}
