#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <dlfcn.h>
#include <stdint.h>

#ifndef RPY_EXPORTED
#  define RPY_EXPORTED extern __attribute__((visibility("default")))
#endif

#ifdef RTENSOR_CUBLAS_PATH
#  define RT_STR_(x) #x
#  define RT_STR(x) RT_STR_(x)
#  define RTENSOR_CUBLAS_DEFAULT RT_STR(RTENSOR_CUBLAS_PATH)
#else
#  define RTENSOR_CUBLAS_DEFAULT "libcublas.so"
#endif

/* ==== cuBLAS entry points, loaded lazily with dlopen ==== */

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
typedef int (*cublasSetMathMode_t)(void *handle, int mode);
static cublasSetMathMode_t p_cublasSetMathMode;
static cublasSgemm_v2_t p_cublasSgemm_v2;
static cublasSgemmStridedBatched_t p_cublasSgemmStridedBatched;
static cublasHgemm_t p_cublasHgemm;
static cublasHgemmStridedBatched_t p_cublasHgemmStridedBatched;
static int cublas_inited;

/* ==== half <-> double conversion ==== */

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

/* ==== device state and initialisation ==== */

#endif

static CUcontext ctx;
static int inited;
typedef struct { CUdeviceptr p; long n; } buf_t;
static buf_t *allocs, *freed;
static long nallocs, capallocs, nfreed, capfreed;
static long live_bytes, budget_bytes = 8L << 20, launches, fresh_since_gc;
static long allocated_since_gc, live_after_gc;
static long count_threshold = 1, just_collected;
#define SLAB_BYTES (32L << 20)
static CUresult rt_last_err;
static CUdeviceptr slab_ptr;
static long slab_left;
static long mem_cap, device_bytes;

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
    p_cublasSetMathMode = (cublasSetMathMode_t)dlsym(cublas_lib,
                                                     "cublasSetMathMode");
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

/* ==== slab allocator: free list, GC trigger, memory cap ==== */

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

static int fell_back_to_cpu;

static long alloc_failed(long nbytes)
{
    if (!fell_back_to_cpu) {
        const char *nm = 0;
        cuGetErrorName(rt_last_err, &nm);
        fprintf(stderr, "metatensor: cuMemAlloc(%ld) failed with %ld MB live (%s), falling back to CPU\n",
                nbytes, live_bytes >> 20, nm ? nm : "?");
    }
    fell_back_to_cpu = 1;
    return 0;
}

RPY_EXPORTED long rt_cuda_alloc_failed(void)
{
    return fell_back_to_cpu;
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
        long need = (nbytes + 511) & ~511L;
        if (need <= SLAB_BYTES / 4) {
            if (slab_left < need) {
                CUdeviceptr s;
                if ((rt_last_err = cuMemAlloc(&s, SLAB_BYTES)) != CUDA_SUCCESS) return alloc_failed(SLAB_BYTES);
                push(&allocs, &nallocs, &capallocs, s, SLAB_BYTES);
                device_bytes += SLAB_BYTES;
                slab_ptr = s;
                slab_left = SLAB_BYTES;
            }
            p = slab_ptr;
            slab_ptr += need;
            slab_left -= need;
        } else {
            if ((rt_last_err = cuMemAlloc(&p, nbytes)) != CUDA_SUCCESS) return alloc_failed(nbytes);
            push(&allocs, &nallocs, &capallocs, p, nbytes);
            device_bytes += nbytes;
        }
        fresh_since_gc++;
        if (live_after_gc < 0) live_after_gc = live_bytes;
        allocated_since_gc += nbytes;
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

RPY_EXPORTED long rt_cuda_live_bytes(void)
{
    return live_bytes;
}

RPY_EXPORTED void rt_cuda_set_budget(long bytes)
{
    budget_bytes = bytes;
}

RPY_EXPORTED long rt_cuda_mem_total(void)
{
    size_t freeb, totalb;
    if (rt_init() && cuMemGetInfo(&freeb, &totalb) == CUDA_SUCCESS)
        return (long)totalb;
    return 0;
}

RPY_EXPORTED int rt_cuda_has_free(long nbytes)
{
    long i;
    for (i = 0; i < nfreed; i++)
        if (freed[i].n == nbytes) return 1;
    return 0;
}

static long rt_mem_cap(void)
{
    size_t freeb, totalb;
    if (!mem_cap) {
        mem_cap = -1;
        if (rt_init() && cuMemGetInfo(&freeb, &totalb) == CUDA_SUCCESS)
            mem_cap = (long)(totalb / 10 * 7);
    }
    return mem_cap;
}

RPY_EXPORTED int rt_cuda_needs_gc(long nbytes)
{
    long threshold = budget_bytes > live_after_gc ? budget_bytes : live_after_gc;
    long cap = rt_mem_cap();
    int reusable = rt_cuda_has_free(nbytes);
    if (just_collected) {
        just_collected = 0;
        if (!reusable && count_threshold < 65536) count_threshold *= 2;
        else if (reusable && count_threshold > 1) count_threshold /= 2;
    }
    if (reusable) return 0;
    if (cap > 0 && device_bytes + nbytes > cap) {
        allocated_since_gc = 0;
        fresh_since_gc = 0;
        live_after_gc = -1;
        just_collected = 1;
        return 1;
    }
    if (allocated_since_gc < threshold && fresh_since_gc < count_threshold) return 0;
    allocated_since_gc = 0;
    fresh_since_gc = 0;
    live_after_gc = -1;
    just_collected = 1;
    return 1;
}

RPY_EXPORTED void rt_cuda_reset(void)
{
    long i;
    for (i = 0; i < nallocs; i++) cuMemFree(allocs[i].p);
    nallocs = nfreed = 0;
    slab_ptr = 0;
    slab_left = 0;
    live_bytes = 0;
}

/* ==== kernel loading and debug names ==== */

RPY_EXPORTED int rt_cuda_available(void)
{
    return rt_init();
}

static struct { long fn; char name[64]; } rt_names[4096];
static int rt_nnames;
static int rt_dbg = -1;
static int dbg_on(void)
{
    if (rt_dbg < 0) rt_dbg = getenv("RTENSOR_DEBUG_LAUNCH") != NULL;
    return rt_dbg;
}
static const char *rt_name_of(long fn)
{
    int i;
    for (i = 0; i < rt_nnames; i++) if (rt_names[i].fn == fn) return rt_names[i].name;
    return "?";
}

static void rt_warn_load(const char *what, const char *name, CUresult r)
{
    static int warned;
    const char *err = 0;
    if (warned) return;
    warned = 1;
    cuGetErrorString(r, &err);
    fprintf(stderr, "metatensor: %s failed for kernel %s: %s\n"
                    "metatensor: every kernel will run on the CPU instead\n",
            what, name, err ? err : "?");
}

RPY_EXPORTED long rt_cuda_load(const char *ptx, const char *name)
{
    CUmodule mod;
    CUfunction fn;
    CUresult r;
    if (!rt_init()) return 0;
    if ((r = cuModuleLoadData(&mod, ptx)) != CUDA_SUCCESS) {
        rt_warn_load("cuModuleLoadData", name, r);
        return 0;
    }
    if ((r = cuModuleGetFunction(&fn, mod, name)) != CUDA_SUCCESS) {
        rt_warn_load("cuModuleGetFunction", name, r);
        return 0;
    }
    if (rt_nnames < 4096) {
        rt_names[rt_nnames].fn = (long)fn;
        snprintf(rt_names[rt_nnames].name, 64, "%s", name);
        rt_nnames++;
    }
    return (long)fn;
}

RPY_EXPORTED void rt_cuda_warn_cpu(long fn)
{
    static int warned;
    if (!warned) {
        warned = 1;
        fprintf(stderr, "metatensor: kernel %s has GPU code but ran on the CPU\n",
                rt_name_of(fn));
    }
}

RPY_EXPORTED void rt_cuda_warn_arity(long compiled, long launched)
{
    static int warned;
    if (!warned) {
        warned = 1;
        fprintf(stderr, "metatensor: kernel compiled for %ld outputs but "
                        "launched with %ld, running on the CPU instead\n",
                compiled, launched);
    }
}

RPY_EXPORTED long rt_cuda_launch_count(void)
{
    return launches;
}

/* ==== host <-> device transfers ==== */

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

/* ==== timing, launch and gather entry points ==== */

RPY_EXPORTED double rt_cuda_now(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
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
    {
        CUresult r = cuLaunchKernel((CUfunction)fn, blocks ? blocks : 1, 1, 1,
                                    threads, 1, 1, shared, 0, params, 0);
        if (dbg_on()) {
            CUresult sr = cuCtxSynchronize();
            fprintf(stderr, "launch %s fn=%ld blocks=%u threads=%d n=%ld elems=%ld cols=%ld nin=%d nout=%d shared=%d nextra=%d -> %d/%d\n",
                    rt_name_of(fn), fn % 1000000000, blocks ? blocks : 1, threads, n,
                    elems_per_block, cols, ninputs, nouts, shared, nextra,
                    (int)r, (int)sr);
        }
        return r == CUDA_SUCCESS;
    }
}

/* Convolution runs its GEMM on the tensor cores in TF32, which is what cuDNN
   does for a conv under torch's default (cudnn.allow_tf32 on, matmul off);
   plain matmuls stay in strict fp32, again as torch does.  RTENSOR_TF32=0
   turns the tensor-core path off everywhere. */
#define RT_TF32_MATH 3
static int tf32_env = -1;

static void rt_set_math(long tf32)
{
    if (!p_cublasSetMathMode) return;
    if (tf32_env < 0) {
        const char *e = getenv("RTENSOR_TF32");
        tf32_env = e ? atoi(e) : 1;
    }
    p_cublasSetMathMode(cublas_handle, (tf32 && tf32_env) ? RT_TF32_MATH : 0);
}

/* ==== cuBLASLt: pick the batched algorithm by measuring it, once per shape ====

   cublasSgemmStridedBatched takes the first algorithm cuBLASLt's heuristic
   offers, and for the attention shapes these models use - a handful of small
   64x64x64 or 197x197x64 products - that is a 128x128-tile kernel that leaves
   the GPU nearly empty: 11 us for 2 MFLOP.  The heuristic list holds better
   entries; only measuring says which.  So the first time a shape appears its
   candidates are timed and the winner is cached with its descriptors, and
   every later call is one cublasLtMatmul with a fixed algorithm.  */

typedef struct { unsigned long long data[8]; } rt_lt_algo;
typedef struct {
    rt_lt_algo algo;
    size_t workspace;
    int state;
    float waves;
    int reserved[4];
} rt_lt_heur;

#define RT_LT_TRANSA 3
#define RT_LT_TRANSB 4
#define RT_LT_BATCH_COUNT 5
#define RT_LT_BATCH_STRIDE 6
#define RT_LT_PREF_WORKSPACE 1
#define RT_LT_R_32F 0
#define RT_LT_COMPUTE_32F 68

typedef int (*ltcreate_t)(void **);
typedef int (*ltdesccreate_t)(void **, int compute, int scale);
typedef int (*ltsetattr_t)(void *, int attr, const void *buf, size_t n);
typedef int (*ltdestroy_t)(void *);
typedef int (*ltlayout_t)(void **, int dtype, unsigned long long rows,
                          unsigned long long cols, long ld);
typedef int (*ltprefcreate_t)(void **);
typedef int (*ltheur_t)(void *lt, void *op, void *a, void *b, void *c, void *d,
                        void *pref, int requested, rt_lt_heur *results,
                        int *returned);
typedef int (*ltmatmul_t)(void *lt, void *op, const void *alpha,
                          const void *A, void *La, const void *B, void *Lb,
                          const void *beta, const void *C, void *Lc,
                          void *D, void *Ld, const rt_lt_algo *algo,
                          void *ws, size_t wsbytes, void *stream);

static void *lt_lib, *lt_handle, *lt_ws;
static size_t lt_ws_bytes = 8u << 20;
static ltdesccreate_t p_ltDescCreate;
static ltsetattr_t p_ltDescSetAttr, p_ltLayoutSetAttr, p_ltPrefSetAttr;
static ltdestroy_t p_ltDescDestroy, p_ltLayoutDestroy, p_ltPrefDestroy;
static ltlayout_t p_ltLayoutCreate;
static ltprefcreate_t p_ltPrefCreate;
static ltheur_t p_ltHeuristic;
static ltmatmul_t p_ltMatmul;
static int lt_inited;

/* ponytail: linear scan over a handful of shapes; a hash table if a model
   ever brings hundreds. */
#define RT_LT_KEYS 12
#define RT_LT_MAX 128
typedef struct {
    long key[RT_LT_KEYS];
    void *op, *La, *Lb, *Lc;
    rt_lt_algo algo;
    int usable;
} rt_lt_entry;
static rt_lt_entry lt_cache[RT_LT_MAX];
static int lt_ncache;

static int rt_lt_init(void)
{
    const char *path;
    char buf[512];
    if (lt_inited) return lt_inited > 0;
    lt_inited = -1;
    if (getenv("RTENSOR_NO_CUBLASLT")) return 0;
    if (!rt_cublas_init()) return 0;
    lt_lib = dlopen("libcublasLt.so.13", RTLD_NOW | RTLD_GLOBAL);
    if (!lt_lib) lt_lib = dlopen("libcublasLt.so.12", RTLD_NOW | RTLD_GLOBAL);
    if (!lt_lib) lt_lib = dlopen("libcublasLt.so", RTLD_NOW | RTLD_GLOBAL);
    if (!lt_lib) {
        /* the venv ships libcublasLt next to libcublas under its own name */
        path = getenv("RTENSOR_CUBLAS");
        if (!path) path = RTENSOR_CUBLAS_DEFAULT;
        {
            const char *base = strstr(path, "libcublas.");
            if (base && (size_t)(base - path) + strlen(base) + 3 < sizeof(buf)) {
                memcpy(buf, path, base - path);
                snprintf(buf + (base - path), sizeof(buf) - (base - path),
                         "libcublasLt.%s", base + strlen("libcublas."));
                lt_lib = dlopen(buf, RTLD_NOW | RTLD_GLOBAL);
            }
        }
    }
    if (!lt_lib) return 0;
    p_ltDescCreate = (ltdesccreate_t)dlsym(lt_lib, "cublasLtMatmulDescCreate");
    p_ltDescSetAttr = (ltsetattr_t)dlsym(lt_lib, "cublasLtMatmulDescSetAttribute");
    p_ltDescDestroy = (ltdestroy_t)dlsym(lt_lib, "cublasLtMatmulDescDestroy");
    p_ltLayoutCreate = (ltlayout_t)dlsym(lt_lib, "cublasLtMatrixLayoutCreate");
    p_ltLayoutSetAttr = (ltsetattr_t)dlsym(lt_lib, "cublasLtMatrixLayoutSetAttribute");
    p_ltLayoutDestroy = (ltdestroy_t)dlsym(lt_lib, "cublasLtMatrixLayoutDestroy");
    p_ltPrefCreate = (ltprefcreate_t)dlsym(lt_lib, "cublasLtMatmulPreferenceCreate");
    p_ltPrefSetAttr = (ltsetattr_t)dlsym(lt_lib, "cublasLtMatmulPreferenceSetAttribute");
    p_ltPrefDestroy = (ltdestroy_t)dlsym(lt_lib, "cublasLtMatmulPreferenceDestroy");
    p_ltHeuristic = (ltheur_t)dlsym(lt_lib, "cublasLtMatmulAlgoGetHeuristic");
    p_ltMatmul = (ltmatmul_t)dlsym(lt_lib, "cublasLtMatmul");
    if (!p_ltDescCreate || !p_ltDescSetAttr || !p_ltLayoutCreate ||
        !p_ltLayoutSetAttr || !p_ltPrefCreate || !p_ltPrefSetAttr ||
        !p_ltHeuristic || !p_ltMatmul) return 0;
    /* cublasLtCreate lives in the same library as the rest of Lt. */
    {
        ltcreate_t create = (ltcreate_t)dlsym(lt_lib, "cublasLtCreate");
        if (!create || create(&lt_handle) != 0) return 0;
    }
    {
        CUdeviceptr p = 0;
        if (cuMemAlloc(&p, lt_ws_bytes) != CUDA_SUCCESS) return 0;
        lt_ws = (void *)p;
    }
    lt_inited = 1;
    return 1;
}

static void rt_lt_destroy(rt_lt_entry *e)
{
    if (e->op && p_ltDescDestroy) p_ltDescDestroy(e->op);
    if (e->La && p_ltLayoutDestroy) p_ltLayoutDestroy(e->La);
    if (e->Lb && p_ltLayoutDestroy) p_ltLayoutDestroy(e->Lb);
    if (e->Lc && p_ltLayoutDestroy) p_ltLayoutDestroy(e->Lc);
    e->op = e->La = e->Lb = e->Lc = 0;
}

/* Descriptors in cuBLASLt's column-major world, matching what
   rt_cuda_bmm hands cublasSgemmStridedBatched: the first operand is our B. */
static int rt_lt_build(rt_lt_entry *e, long batch, long rows, long inner,
                       long cols, long ta, long tb, long lda, long ldb,
                       long ldc, long long sa, long long sb, long long sc)
{
    int compute = RT_LT_COMPUTE_32F, scale = RT_LT_R_32F;
    int opB = tb ? 1 : 0, opA = ta ? 1 : 0;
    int32_t bc = (int32_t)batch;
    if (p_ltDescCreate(&e->op, compute, scale) != 0) return 0;
    if (p_ltDescSetAttr(e->op, RT_LT_TRANSA, &opB, sizeof(opB)) != 0) return 0;
    if (p_ltDescSetAttr(e->op, RT_LT_TRANSB, &opA, sizeof(opA)) != 0) return 0;
    if (p_ltLayoutCreate(&e->La, RT_LT_R_32F,
                         (unsigned long long)(tb ? inner : cols),
                         (unsigned long long)(tb ? cols : inner), ldb) != 0)
        return 0;
    if (p_ltLayoutCreate(&e->Lb, RT_LT_R_32F,
                         (unsigned long long)(ta ? rows : inner),
                         (unsigned long long)(ta ? inner : rows), lda) != 0)
        return 0;
    if (p_ltLayoutCreate(&e->Lc, RT_LT_R_32F, (unsigned long long)cols,
                         (unsigned long long)rows, ldc) != 0)
        return 0;
    if (batch > 1) {
        if (p_ltLayoutSetAttr(e->La, RT_LT_BATCH_COUNT, &bc, sizeof(bc)) != 0 ||
            p_ltLayoutSetAttr(e->Lb, RT_LT_BATCH_COUNT, &bc, sizeof(bc)) != 0 ||
            p_ltLayoutSetAttr(e->Lc, RT_LT_BATCH_COUNT, &bc, sizeof(bc)) != 0 ||
            p_ltLayoutSetAttr(e->La, RT_LT_BATCH_STRIDE, &sb, sizeof(sb)) != 0 ||
            p_ltLayoutSetAttr(e->Lb, RT_LT_BATCH_STRIDE, &sa, sizeof(sa)) != 0 ||
            p_ltLayoutSetAttr(e->Lc, RT_LT_BATCH_STRIDE, &sc, sizeof(sc)) != 0)
            return 0;
    }
    return 1;
}

/* Time every candidate on the real operands and keep the fastest. */
static int rt_lt_tune(rt_lt_entry *e, const float *A, const float *B, float *C)
{
    rt_lt_heur res[8];
    void *pref = 0;
    float alpha = 1.0f, beta = 0.0f;
    int nres = 0, i, j, best = -1;
    double bestt = 0.0;
    if (p_ltPrefCreate(&pref) != 0) return 0;
    if (p_ltPrefSetAttr(pref, RT_LT_PREF_WORKSPACE, &lt_ws_bytes,
                        sizeof(lt_ws_bytes)) != 0) {
        p_ltPrefDestroy(pref);
        return 0;
    }
    if (p_ltHeuristic(lt_handle, e->op, e->La, e->Lb, e->Lc, e->Lc, pref,
                      8, res, &nres) != 0 || nres <= 0) {
        p_ltPrefDestroy(pref);
        return 0;
    }
    p_ltPrefDestroy(pref);
    for (i = 0; i < nres; i++) {
        double t0;
        if (res[i].state != 0) continue;
        for (j = 0; j < 3; j++)
            if (p_ltMatmul(lt_handle, e->op, &alpha, B, e->La, A, e->Lb,
                           &beta, C, e->Lc, C, e->Lc, &res[i].algo,
                           lt_ws, lt_ws_bytes, 0) != 0)
                goto next;
        if (cuCtxSynchronize() != CUDA_SUCCESS) goto next;
        t0 = rt_cuda_now();
        for (j = 0; j < 20; j++)
            p_ltMatmul(lt_handle, e->op, &alpha, B, e->La, A, e->Lb,
                       &beta, C, e->Lc, C, e->Lc, &res[i].algo,
                       lt_ws, lt_ws_bytes, 0);
        cuCtxSynchronize();
        t0 = rt_cuda_now() - t0;
        if (best < 0 || t0 < bestt) { best = i; bestt = t0; }
    next:;
    }
    if (best < 0) return 0;
    e->algo = res[best].algo;
    return 1;
}

static rt_lt_entry *rt_lt_lookup(const long *key, long batch, long rows,
                                 long inner, long cols, long ta, long tb,
                                 long lda, long ldb, long ldc, long long sa,
                                 long long sb, long long sc,
                                 const float *A, const float *B, float *C)
{
    int i, k;
    rt_lt_entry *e;
    for (i = 0; i < lt_ncache; i++) {
        for (k = 0; k < RT_LT_KEYS; k++)
            if (lt_cache[i].key[k] != key[k]) break;
        if (k == RT_LT_KEYS) return &lt_cache[i];
    }
    if (lt_ncache >= RT_LT_MAX) return 0;
    e = &lt_cache[lt_ncache++];
    for (k = 0; k < RT_LT_KEYS; k++) e->key[k] = key[k];
    e->usable = 0;
    if (rt_lt_build(e, batch, rows, inner, cols, ta, tb, lda, ldb, ldc,
                    sa, sb, sc) && rt_lt_tune(e, A, B, C))
        e->usable = 1;
    else
        rt_lt_destroy(e);
    return e;
}

/* Returns 0 when Lt cannot serve this call and the legacy path should. */
static int rt_lt_bmm(long a, long b, long c, long batch, long rows,
                     long inner, long cols, long ta, long tb, long lda,
                     long ldb, long ldc, long long sa, long long sb,
                     long long sc)
{
    long key[RT_LT_KEYS];
    rt_lt_entry *e;
    float alpha = 1.0f, beta = 0.0f;
    if (!rt_lt_init()) return 0;
    key[0] = batch; key[1] = rows; key[2] = inner; key[3] = cols;
    key[4] = ta; key[5] = tb; key[6] = lda; key[7] = ldb; key[8] = ldc;
    key[9] = (long)sa; key[10] = (long)sb; key[11] = (long)sc;
    e = rt_lt_lookup(key, batch, rows, inner, cols, ta, tb, lda, ldb, ldc,
                     sa, sb, sc, (const float *)a, (const float *)b,
                     (float *)c);
    if (!e || !e->usable) return 0;
    return p_ltMatmul(lt_handle, e->op, &alpha, (const void *)b, e->La,
                      (const void *)a, e->Lb, &beta, (const void *)c, e->Lc,
                      (void *)c, e->Lc, &e->algo, lt_ws, lt_ws_bytes, 0) == 0;
}

/* ==== cuBLAS gemm ==== */

RPY_EXPORTED int rt_cuda_matmul(long a, long b, long c, long rows,
                                long inner, long cols, long ta, long tb,
                                long dtype, long tf32)
{
    double alpha = 1.0, beta = 0.0;
    float alphaf = 1.0f, betaf = 0.0f;
    unsigned short alphah = 0x3c00, betah = 0;
    int ldb = tb ? (int)inner : (int)cols;
    int lda = ta ? (int)rows : (int)inner;
    if (!rt_cublas_init()) return 0;
    if (dtype == 1) {
        if (!p_cublasSgemm_v2) return 0;
        rt_set_math(tf32);
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
                             long dtype, long lda_, long ldb_, long ldc_,
                             long sa, long sb, long sc)
{
    double alpha = 1.0, beta = 0.0;
    float alphaf = 1.0f, betaf = 0.0f;
    unsigned short alphah = 0x3c00, betah = 0;
    int ldb = ldb_ > 0 ? (int)ldb_ : (tb ? (int)inner : (int)cols);
    int lda = lda_ > 0 ? (int)lda_ : (ta ? (int)rows : (int)inner);
    int ldc = ldc_ > 0 ? (int)ldc_ : (int)cols;
    long long stra = sa > 0 ? (long long)sa : (long long)(rows * inner);
    long long strb = sb > 0 ? (long long)sb : (long long)(inner * cols);
    long long strc = sc > 0 ? (long long)sc : (long long)(rows * cols);
    if (!rt_cublas_init()) return 0;
    if (dtype == 1) {
        if (rt_lt_bmm(a, b, c, batch, rows, inner, cols, ta, tb, lda, ldb,
                      ldc, stra, strb, strc))
            return 1;
        if (!p_cublasSgemmStridedBatched) return 0;
        /* a preceding convolution may have left the handle in TF32 */
        rt_set_math(0);
        return p_cublasSgemmStridedBatched(
            cublas_handle, tb ? 1 : 0, ta ? 1 : 0,
            (int)cols, (int)rows, (int)inner, &alphaf,
            (const float *)b, ldb, strb,
            (const float *)a, lda, stra,
            &betaf, (float *)c, ldc, strc,
            (int)batch) == 0;
    }
    if (dtype == 2) {
        if (!p_cublasHgemmStridedBatched) return 0;
        return p_cublasHgemmStridedBatched(
            cublas_handle, tb ? 1 : 0, ta ? 1 : 0,
            (int)cols, (int)rows, (int)inner, &alphah,
            (const unsigned short *)b, ldb, strb,
            (const unsigned short *)a, lda, stra,
            &betah, (unsigned short *)c, ldc, strc,
            (int)batch) == 0;
    }
    if (!p_cublasDgemmStridedBatched) return 0;
    return p_cublasDgemmStridedBatched(cublas_handle, tb ? 1 : 0, ta ? 1 : 0,
                                       (int)cols, (int)rows, (int)inner, &alpha,
                                       (const double *)b, ldb,
                                       strb,
                                       (const double *)a, lda,
                                       stra,
                                       &beta, (double *)c, ldc,
                                       strc,
                                       (int)batch) == 0;
}
