from rpython.rlib import rgc
from rpython.rlib.rarithmetic import intmask
from rpython.rtyper.annlowlevel import cast_instance_to_base_ptr
from rpython.rtyper.lltypesystem import lltype
from rpython.rtyper.lltypesystem import rffi
from rpython.translator.tool.cbuild import ExternalCompilationInfo
import os
from rpython.rtensor.core import (F64, HOSTARRAY, SHAPEARRAY, TENSOR, TENSORARRAY, _shape1, nbytes)

class DeviceBuffer(object):
    def __init__(self, dptr, n):
        self.dptr = dptr
        self.n = n

    @rgc.must_be_light_finalizer
    def __del__(self):
        rt_cuda_free(self.dptr, self.n)

def attach_buffer(t, dptr, nb):
    t.dptr = dptr
    t.buf = cast_instance_to_base_ptr(DeviceBuffer(dptr, nb))

def device_tensor(n, dptr, shape=lltype.nullptr(SHAPEARRAY), dtype=F64):
    t = lltype.malloc(TENSOR)
    t.size = n
    t.shape = shape if shape else _shape1(n)
    t.host = lltype.nullptr(HOSTARRAY)
    t.extra = lltype.nullptr(TENSORARRAY)
    t.dtype = dtype
    attach_buffer(t, dptr, nbytes(n, dtype))
    return t

def host(t):
    if not t.host:
        t.host = lltype.malloc(HOSTARRAY, t.size)
        download(t.dptr, t.host, t.dtype)
    return t.host

def dev(t):
    if t.dptr == 0:
        dptr = upload(t.host, t.dtype)
        if dptr != 0:
            attach_buffer(t, dptr, nbytes(t.size, t.dtype))
    return t.dptr

DOUBLEARRAY = rffi.CArray(rffi.DOUBLE)
SIGNEDARRAY = rffi.CArray(lltype.Signed)
_here = os.path.dirname(os.path.abspath(__file__))
CUDA_HOME = os.environ.get('CUDA_HOME', '/usr/local/cuda')
eci = ExternalCompilationInfo(
    include_dirs=[os.path.join(CUDA_HOME, 'include')],
    separate_module_files=[os.path.join(_here, 'cuda.c')],
    post_include_bits=["""
RPY_EXTERN int rt_cuda_available(void);
RPY_EXTERN long rt_cuda_load(const char *ptx, const char *name);
RPY_EXTERN long rt_cuda_alloc(long nbytes, long zero);
RPY_EXTERN long rt_cuda_upload(double *host, long n, long dtype);
RPY_EXTERN int rt_cuda_download(long dptr, double *host, long n, long dtype);
RPY_EXTERN void rt_cuda_reset(void);
RPY_EXTERN void rt_cuda_free(long dptr, long nbytes);
RPY_EXTERN int rt_cuda_copy(long dst, long src, long nbytes);
RPY_EXTERN void rt_cuda_set_budget(long bytes);
RPY_EXTERN long rt_cuda_live_bytes(void);
RPY_EXTERN long rt_cuda_launch_count(void);
RPY_EXTERN int rt_cuda_needs_gc(long nbytes);
RPY_EXTERN void rt_cuda_sync(void);
RPY_EXTERN double rt_cuda_now(void);
RPY_EXTERN int rt_cuda_launch(long fn, long *inputs, int ninputs, long n,
                              long *outs, int nouts, int threads,
                              long elems_per_block, int shared, int nextra,
                              long cols);
RPY_EXTERN int rt_cuda_matmul(long a, long b, long c, long rows, long inner,
                              long cols, long ta, long tb, long dtype);
RPY_EXTERN int rt_cuda_bmm(long a, long b, long c, long batch, long rows,
                           long inner, long cols, long ta, long tb,
                           long dtype);
"""],
    libraries=['cuda', 'dl'])
rt_cuda_load = rffi.llexternal('rt_cuda_load', [rffi.CCHARP, rffi.CCHARP],
                               lltype.Signed, compilation_info=eci,
                                releasegil=False)
rt_cuda_launch = rffi.llexternal(
    'rt_cuda_launch',
    [lltype.Signed, rffi.CArrayPtr(lltype.Signed), rffi.INT, lltype.Signed,
     rffi.CArrayPtr(lltype.Signed), rffi.INT, rffi.INT, lltype.Signed,
     rffi.INT, rffi.INT, lltype.Signed],
    rffi.INT, compilation_info=eci,
                                releasegil=False)
rt_cuda_available = rffi.llexternal('rt_cuda_available', [], rffi.INT,
                                    compilation_info=eci,
                                releasegil=False)
rt_cuda_alloc = rffi.llexternal('rt_cuda_alloc',
                                [lltype.Signed, lltype.Signed],
                                lltype.Signed, compilation_info=eci,
                                releasegil=False)
rt_cuda_upload = rffi.llexternal('rt_cuda_upload',
                                 [rffi.DOUBLEP, lltype.Signed,
                                  lltype.Signed],
                                 lltype.Signed, compilation_info=eci,
                                releasegil=False)
rt_cuda_download = rffi.llexternal('rt_cuda_download',
                                   [lltype.Signed, rffi.DOUBLEP,
                                    lltype.Signed, lltype.Signed],
                                   rffi.INT, compilation_info=eci,
                                releasegil=False)
rt_cuda_reset = rffi.llexternal('rt_cuda_reset', [], lltype.Void,
                                compilation_info=eci,
                                releasegil=False)
rt_cuda_now = rffi.llexternal('rt_cuda_now', [], rffi.DOUBLE,
                              compilation_info=eci, releasegil=False)
rt_cuda_sync = rffi.llexternal('rt_cuda_sync', [], lltype.Void,
                               compilation_info=eci,
                                releasegil=False)
rt_cuda_matmul = rffi.llexternal(
    'rt_cuda_matmul',
    [lltype.Signed, lltype.Signed, lltype.Signed, lltype.Signed,
     lltype.Signed, lltype.Signed, lltype.Signed, lltype.Signed,
     lltype.Signed],
    rffi.INT, compilation_info=eci,
                                releasegil=False)
rt_cuda_bmm = rffi.llexternal(
    'rt_cuda_bmm',
    [lltype.Signed, lltype.Signed, lltype.Signed, lltype.Signed,
     lltype.Signed, lltype.Signed, lltype.Signed, lltype.Signed,
     lltype.Signed, lltype.Signed],
    rffi.INT, compilation_info=eci,
                                releasegil=False)

def _env(name, default):
    value = os.environ.get(name)
    if value is None:
        return default
    return value

def gpu_enabled():
    if os.environ.get('RTENSOR_CPU') is not None:
        return False
    return rffi.cast(lltype.Signed, rt_cuda_available()) != 0

def _upload_impl(hostarray, dtype=F64):
    if not gpu_enabled():
        return 0
    n = len(hostarray)
    buf = lltype.malloc(DOUBLEARRAY, n, flavor='raw')
    for i in range(n):
        buf[i] = hostarray[i]
    dptr = rt_cuda_upload(buf, n, dtype)
    lltype.free(buf, flavor='raw')
    return dptr

def upload(hostarray, dtype=F64):
    t0 = prof_begin()
    r = _upload_impl(hostarray, dtype)
    prof_end(intmask(10), intmask(len(hostarray)), t0)
    return r

def _download_impl(dptr, hostarray, dtype=F64):
    n = len(hostarray)
    buf = lltype.malloc(DOUBLEARRAY, n, flavor='raw')
    rt_cuda_download(dptr, buf, n, dtype)
    for i in range(n):
        hostarray[i] = buf[i]
    lltype.free(buf, flavor='raw')

def download(dptr, hostarray, dtype=F64):
    t0 = prof_begin()
    _download_impl(dptr, hostarray, dtype)
    prof_end(intmask(11), intmask(len(hostarray)), t0)



rt_cuda_free = rffi.llexternal('rt_cuda_free', [lltype.Signed, lltype.Signed],
                               lltype.Void, compilation_info=eci,
                               releasegil=False)

rt_cuda_copy = rffi.llexternal('rt_cuda_copy',
                               [lltype.Signed, lltype.Signed, lltype.Signed],
                               rffi.INT, compilation_info=eci,
                               releasegil=False)

rt_cuda_set_budget = rffi.llexternal('rt_cuda_set_budget', [lltype.Signed],
                                     lltype.Void, compilation_info=eci,
                                     releasegil=False)
rt_cuda_live_bytes = rffi.llexternal('rt_cuda_live_bytes', [], lltype.Signed,
                                     compilation_info=eci, releasegil=False)
rt_cuda_needs_gc = rffi.llexternal('rt_cuda_needs_gc', [lltype.Signed], rffi.INT,
                                   compilation_info=eci, releasegil=False)

rt_cuda_launch_count = rffi.llexternal('rt_cuda_launch_count', [],
                                       lltype.Signed, compilation_info=eci,
                                       releasegil=False)

def launch_count():
    return rt_cuda_launch_count()

class Profile(object):
    def __init__(self):
        self.enabled = False
        self.times = {}
        self.counts = {}
profile = Profile()

def prof_begin():
    if not profile.enabled:
        return 0.0
    rt_cuda_sync()
    return rt_cuda_now()

PROF_NAMES = ['fused', 'per-node', 'matmul', 'bmm', 'im2col', 'col2chw',
              'maxpool2', 'head_split', 'head_merge', 'assign', 'upload',
              'download']

def prof_end(kind, extra, t0):
    if not profile.enabled:
        return
    rt_cuda_sync()
    dt = rt_cuda_now() - t0
    key = kind * 1000000000 + extra
    profile.times[key] = profile.times.get(key, 0.0) + dt
    profile.counts[key] = profile.counts.get(key, 0) + 1

def profile_report():
    if not profile.enabled:
        return
    total = 0.0
    for name in profile.times:
        total += profile.times[name]
    print 'profile total %f s' % total
    for key in profile.times:
        print '%f s %d calls %s %d' % (profile.times[key], profile.counts[key],
                                           PROF_NAMES[key // 1000000000],
                                           key % 1000000000)

def collect_if_needed(nb):
    if rffi.cast(lltype.Signed, rt_cuda_needs_gc(nb)) != 0:
        rgc.collect(0)
        if rffi.cast(lltype.Signed, rt_cuda_needs_gc(nb)) != 0:
            rgc.collect()
            if rffi.cast(lltype.Signed, rt_cuda_needs_gc(nb)) != 0:
                rt_cuda_set_budget(rt_cuda_live_bytes() * 2)

def sync_device():
    rt_cuda_sync()
