from rpython.rlib import jit
from rpython.rlib.rarithmetic import intmask
from rpython.rtyper.lltypesystem import lltype
from rpython.rtyper.lltypesystem import rffi
from rpython.metatensor.core import (NDTYPES, F16, GA_ROWS, GA_COL2CHW, GA_HEADMERGE, GA_HEADSPLIT, GA_IM2COL, GA_IM2COL_NHWC, GA_ROTHALF, GA_MAXPOOL, GA_MAXPOOL_NHWC, HOSTARRAY, NEG_INF, NULLTENSOR, SHAPEARRAY, _shape2, cols, config, nbytes, new_tensor, policy)
from rpython.metatensor.device import (SIGNEDARRAY, collect_if_needed, dev, device_tensor, gpu_enabled, host, prof_begin, prof_end, rt_cuda_alloc, rt_cuda_bmm2, rt_cuda_copy, rt_cuda_free, rt_cuda_launch, rt_cuda_matmul)
from rpython.metatensor.kernels import (gather_kernel)

def matmul_cpu(a, b, rows, cols, inner, ta, tb):
    ha = host(a)
    hb = host(b)
    shape = lltype.malloc(SHAPEARRAY, 2)
    shape[0] = rows
    shape[1] = cols
    r = new_tensor(rows * cols, shape, a.dtype)
    hr = r.host
    for i in range(rows):
        for j in range(cols):
            acc = 0.0
            for k in range(inner):
                if ta:
                    va = ha[k * rows + i]
                else:
                    va = ha[i * inner + k]
                if tb:
                    vb = hb[j * inner + k]
                else:
                    vb = hb[k * cols + j]
                acc += va * vb
            hr[i * cols + j] = acc
    return r

@jit.dont_look_inside
def _tensor_matmul_impl(a, b, rows, cols, inner, ta, tb, tf32):
    if gpu_enabled() and a.dtype == b.dtype:
        dt = a.dtype
        dptr_a = dev(a)
        dptr_b = dev(b)
        if dptr_a != 0 and dptr_b != 0:
            n = rows * cols
            nb = nbytes(n, dt)
            collect_if_needed(nb)
            outptr = rt_cuda_alloc(nb, 0)
            if outptr != 0:
                ok = rffi.cast(lltype.Signed, rt_cuda_matmul(
                    dptr_a, dptr_b, outptr, rows, inner, cols, ta, tb,
                    dt, tf32)) != 0
                if ok:
                    shape = lltype.malloc(SHAPEARRAY, 2)
                    shape[0] = rows
                    shape[1] = cols
                    return device_tensor(n, outptr, shape, dt)
                rt_cuda_free(outptr, nb)
    return matmul_cpu(a, b, rows, cols, inner, ta, tb)

@jit.dont_look_inside
def tensor_matmul(a, b, rows, cols, inner, ta, tb, tf32=0):
    t0 = prof_begin()
    r = _tensor_matmul_impl(a, b, rows, cols, inner, ta, tb, tf32)
    prof_end(intmask(2), intmask(rows * 1000000 + inner * 1000 + cols), t0)
    return r


def _lds(rows, cols, inner, ta, tb, lda, ldb, ldc, sa, sb, sc):
    if lda <= 0:
        lda = rows if ta else inner
    if ldb <= 0:
        ldb = inner if tb else cols
    if ldc <= 0:
        ldc = cols
    if sa <= 0:
        sa = rows * inner
    if sb <= 0:
        sb = inner * cols
    if sc <= 0:
        sc = rows * cols
    return lda, ldb, ldc, sa, sb, sc

def bmm_cpu(a, b, batch, rows, cols, inner, ta, tb, lda, ldb, ldc,
            sa, sb, sc, outn, outrows, oa, ob, oc, nb2=1, sa2=0, sb2=0,
            sc2=0):
    lda, ldb, ldc, sa, sb, sc = _lds(rows, cols, inner, ta, tb, lda, ldb,
                                     ldc, sa, sb, sc)
    ha = host(a)
    hb = host(b)
    r = new_tensor(outn, _shape2(outrows, outn // outrows), a.dtype)
    hr = r.host
    for i in range(outn):
        hr[i] = 0.0
    for g in range(nb2):
        ga = oa + g * sa2
        gb = ob + g * sb2
        gc = oc + g * sc2
        for t in range(batch):
            for i in range(rows):
                for j in range(cols):
                    acc = 0.0
                    for k in range(inner):
                        if ta:
                            ia = ga + t * sa + k * lda + i
                        else:
                            ia = ga + t * sa + i * lda + k
                        if tb:
                            ib = gb + t * sb + j * ldb + k
                        else:
                            ib = gb + t * sb + k * ldb + j
                        assert ia >= 0
                        assert ib >= 0
                        acc += ha[ia] * hb[ib]
                    ic = gc + t * sc + i * ldc + j
                    assert ic >= 0
                    hr[ic] = acc
    return r

@jit.dont_look_inside
def _tensor_bmm_impl(a, b, batch, rows, cols, inner, ta, tb, lda, ldb, ldc,
                     sa, sb, sc, outn, outrows, oa, ob, oc, zc, nb2=1,
                     sa2=0, sb2=0, sc2=0):
    if gpu_enabled() and a.dtype == b.dtype:
        dt = a.dtype
        dptr_a = dev(a)
        dptr_b = dev(b)
        if dptr_a != 0 and dptr_b != 0:
            nb = nbytes(outn, dt)
            collect_if_needed(nb)
            outptr = rt_cuda_alloc(nb, zc)
            if outptr != 0:
                isz = nbytes(1, dt)
                # The outer batch level is a second, independent stride on
                # each operand, which a strided-batched GEMM cannot express
                # (it has one stride per operand).  rt_cuda_bmm2 issues the
                # whole two-level batch as one gemmBatched over a device
                # array of pointers; nb2 == 1 is the single strided call
                # this always was.
                ok = rffi.cast(lltype.Signed, rt_cuda_bmm2(
                    dptr_a + oa * isz, dptr_b + ob * isz, outptr + oc * isz,
                    nb2, sa2, sb2, sc2, batch, rows, inner, cols,
                    ta, tb, dt, lda, ldb, ldc, sa, sb, sc)) != 0
                if ok:
                    return device_tensor(outn, outptr,
                                         _shape2(outrows, outn // outrows),
                                         dt)
                rt_cuda_free(outptr, nb)
    return bmm_cpu(a, b, batch, rows, cols, inner, ta, tb, lda, ldb, ldc,
                   sa, sb, sc, outn, outrows, oa, ob, oc, nb2, sa2, sb2, sc2)

@jit.dont_look_inside
def tensor_bmm(a, b, batch, rows, cols, inner, ta, tb, lda=0, ldb=0, ldc=0,
               sa=0, sb=0, sc=0, outn=0, outrows=0, oa=0, ob=0, oc=0, zc=0,
               nb2=1, sa2=0, sb2=0, sc2=0):
    """nb2 is an outer batch level: nb2 groups of `batch` matrices, the groups
    sa2/sb2/sc2 elements apart.  Attention over a folded batch of sequences
    needs it, because there the batch is two-dimensional - (sequence, head) -
    and the offset b*seq*lda + head*dh is not an arithmetic progression in a
    single index, so no choice of the single stride sa can express it."""
    if outn <= 0:
        outn = nb2 * batch * rows * cols
    if outrows <= 0:
        outrows = nb2 * batch * rows
    t0 = prof_begin()
    r = _tensor_bmm_impl(a, b, batch, rows, cols, inner, ta, tb, lda, ldb,
                         ldc, sa, sb, sc, outn, outrows, oa, ob, oc, zc,
                         nb2, sa2, sb2, sc2)
    prof_end(intmask(3), intmask(rows * 1000000 + inner * 1000 + cols), t0)
    return r


def gather_gpu(op, params, x, outn, shape):
    dt = x.dtype
    kernel = gather_kernel(op, params, dt)
    if kernel.fn == 0:
        return NULLTENSOR
    dptr = dev(x)
    if dptr == 0:
        return NULLTENSOR
    collect_if_needed(nbytes(outn, dt))
    ins = lltype.malloc(SIGNEDARRAY, 1, flavor='raw')
    outs = lltype.malloc(SIGNEDARRAY, 1, flavor='raw')
    ins[0] = dptr
    outs[0] = rt_cuda_alloc(nbytes(outn, dt), 0)
    ok = outs[0] != 0
    if ok:
        ok = rffi.cast(lltype.Signed, rt_cuda_launch(
            kernel.fn, ins, rffi.cast(rffi.INT, 1), outn, outs,
            rffi.cast(rffi.INT, 1), rffi.cast(rffi.INT, kernel.threads),
            config.flat, rffi.cast(rffi.INT, kernel.shared),
            rffi.cast(rffi.INT, kernel.nextra), 0)) != 0
    result = NULLTENSOR
    if ok:
        result = device_tensor(outn, outs[0], shape, dt)
    lltype.free(ins, flavor='raw')
    lltype.free(outs, flavor='raw')
    return result


def rowgather_gpu(table, idx, rows, cols):
    dt = table.dtype
    kernel = gather_kernel(GA_ROWS, [cols], dt)
    if kernel.fn == 0:
        return NULLTENSOR
    tptr = dev(table)
    iptr = dev(idx)
    if tptr == 0 or iptr == 0:
        return NULLTENSOR
    outn = rows * cols
    collect_if_needed(nbytes(outn, dt))
    ins = lltype.malloc(SIGNEDARRAY, 2, flavor='raw')
    outs = lltype.malloc(SIGNEDARRAY, 1, flavor='raw')
    ins[0] = tptr
    ins[1] = iptr
    outs[0] = rt_cuda_alloc(nbytes(outn, dt), 0)
    ok = outs[0] != 0
    if ok:
        ok = rffi.cast(lltype.Signed, rt_cuda_launch(
            kernel.fn, ins, rffi.cast(rffi.INT, 2), outn, outs,
            rffi.cast(rffi.INT, 1), rffi.cast(rffi.INT, kernel.threads),
            config.flat, rffi.cast(rffi.INT, kernel.shared),
            rffi.cast(rffi.INT, kernel.nextra), 0)) != 0
    result = NULLTENSOR
    if ok:
        result = device_tensor(outn, outs[0], _shape2(rows, cols), dt)
    lltype.free(ins, flavor='raw')
    lltype.free(outs, flavor='raw')
    return result


def rowgather_cpu(table, idx, rows, cols):
    ht = host(table)
    hi = host(idx)
    r = new_tensor(rows * cols, _shape2(rows, cols), table.dtype)
    hr = r.host
    for i in range(rows * cols):
        row = int(hi[i // cols])
        assert row >= 0
        hr[i] = ht[row * cols + i % cols]
    return r


@jit.dont_look_inside
def rowgather(table, idx, rows, cols):
    if gpu_enabled() and table.dtype != F16:
        r = rowgather_gpu(table, idx, rows, cols)
        if r:
            return r
    return rowgather_cpu(table, idx, rows, cols)


def im2col_cpu(x, n, c, h, w, k, pad, stride):
    hx = host(x)
    oh = (h + 2 * pad - k) // stride + 1
    ow = (w + 2 * pad - k) // stride + 1
    ohw = oh * ow
    chw = c * h * w
    kk = k * k
    rows = n * ohw
    cols = c * kk
    r = new_tensor(rows * cols, _shape2(rows, cols), x.dtype)
    hr = r.host
    for i in range(rows * cols):
        row = i // cols
        col = i % cols
        img = row // ohw
        pos = row % ohw
        ih = pos // ow * stride + col % kk // k - pad
        iw = pos % ow * stride + col % k - pad
        v = 0.0
        if ih >= 0 and ih < h and iw >= 0 and iw < w:
            idx = img * chw + col // kk * h * w + ih * w + iw
            assert idx >= 0
            v = hx[idx]
        hr[i] = v
    return r


@jit.dont_look_inside
def _im2col_impl(x, c, h, w, k, pad, stride):
    n = x.size // (c * h * w)
    oh = (h + 2 * pad - k) // stride + 1
    ow = (w + 2 * pad - k) // stride + 1
    rows = n * oh * ow
    cols = c * k * k
    if gpu_enabled():
        params = [n]
        params.append(c)
        params.append(h)
        params.append(w)
        params.append(k)
        params.append(pad)
        params.append(stride)
        r = gather_gpu(GA_IM2COL, params, x, rows * cols,
                       _shape2(rows, cols))
        if r:
            return r
    return im2col_cpu(x, n, c, h, w, k, pad, stride)

@jit.dont_look_inside
def im2col(x, c, h, w, k, pad, stride=1):
    t0 = prof_begin()
    r = _im2col_impl(x, c, h, w, k, pad, stride)
    prof_end(intmask(4), intmask(0), t0)
    return r


def im2col_nhwc_cpu(x, n, c, h, w, k, pad, stride):
    hx = host(x)
    oh = (h + 2 * pad - k) // stride + 1
    ow = (w + 2 * pad - k) // stride + 1
    ohw = oh * ow
    kk = k * k
    rows = n * ohw
    cols = kk * c
    r = new_tensor(rows * cols, _shape2(rows, cols), x.dtype)
    hr = r.host
    for i in range(rows * cols):
        row = i // cols
        col = i % cols
        ci = col % c
        rk = col // c
        img = row // ohw
        pos = row % ohw
        ih = pos // ow * stride + rk // k - pad
        iw = pos % ow * stride + rk % k - pad
        v = 0.0
        if ih >= 0 and ih < h and iw >= 0 and iw < w:
            idx = ((img * h + ih) * w + iw) * c + ci
            assert idx >= 0
            v = hx[idx]
        hr[i] = v
    return r


@jit.dont_look_inside
def _im2col_nhwc_impl(x, c, h, w, k, pad, stride):
    n = x.size // (c * h * w)
    oh = (h + 2 * pad - k) // stride + 1
    ow = (w + 2 * pad - k) // stride + 1
    rows = n * oh * ow
    cols = k * k * c
    if gpu_enabled():
        params = [n]
        params.append(c)
        params.append(h)
        params.append(w)
        params.append(k)
        params.append(pad)
        params.append(stride)
        r = gather_gpu(GA_IM2COL_NHWC, params, x, rows * cols,
                       _shape2(rows, cols))
        if r:
            return r
    return im2col_nhwc_cpu(x, n, c, h, w, k, pad, stride)

@jit.dont_look_inside
def im2col_nhwc(x, c, h, w, k, pad, stride=1):
    t0 = prof_begin()
    r = _im2col_nhwc_impl(x, c, h, w, k, pad, stride)
    prof_end(intmask(4), intmask(0), t0)
    return r


def maxpool2_nhwc_cpu(x, n, c, h, w, k, stride, pad):
    hx = host(x)
    oh = (h + 2 * pad - k) // stride + 1
    ow = (w + 2 * pad - k) // stride + 1
    ohw = oh * ow
    rows = n * ohw
    r = new_tensor(rows * c, _shape2(rows, c), x.dtype)
    hr = r.host
    for i in range(rows * c):
        ci = i % c
        row = i // c
        img = row // ohw
        pos = row % ohw
        ph = pos // ow * stride
        pw = pos % ow * stride
        v = NEG_INF
        for a in range(k):
            ih = ph + a - pad
            if ih < 0:
                ih = 0
            elif ih >= h:
                ih = h - 1
            for b in range(k):
                iw = pw + b - pad
                if iw < 0:
                    iw = 0
                elif iw >= w:
                    iw = w - 1
                idx = ((img * h + ih) * w + iw) * c + ci
                assert idx >= 0
                if hx[idx] > v:
                    v = hx[idx]
        hr[i] = v
    return r


@jit.dont_look_inside
def _maxpool2_nhwc_impl(x, c, h, w, k, stride, pad):
    n = x.size // (c * h * w)
    oh = (h + 2 * pad - k) // stride + 1
    ow = (w + 2 * pad - k) // stride + 1
    rows = n * oh * ow
    if gpu_enabled():
        params = [n]
        params.append(c)
        params.append(h)
        params.append(w)
        params.append(k)
        params.append(stride)
        params.append(pad)
        r = gather_gpu(GA_MAXPOOL_NHWC, params, x, rows * c,
                       _shape2(rows, c))
        if r:
            return r
    return maxpool2_nhwc_cpu(x, n, c, h, w, k, stride, pad)

@jit.dont_look_inside
def maxpool2_nhwc(x, c, h, w, k=2, stride=2, pad=0):
    t0 = prof_begin()
    r = _maxpool2_nhwc_impl(x, c, h, w, k, stride, pad)
    prof_end(intmask(6), intmask(0), t0)
    return r


def col2chw_cpu(y, n, hw, o):
    hy = host(y)
    r = new_tensor(n * o * hw, _shape2(n, o * hw), y.dtype)
    hr = r.host
    for i in range(n * o * hw):
        img = i // (o * hw)
        rem = i % (o * hw)
        idx = (img * hw + rem % hw) * o + rem // hw
        assert idx >= 0
        hr[i] = hy[idx]
    return r


@jit.dont_look_inside
def _col2chw_impl(y, n, hw, o):
    outn = n * o * hw
    if gpu_enabled():
        params = [n]
        params.append(hw)
        params.append(o)
        r = gather_gpu(GA_COL2CHW, params, y, outn, _shape2(n, o * hw))
        if r:
            return r
    return col2chw_cpu(y, n, hw, o)

@jit.dont_look_inside
def col2chw(y, n, hw, o):
    t0 = prof_begin()
    r = _col2chw_impl(y, n, hw, o)
    prof_end(intmask(5), intmask(0), t0)
    return r


def maxpool2_cpu(x, n, c, h, w, k, stride, pad):
    hx = host(x)
    oh = (h + 2 * pad - k) // stride + 1
    ow = (w + 2 * pad - k) // stride + 1
    outn = n * c * oh * ow
    r = new_tensor(outn, _shape2(n, c * oh * ow), x.dtype)
    hr = r.host
    for i in range(outn):
        pw = i % ow
        ph = i // ow % oh
        plane = i // (oh * ow)
        v = NEG_INF
        for a in range(k):
            ih = ph * stride + a - pad
            if ih < 0:
                ih = 0
            elif ih >= h:
                ih = h - 1
            for b in range(k):
                iw = pw * stride + b - pad
                if iw < 0:
                    iw = 0
                elif iw >= w:
                    iw = w - 1
                idx = (plane * h + ih) * w + iw
                assert idx >= 0
                if hx[idx] > v:
                    v = hx[idx]
        hr[i] = v
    return r


@jit.dont_look_inside
def _maxpool2_impl(x, c, h, w, k, stride, pad):
    n = x.size // (c * h * w)
    oh = (h + 2 * pad - k) // stride + 1
    ow = (w + 2 * pad - k) // stride + 1
    outn = n * c * oh * ow
    if gpu_enabled():
        params = [n]
        params.append(c)
        params.append(h)
        params.append(w)
        params.append(k)
        params.append(stride)
        params.append(pad)
        r = gather_gpu(GA_MAXPOOL, params, x, outn,
                       _shape2(n, c * oh * ow))
        if r:
            return r
    return maxpool2_cpu(x, n, c, h, w, k, stride, pad)

@jit.dont_look_inside
def maxpool2(x, c, h, w, k=2, stride=2, pad=0):
    t0 = prof_begin()
    r = _maxpool2_impl(x, c, h, w, k, stride, pad)
    prof_end(intmask(6), intmask(0), t0)
    return r


def rot_half_cpu(x, dh):
    hx = host(x)
    n = x.size
    half = dh // 2
    r = new_tensor(n, _shape2(n // cols(x), cols(x)), x.dtype)
    hr = r.host
    for i in range(n):
        idx = i // dh * dh + (i % dh + half) % dh
        assert idx >= 0
        hr[i] = hx[idx]
    return r


@jit.dont_look_inside
def _rot_half_impl(x, dh):
    if gpu_enabled():
        params = [dh]
        r = gather_gpu(GA_ROTHALF, params, x, x.size,
                       _shape2(x.size // cols(x), cols(x)))
        if r:
            return r
    return rot_half_cpu(x, dh)

@jit.dont_look_inside
def rot_half(x, dh):
    t0 = prof_begin()
    r = _rot_half_impl(x, dh)
    prof_end(intmask(10), intmask(0), t0)
    return r


def head_split_cpu(x, rows, dh, heads):
    hx = host(x)
    r = new_tensor(rows * dh * heads, _shape2(heads * rows, dh), x.dtype)
    hr = r.host
    for i in range(rows * dh * heads):
        hi = i // (rows * dh)
        rem = i % (rows * dh)
        idx = rem // dh * (heads * dh) + hi * dh + rem % dh
        assert idx >= 0
        hr[i] = hx[idx]
    return r


def head_merge_cpu(x, rows, dh, heads):
    hx = host(x)
    r = new_tensor(rows * dh * heads, _shape2(rows, heads * dh), x.dtype)
    hr = r.host
    for i in range(rows * dh * heads):
        ri = i // (heads * dh)
        rem = i % (heads * dh)
        idx = rem // dh * (rows * dh) + ri * dh + rem % dh
        assert idx >= 0
        hr[i] = hx[idx]
    return r

@jit.dont_look_inside
def _head_split_impl(x, rows, dh, heads):
    outn = rows * dh * heads
    if gpu_enabled():
        params = [rows]
        params.append(dh)
        params.append(heads)
        r = gather_gpu(GA_HEADSPLIT, params, x, outn,
                       _shape2(heads * rows, dh))
        if r:
            return r
    return head_split_cpu(x, rows, dh, heads)


@jit.dont_look_inside
def _head_merge_impl(x, rows, dh, heads):
    outn = rows * dh * heads
    if gpu_enabled():
        params = [rows]
        params.append(dh)
        params.append(heads)
        r = gather_gpu(GA_HEADMERGE, params, x, outn,
                       _shape2(rows, heads * dh))
        if r:
            return r
    return head_merge_cpu(x, rows, dh, heads)

@jit.dont_look_inside
def head_split(x, rows, dh, heads):
    t0 = prof_begin()
    r = _head_split_impl(x, rows, dh, heads)
    prof_end(intmask(7), intmask(0), t0)
    return r

@jit.dont_look_inside
def head_merge(x, rows, dh, heads):
    t0 = prof_begin()
    r = _head_merge_impl(x, rows, dh, heads)
    prof_end(intmask(8), intmask(0), t0)
    return r


@jit.dont_look_inside
def _tensor_assign_impl(dst, src):
    if dst.dptr != 0 and gpu_enabled():
        dptr_src = dev(src)
        if dptr_src != 0:
            ok = rffi.cast(lltype.Signed, rt_cuda_copy(
                dst.dptr, dptr_src, nbytes(dst.size, dst.dtype))) != 0
            if ok:
                dst.host = lltype.nullptr(HOSTARRAY)
                return dst
    hdst = host(dst)
    hsrc = host(src)
    for i in range(dst.size):
        hdst[i] = hsrc[i]
    return dst

@jit.dont_look_inside
def tensor_assign(dst, src):
    t0 = prof_begin()
    r = _tensor_assign_impl(dst, src)
    prof_end(intmask(9), intmask(0), t0)
    return r


class Ones(object):
    def __init__(self):
        self.n = [-1] * NDTYPES
        self.t = [NULLTENSOR] * NDTYPES
        self.one = [NULLTENSOR] * NDTYPES
        self.cache = {}
ones = Ones()

def _make_ones(n, dtype):
    t = new_tensor(n, lltype.nullptr(SHAPEARRAY), dtype)
    for i in range(n):
        t.host[i] = 1.0
    dev(t)
    return t


class ScalarCache(object):
    def __init__(self):
        self.tensors = [{}, {}, {}]
scalars = ScalarCache()


def scalar(value):
    return scalar_of(value, policy.dtype)


@jit.elidable
def scalar_of(value, dtype):
    t = scalars.tensors[dtype].get(value, NULLTENSOR)
    if not t:
        t = new_tensor(1, lltype.nullptr(SHAPEARRAY), dtype)
        t.host[0] = value
        dev(t)
        scalars.tensors[dtype][value] = t
    return t
