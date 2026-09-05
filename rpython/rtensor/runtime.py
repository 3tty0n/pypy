from rpython.rlib import jit
from rpython.rlib.rarithmetic import intmask
from rpython.rlib.rfloat import INFINITY
from rpython.rlib.rfloat import NAN
from rpython.rtyper.lltypesystem import lltype
from rpython.rtyper.lltypesystem import rffi
import math
from rpython.rtensor.core import (ADD, ARITY, BC_L_COL, BC_L_ROW, BC_L_SCALAR, BC_R_COL, BC_R_ROW, BC_R_SCALAR, DIV, EQMASK, EXP, GA_COL2CHW, GA_HEADMERGE, GA_HEADSPLIT, GA_IM2COL, GA_MAXPOOL, HOSTARRAY, MAXR, MUL, NDTYPES, NEG_INF, NULLTENSOR, RELU, SHAPEARRAY, SUB, SUM, TENSORARRAY, _shape1, _shape2, cols, config, nbytes, new_tensor, policy)
from rpython.rtensor.device import (SIGNEDARRAY, collect_if_needed, dev, device_tensor, gpu_enabled, host, prof_begin, prof_end, profile_report, rt_cuda_alloc, rt_cuda_bmm, rt_cuda_copy, rt_cuda_free, rt_cuda_launch, rt_cuda_matmul, rt_cuda_reset)
from rpython.rtensor.kernels import (gather_kernel, needs_zero, row_tile, single_kernel)

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


@jit.elidable
def scalar(value):
    dtype = policy.dtype
    t = scalars.tensors[dtype].get(value, NULLTENSOR)
    if not t:
        t = new_tensor(1, lltype.nullptr(SHAPEARRAY), dtype)
        t.host[0] = value
        dev(t)
        scalars.tensors[dtype][value] = t
    return t


def eval_op(opcode, a, b, p):
    kernel = single_kernel(opcode, p, a.dtype)
    if kernel.fn != 0:
        inputs = [a]
        if ARITY[opcode] == 2:
            inputs.append(b)
        r = launch_gpu(kernel, inputs)
        if r:
            return r
    return eval_op_cpu(opcode, a, b, p)

def _exp(v):
    if v > 709.0:
        return INFINITY
    return math.exp(v)

def _sqrt(v):
    if v < 0.0:
        return NAN
    return math.sqrt(v)

def _div(x, y):
    if y == 0.0:
        if x == 0.0:
            return NAN
        return INFINITY if x > 0.0 else NEG_INF
    return x / y

def eval_op_cpu(opcode, a, b, p):
    if opcode == SUM or opcode == MAXR:
        return reduce_cpu(opcode, a, p)
    ha = host(a)
    if ARITY[opcode] == 1:
        n = a.size
        r = new_tensor(n, a.shape, a.dtype)
        hr = r.host
        for i in range(n):
            v = ha[i]
            if opcode == RELU:
                hr[i] = v if v > 0.0 else 0.0
            elif opcode == EXP:
                hr[i] = _exp(v)
            else:
                hr[i] = _sqrt(v)
        return r
    big = a
    if p == BC_L_ROW or p == BC_L_SCALAR or p == BC_L_COL:
        big = b
    n = big.size
    c = cols(big)
    hb = host(b)
    r = new_tensor(n, big.shape, big.dtype)
    hr = r.host
    for i in range(n):
        ia = i
        ib = i
        if p == BC_R_ROW:
            ib = i % c
        elif p == BC_R_SCALAR:
            ib = 0
        elif p == BC_R_COL:
            ib = i // c
        elif p == BC_L_ROW:
            ia = i % c
        elif p == BC_L_SCALAR:
            ia = 0
        elif p == BC_L_COL:
            ia = i // c
        assert ia >= 0
        assert ib >= 0
        if opcode == ADD:
            hr[i] = ha[ia] + hb[ib]
        elif opcode == MUL:
            hr[i] = ha[ia] * hb[ib]
        elif opcode == SUB:
            hr[i] = ha[ia] - hb[ib]
        elif opcode == DIV:
            hr[i] = _div(ha[ia], hb[ib])
        elif opcode == EQMASK:
            hr[i] = 1.0 if ha[ia] == hb[ib] else 0.0
        else:
            hr[i] = hb[ib] if ha[ia] > 0.0 else 0.0
    return r

def reduce_cpu(opcode, a, axis):
    ha = host(a)
    n = a.size
    c = cols(a)
    if axis == 0:
        m = c
    elif axis == 1:
        m = n // c
    else:
        m = 1
    if m <= 0:
        m = 1
    r = new_tensor(m, lltype.nullptr(SHAPEARRAY), a.dtype)
    hr = r.host
    for i in range(m):
        hr[i] = 0.0 if opcode == SUM else NEG_INF
    for i in range(n):
        if axis == 0:
            k = i % c
        elif axis == 1:
            k = i // c
        else:
            k = 0
        if k >= m:
            k = m - 1
        assert k >= 0
        if opcode == SUM:
            hr[k] += ha[i]
        elif ha[i] > hr[k]:
            hr[k] = ha[i]
    return r

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
def _tensor_matmul_impl(a, b, rows, cols, inner, ta, tb):
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
                    dt)) != 0
                if ok:
                    shape = lltype.malloc(SHAPEARRAY, 2)
                    shape[0] = rows
                    shape[1] = cols
                    return device_tensor(n, outptr, shape, dt)
                rt_cuda_free(outptr, nb)
    return matmul_cpu(a, b, rows, cols, inner, ta, tb)

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

def launch(kernel, a, b, c, d=NULLTENSOR, e=NULLTENSOR, f=NULLTENSOR):
    values = [a]
    if kernel.ninputs > 1:
        values.append(b)
    if kernel.ninputs > 2:
        values.append(c)
    if kernel.ninputs > 3:
        values.append(d)
    if kernel.ninputs > 4:
        values.append(e)
    if kernel.ninputs > 5:
        values.append(f)
    nmax = 0
    for k in range(len(values)):
        if values[k].size > nmax:
            nmax = values[k].size
    if kernel.fn != 0 and (kernel.n == 0 or kernel.n == nmax):
        t0 = prof_begin()
        r = launch_gpu(kernel, values)
        if r:
            prof_end(intmask(0), intmask(kernel.fn % 1000000000), t0)
            return r
    t0 = prof_begin()
    nodes = kernel.nodes
    for i in range(len(nodes)):
        node = nodes[i]
        opcode = node.opcode
        assert opcode >= 0
        right = values[node.b] if node.b >= 0 else NULLTENSOR
        values.append(eval_op(opcode, values[node.a], right, node.p))
    result = values[len(values) - 1]
    prof_end(intmask(1), intmask(len(kernel.nodes)), t0)
    nout = len(kernel.outputs)
    if nout > 0:
        result.extra = lltype.malloc(TENSORARRAY, nout)
        for k in range(nout):
            result.extra[k] = values[kernel.outputs[k]]
    return result

def reset_device():
    profile_report()
    ones.cache.clear()
    for i in range(NDTYPES):
        scalars.tensors[i].clear()
        ones.n[i] = -1
        ones.t[i] = NULLTENSOR
        ones.one[i] = NULLTENSOR
    rt_cuda_reset()

def launch_gpu(kernel, inputs):
    nin = len(inputs)
    dt = kernel.dtype
    for k in range(nin):
        if inputs[k].dtype != dt:
            return NULLTENSOR
    big = 0
    for k in range(1, nin):
        if inputs[k].size > inputs[big].size:
            big = k
    n = inputs[big].size
    c = cols(inputs[big])
    outlen = n
    shape = inputs[big].shape
    elems = config.block
    if kernel.rowmode:
        if c <= 0 or c > row_tile(kernel) or n % c != 0:
            return NULLTENSOR
        if kernel.cols > 0 and c != kernel.cols:
            return NULLTENSOR
        elems = c
    if kernel.sumroot:
        axis = kernel.nodes[len(kernel.nodes) - 1].p
        if axis == 0:
            outlen = c
        elif axis == 1:
            outlen = n // c
        else:
            outlen = 1
        if outlen <= 0:
            outlen = 1
        shape = _shape1(outlen)
    nout = 1 + len(kernel.outputs)
    collect_if_needed(nbytes(n, dt))
    dptrs = lltype.malloc(SIGNEDARRAY, nin, flavor='raw')
    outs = lltype.malloc(SIGNEDARRAY, nout, flavor='raw')
    ok = True
    for k in range(nin):
        dptrs[k] = dev(inputs[k])
        if dptrs[k] == 0:
            ok = False
    outs[0] = rt_cuda_alloc(nbytes(outlen, dt),
                            needs_zero(kernel)) if ok else 0
    if outs[0] == 0:
        ok = False
    for k in range(1, nout):
        outs[k] = rt_cuda_alloc(nbytes(n, dt), 0) if ok else 0
        if outs[k] == 0:
            ok = False
    if ok:
        ok = rffi.cast(lltype.Signed, rt_cuda_launch(
            kernel.fn, dptrs, rffi.cast(rffi.INT, nin), n,
            outs, rffi.cast(rffi.INT, nout),
            rffi.cast(rffi.INT, kernel.threads), elems,
            rffi.cast(rffi.INT, kernel.shared),
            rffi.cast(rffi.INT, kernel.nextra), c)) != 0
    result = NULLTENSOR
    if ok:
        result = device_tensor(outlen, outs[0], shape, dt)
        if nout > 1:
            result.extra = lltype.malloc(TENSORARRAY, nout - 1)
            for k in range(1, nout):
                result.extra[k - 1] = device_tensor(n, outs[k],
                                                    inputs[big].shape, dt)
    lltype.free(dptrs, flavor='raw')
    lltype.free(outs, flavor='raw')
    return result


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
            config.block, rffi.cast(rffi.INT, kernel.shared),
            rffi.cast(rffi.INT, kernel.nextra), 0)) != 0
    result = NULLTENSOR
    if ok:
        result = device_tensor(outn, outs[0], shape, dt)
    lltype.free(ins, flavor='raw')
    lltype.free(outs, flavor='raw')
    return result


def im2col_cpu(x, n, c, h, w, k, pad):
    hx = host(x)
    hw = h * w
    chw = c * hw
    kk = k * k
    rows = n * hw
    cols = c * kk
    r = new_tensor(rows * cols, _shape2(rows, cols), x.dtype)
    hr = r.host
    for i in range(rows * cols):
        row = i // cols
        col = i % cols
        img = row // hw
        pos = row % hw
        ih = pos // w + col % kk // k - pad
        iw = pos % w + col % k - pad
        v = 0.0
        if ih >= 0 and ih < h and iw >= 0 and iw < w:
            idx = img * chw + col // kk * hw + ih * w + iw
            assert idx >= 0
            v = hx[idx]
        hr[i] = v
    return r


@jit.dont_look_inside
def _im2col_impl(x, c, h, w, k, pad):
    n = x.size // (c * h * w)
    rows = n * h * w
    cols = c * k * k
    if gpu_enabled():
        params = [n]
        params.append(c)
        params.append(h)
        params.append(w)
        params.append(k)
        params.append(pad)
        r = gather_gpu(GA_IM2COL, params, x, rows * cols,
                       _shape2(rows, cols))
        if r:
            return r
    return im2col_cpu(x, n, c, h, w, k, pad)


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


def maxpool2_cpu(x, n, c, h, w):
    hx = host(x)
    oh = h // 2
    ow = w // 2
    outn = n * c * oh * ow
    r = new_tensor(outn, _shape2(n, c * oh * ow), x.dtype)
    hr = r.host
    for i in range(outn):
        q = i // ow
        base = (i // (oh * ow) * h + q % oh * 2) * w + i % ow * 2
        assert base >= 0
        v = hx[base]
        if hx[base + 1] > v:
            v = hx[base + 1]
        if hx[base + w] > v:
            v = hx[base + w]
        if hx[base + w + 1] > v:
            v = hx[base + w + 1]
        hr[i] = v
    return r


@jit.dont_look_inside
def _maxpool2_impl(x, c, h, w):
    n = x.size // (c * h * w)
    outn = n * c * (h // 2) * (w // 2)
    if gpu_enabled():
        params = [n]
        params.append(c)
        params.append(h)
        params.append(w)
        r = gather_gpu(GA_MAXPOOL, params, x, outn,
                       _shape2(n, c * (h // 2) * (w // 2)))
        if r:
            return r
    return maxpool2_cpu(x, n, c, h, w)


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


def bmm_cpu(a, b, batch, rows, cols, inner, ta, tb):
    ha = host(a)
    hb = host(b)
    r = new_tensor(batch * rows * cols, _shape2(batch * rows, cols), a.dtype)
    hr = r.host
    for t in range(batch):
        abase = t * rows * inner
        bbase = t * inner * cols
        cbase = t * rows * cols
        for i in range(rows):
            for j in range(cols):
                acc = 0.0
                for k in range(inner):
                    if tb:
                        vb = hb[bbase + j * inner + k]
                    else:
                        vb = hb[bbase + k * cols + j]
                    if ta:
                        va = ha[abase + k * rows + i]
                    else:
                        va = ha[abase + i * inner + k]
                    acc += va * vb
                hr[cbase + i * cols + j] = acc
    return r


@jit.dont_look_inside
def _tensor_bmm_impl(a, b, batch, rows, cols, inner, ta, tb):
    if gpu_enabled() and a.dtype == b.dtype:
        dt = a.dtype
        dptr_a = dev(a)
        dptr_b = dev(b)
        if dptr_a != 0 and dptr_b != 0:
            n = batch * rows * cols
            nb = nbytes(n, dt)
            collect_if_needed(nb)
            outptr = rt_cuda_alloc(nb, 0)
            if outptr != 0:
                ok = rffi.cast(lltype.Signed, rt_cuda_bmm(
                    dptr_a, dptr_b, outptr, batch, rows, inner, cols,
                    ta, tb, dt)) != 0
                if ok:
                    return device_tensor(n, outptr,
                                         _shape2(batch * rows, cols), dt)
                rt_cuda_free(outptr, nb)
    return bmm_cpu(a, b, batch, rows, cols, inner, ta, tb)

@jit.dont_look_inside
def tensor_matmul(a, b, rows, cols, inner, ta, tb):
    t0 = prof_begin()
    r = _tensor_matmul_impl(a, b, rows, cols, inner, ta, tb)
    prof_end(intmask(2), intmask(rows * 1000000 + inner * 1000 + cols), t0)
    return r

@jit.dont_look_inside
def tensor_bmm(a, b, batch, rows, cols, inner, ta, tb):
    t0 = prof_begin()
    r = _tensor_bmm_impl(a, b, batch, rows, cols, inner, ta, tb)
    prof_end(intmask(3), intmask(rows * 1000000 + inner * 1000 + cols), t0)
    return r

@jit.dont_look_inside
def im2col(x, c, h, w, k, pad):
    t0 = prof_begin()
    r = _im2col_impl(x, c, h, w, k, pad)
    prof_end(intmask(4), intmask(0), t0)
    return r

@jit.dont_look_inside
def col2chw(y, n, hw, o):
    t0 = prof_begin()
    r = _col2chw_impl(y, n, hw, o)
    prof_end(intmask(5), intmask(0), t0)
    return r

@jit.dont_look_inside
def maxpool2(x, c, h, w):
    t0 = prof_begin()
    r = _maxpool2_impl(x, c, h, w)
    prof_end(intmask(6), intmask(0), t0)
    return r

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
def tensor_assign(dst, src):
    t0 = prof_begin()
    r = _tensor_assign_impl(dst, src)
    prof_end(intmask(9), intmask(0), t0)
    return r
