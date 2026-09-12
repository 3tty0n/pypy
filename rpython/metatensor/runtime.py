from rpython.rlib import jit
from rpython.rlib.rarithmetic import intmask
from rpython.rlib.rfloat import INFINITY
from rpython.rlib.rfloat import NAN
from rpython.rtyper.lltypesystem import lltype
from rpython.rtyper.lltypesystem import rffi
import math
from rpython.metatensor.core import (ADD, ARITY, BC_L_COL, BC_L_ROW, BC_L_SCALAR, BC_R_COL, BC_R_ROW, BC_R_SCALAR, DIV, EQMASK, EXP, MAXR, MUL, NDTYPES, NEG_INF, NULLTENSOR, RELU, SHAPEARRAY, SUB, SUM, TENSORARRAY, _shape1, cols, config, nbytes, new_tensor)
from rpython.metatensor.device import (SIGNEDARRAY, collect_if_needed, dev, device_tensor, host, prof_begin, prof_end, profile_report, rt_cuda_alloc, rt_cuda_free, rt_cuda_launch, rt_cuda_reset, rt_cuda_warn_arity, rt_cuda_warn_cpu)
from rpython.metatensor.kernels import (needs_zero, row_tile, single_kernel)
from rpython.metatensor.devops import (_make_ones, ones, col2chw, head_merge, head_split, im2col, im2col_nhwc, maxpool2, maxpool2_nhwc, rot_half, rowgather, scalar, scalar_of, scalars, tensor_assign, tensor_bmm, tensor_matmul)

def eval_op(opcode, a, b, p):
    kernel = single_kernel(opcode, p, a.dtype)
    if kernel.fn != 0:
        inputs = [a]
        if ARITY[opcode] == 2:
            inputs.append(b)
        r = launch_gpu(kernel, inputs)
        if r:
            return r
        rt_cuda_warn_cpu(kernel.fn)
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

def extra_size(kernel, k, n, c):
    mode = (kernel.outmodes >> (2 * k)) & 3
    if mode == 1:
        m = c
    elif mode == 2:
        m = 1
    elif mode == 3:
        m = n // c if c > 0 else n
    else:
        m = n
    if m <= 0:
        m = 1
    return m

def arity_mismatch(kernel):
    """True when kernel.fn was compiled for a different number of outputs.

    The launcher packs the output pointers ahead of n and cols, so one output
    too many silently shifts cols into a device pointer and the kernel walks
    off the buffer.  Refuse the launch and let the caller run on the CPU.
    """
    if kernel.nouts == 1 + len(kernel.outputs):
        return False
    rt_cuda_warn_arity(kernel.nouts, 1 + len(kernel.outputs))
    return True

@jit.unroll_safe
def modes_fit(kernel, inputs, n, c):
    packed = kernel.modes
    for k in range(len(inputs)):
        mode = (packed >> (2 * k)) & 3
        if mode == 1:
            need = c
        elif mode == 2:
            need = 1
        elif mode == 3:
            need = n // c if c > 0 else n
        else:
            need = n
        if inputs[k].size < need:
            return False
    return True

def launch_gpu(kernel, inputs):
    nin = len(inputs)
    dt = kernel.dtype
    if arity_mismatch(kernel):
        return NULLTENSOR
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
    elems = config.flat
    if kernel.rowmode:
        if c <= 0 or c > row_tile(kernel) or n % c != 0:
            return NULLTENSOR
        if kernel.cols > 0 and c != kernel.cols:
            return NULLTENSOR
        elems = c
    if not modes_fit(kernel, inputs, n, c):
        return NULLTENSOR
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
    esizes = lltype.malloc(SIGNEDARRAY, nout, flavor='raw')
    esizes[0] = outlen
    outs[0] = rt_cuda_alloc(nbytes(outlen, dt),
                            needs_zero(kernel)) if ok else 0
    if outs[0] == 0:
        ok = False
    for k in range(1, nout):
        esizes[k] = extra_size(kernel, k - 1, n, c)
        outs[k] = rt_cuda_alloc(nbytes(esizes[k], dt), 0) if ok else 0
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
    if not ok:
        # An allocation or the launch failed part way through.  Nothing owns
        # the buffers we did get, so hand them back instead of leaking them
        # and pushing the next allocation closer to the same failure.
        for k in range(nout):
            if outs[k] != 0:
                rt_cuda_free(outs[k], nbytes(esizes[k], dt))
    if ok:
        result = device_tensor(outlen, outs[0], shape, dt)
        if nout > 1:
            result.extra = lltype.malloc(TENSORARRAY, nout - 1)
            for k in range(1, nout):
                eshape = inputs[big].shape
                if esizes[k] != n:
                    eshape = _shape1(esizes[k])
                result.extra[k - 1] = device_tensor(esizes[k], outs[k],
                                                    eshape, dt)
    lltype.free(esizes, flavor='raw')
    lltype.free(dptrs, flavor='raw')
    lltype.free(outs, flavor='raw')
    return result

def launch(kernel, a, b, c, d=NULLTENSOR, e=NULLTENSOR, f=NULLTENSOR,
           g=NULLTENSOR, h=NULLTENSOR):
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
    if kernel.ninputs > 6:
        values.append(g)
    if kernel.ninputs > 7:
        values.append(h)
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
    if kernel.fn != 0:
        rt_cuda_warn_cpu(kernel.fn)
    t0 = prof_begin()
    for j in range(len(kernel.consts)):
        c = new_tensor(1, lltype.nullptr(SHAPEARRAY), kernel.dtype)
        c.host[0] = kernel.consts[j]
        values.append(c)
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
