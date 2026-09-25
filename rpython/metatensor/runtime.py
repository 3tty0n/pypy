from rpython.rlib import jit
from rpython.rlib.rarithmetic import intmask
from rpython.rlib.rfloat import INFINITY
from rpython.rlib.rfloat import NAN
from rpython.rlib.rfloat import erf
from rpython.rtyper.lltypesystem import lltype
from rpython.rtyper.lltypesystem import rffi
import math
from rpython.metatensor.core import (ADD, ARITY, B_GE, B_GT, B_KEEP_NZ, B_KEEP_Z, B_LE, B_LT, B_MAX, B_MIN, B_NE, B_POW, BINARY, U_ABS, U_COS, U_ERF, U_FLOOR, U_LOG, U_SIGMOID, U_SIN, U_TANH, UNARY, bc_mode, binary_fn, BC_L_COL, BC_L_ROW, BC_L_SCALAR, BC_R_COL, BC_R_ROW, BC_R_SCALAR, DIV, EQMASK, EXP, GATHER, MAXR, MUL, NDTYPES, NEG_INF, NULLTENSOR, RELU, SHAPEARRAY, SUB, SUM, TENSORARRAY, _shape1, cols, config, gather_changes_shape, gather_shape, nbytes, new_tensor, nvals)
from rpython.metatensor.device import (SIGNEDARRAY, collect_if_needed, dev, device_tensor, host, note_cpu_fallback, prof_begin, prof_end, profile_report, rt_cuda_alloc, rt_cuda_free, rt_cuda_launch, rt_cuda_reset, rt_cuda_warn_arity, rt_cuda_warn_cpu)
from rpython.metatensor.kernels import (has_gather, needs_zero, row_tile, single_kernel)
from rpython.metatensor.devops import (_make_ones, ones, adaptive_avg_pool2d, avg_pool2d, cat, depthwise_conv2d, im2col2, im2col_t, pad, pool_out, strided, take, col2chw, gather_op, head_merge, head_split, im2col, im2col_nhwc, maxpool2, maxpool2_nhwc, rot_half, rowgather, scalar, scalar_of, scalars, tensor_assign, tensor_bmm, tensor_matmul, tensor_write_rows)

def eval_op(opcode, a, b, p):
    if opcode == GATHER:
        return gather_op(a, p)
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

def _log(v):
    if v < 0.0:
        return NAN
    if v == 0.0:
        return NEG_INF
    return math.log(v)

def _unary(fn, v):
    if fn == U_TANH:
        return math.tanh(v)
    if fn == U_SIGMOID:
        return 1.0 / (1.0 + _exp(-v))
    if fn == U_LOG:
        return _log(v)
    if fn == U_ABS:
        return abs(v)
    if fn == U_SIN or fn == U_COS:
        if v - v != 0.0:
            return NAN
        return math.sin(v) if fn == U_SIN else math.cos(v)
    if fn == U_ERF:
        return erf(v)
    if fn == U_FLOOR:
        return math.floor(v)
    return -v

def _pow(x, y):
    try:
        return math.pow(x, y)
    except ValueError:
        return INFINITY if x == 0.0 else NAN
    except OverflowError:
        if x < 0.0 and math.fmod(y, 2.0) != 0.0:
            return NEG_INF
        return INFINITY

def _binary(fn, x, y):
    if fn == B_MAX or fn == B_MIN:
        if x != x or y != y:
            return NAN
        if fn == B_MAX:
            return x if x > y else y
        return x if x < y else y
    if fn == B_POW:
        return _pow(x, y)
    if fn == B_LT:
        t = x < y
    elif fn == B_LE:
        t = x <= y
    elif fn == B_GT:
        t = x > y
    elif fn == B_GE:
        t = x >= y
    elif fn == B_NE:
        t = x != y
    elif fn == B_KEEP_NZ:
        return y if x != 0.0 else 0.0
    elif fn == B_KEEP_Z:
        return y if x == 0.0 else 0.0
    else:
        t = x == y
    return 1.0 if t else 0.0

def eval_op_cpu(opcode, a, b, p):
    note_cpu_fallback()
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
            elif opcode == UNARY:
                hr[i] = _unary(p, v)
            else:
                hr[i] = _sqrt(v)
        return r
    fn = binary_fn(p)
    p = bc_mode(opcode, p)
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
        elif opcode == BINARY:
            hr[i] = _binary(fn, ha[ia], hb[ib])
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

def result_shape(kernel, inputs, n, like, v=-1):
    """Value v's shape (the root's by default), found the way eager
    evaluation finds it: down the big operand of each node, stopping at a
    gather that reshapes."""
    nin = nvals(kernel)
    if v < 0:
        v = nin + len(kernel.nodes) - 1
    while v >= nin:
        node = kernel.nodes[v - nin]
        if node.opcode == GATHER and gather_changes_shape(node.p):
            return gather_shape(node.p, n, like)
        bc = bc_mode(node.opcode, node.p)
        if bc == BC_L_ROW or bc == BC_L_SCALAR or bc == BC_L_COL:
            v = node.b
        else:
            v = node.a
    if v < kernel.ninputs:
        return inputs[v].shape
    return like

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
    elif has_gather(kernel):
        shape = result_shape(kernel, inputs, n, shape)
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
            gathered = has_gather(kernel)
            for k in range(1, nout):
                eshape = inputs[big].shape
                if esizes[k] != n:
                    eshape = _shape1(esizes[k])
                elif gathered:
                    eshape = result_shape(kernel, inputs, n, eshape,
                                          kernel.outputs[k - 1])
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
