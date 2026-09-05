import math
import os
from rpython.rlib import jit, rgc
from rpython.rlib.rfloat import INFINITY, NAN
from rpython.rtyper.annlowlevel import cast_instance_to_base_ptr
from rpython.rtyper.rclass import OBJECTPTR
from rpython.rtyper.lltypesystem import lltype, rffi
from rpython.translator.tool.cbuild import ExternalCompilationInfo

HOSTARRAY = lltype.GcArray(lltype.Float)
SHAPEARRAY = lltype.GcArray(lltype.Signed)
TENSOR = lltype.GcForwardReference()
TENSORPTR = lltype.Ptr(TENSOR)
TENSORARRAY = lltype.GcArray(TENSORPTR)
TENSOR.become(lltype.GcStruct('TENSOR', ('size', lltype.Signed),
                              ('shape', lltype.Ptr(SHAPEARRAY)),
                              ('dptr', lltype.Signed),
                              ('host', lltype.Ptr(HOSTARRAY)),
                              ('extra', lltype.Ptr(TENSORARRAY)),
                              ('buf', OBJECTPTR)))
NULLTENSOR = lltype.nullptr(TENSOR)

ADD, MUL, RELU, SUM, RELUGRAD = 0, 1, 2, 3, 4
SUB, DIV, EXP, SQRT, MAXR = 5, 6, 7, 8, 9
EQMASK = 10
NOPCODES = 11
ARITY = [2, 2, 1, 1, 2, 2, 2, 1, 1, 1, 2]
NAMES = ['add', 'mul', 'relu', 'sum', 'relugrad',
         'sub', 'div', 'exp', 'sqrt', 'maxr', 'eqmask']
HAS_PARAM = [True, True, False, True, True,
             True, True, False, False, True, True]
MAX_INPUTS = 6
BC_NONE, BC_R_ROW, BC_R_SCALAR, BC_L_ROW, BC_L_SCALAR = 0, 1, 2, 3, 4
BC_R_COL, BC_L_COL = 5, 6
NPARAMS = 7
AXIS_ALL = -1
NEG_INF_BITS = '0xFFF0000000000000'
NEG_INF = -INFINITY

NODE = lltype.Struct('TENSOR_NODE', ('opcode', lltype.Signed),
                     ('a', lltype.Signed), ('b', lltype.Signed),
                     ('p', lltype.Signed))
NODEARRAY = lltype.GcArray(NODE)
KERNEL = lltype.GcStruct('TENSOR_KERNEL', ('ninputs', lltype.Signed),
                         ('nodes', lltype.Ptr(NODEARRAY)),
                         ('fn', lltype.Signed),
                         ('sumroot', lltype.Signed),
                         ('rowmode', lltype.Signed),
                         ('threads', lltype.Signed),
                         ('shared', lltype.Signed),
                         ('nextra', lltype.Signed),
                         ('n', lltype.Signed),
                         ('cols', lltype.Signed),
                         ('outputs', lltype.Ptr(SHAPEARRAY)))
KERNELPTR = lltype.Ptr(KERNEL)

class DeviceBuffer(object):
    def __init__(self, dptr, n):
        self.dptr = dptr
        self.n = n

    @rgc.must_be_light_finalizer
    def __del__(self):
        rt_cuda_free(self.dptr, self.n)

def attach_buffer(t, dptr, n):
    t.dptr = dptr
    t.buf = cast_instance_to_base_ptr(DeviceBuffer(dptr, n))

def _shape1(n):
    shape = lltype.malloc(SHAPEARRAY, 1)
    shape[0] = n
    return shape

def new_tensor(n, shape=lltype.nullptr(SHAPEARRAY)):
    t = lltype.malloc(TENSOR)
    t.size = n
    t.shape = shape if shape else _shape1(n)
    t.dptr = 0
    t.host = lltype.malloc(HOSTARRAY, n)
    t.extra = lltype.nullptr(TENSORARRAY)
    t.buf = lltype.nullptr(OBJECTPTR.TO)
    return t

def device_tensor(n, dptr, shape=lltype.nullptr(SHAPEARRAY)):
    t = lltype.malloc(TENSOR)
    t.size = n
    t.shape = shape if shape else _shape1(n)
    t.host = lltype.nullptr(HOSTARRAY)
    t.extra = lltype.nullptr(TENSORARRAY)
    attach_buffer(t, dptr, n)
    return t

def zeros(shape_list):
    n = 1
    for d in shape_list:
        n *= d
    shape = lltype.malloc(SHAPEARRAY, len(shape_list))
    for i in range(len(shape_list)):
        shape[i] = shape_list[i]
    return new_tensor(n, shape)

def from_list(values):
    t = new_tensor(len(values))
    for i in range(len(values)):
        t.host[i] = values[i]
    return t

def host(t):
    if not t.host:
        t.host = lltype.malloc(HOSTARRAY, t.size)
        download(t.dptr, t.host)
    return t.host

def dev(t):
    if t.dptr == 0:
        dptr = upload(t.host)
        if dptr != 0:
            attach_buffer(t, dptr, t.size)
    return t.dptr

def cols(t):
    nd = len(t.shape)
    if nd > 1:
        c = t.shape[nd - 1]
        if c > 0:
            return c
    return 1

def eval_op(opcode, a, b, p):
    kernel = single_kernel(opcode, p)
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
        r = new_tensor(n, a.shape)
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
    r = new_tensor(n, big.shape)
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
    r = new_tensor(m)
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

class SingleKernels(object):
    def __init__(self):
        self.kernels = [_empty_kernel() for i in range(NOPCODES * NPARAMS)]

def _empty_kernel():
    k = lltype.malloc(KERNEL)
    k.ninputs = 0
    k.nodes = lltype.malloc(NODEARRAY, 0)
    k.fn = k.sumroot = k.threads = k.shared = k.nextra = 0
    k.rowmode = 0
    k.n = 0
    k.cols = 0
    k.outputs = lltype.malloc(SHAPEARRAY, 0)
    return k
single_kernels = SingleKernels()

def is_reduction(opcode):
    return opcode == SUM or opcode == MAXR

def param_slot(opcode, p):
    if is_reduction(opcode):
        if p == 0 or p == 1:
            return p + 1
        return 0
    if not HAS_PARAM[opcode]:
        return 0
    if p > 0 and p < NPARAMS:
        return p
    return 0

def slot_param(opcode, slot):
    if is_reduction(opcode):
        return slot - 1
    return slot

def slot_used(opcode, slot):
    if is_reduction(opcode):
        return slot < 3
    if not HAS_PARAM[opcode]:
        return slot == 0
    return True

def single_kernel(opcode, p):
    return single_kernels.kernels[opcode * NPARAMS + param_slot(opcode, p)]

def init_device():
    try:
        config.block = int(_env('RTENSOR_BLOCK', '4096'))
        config.num_warps = int(_env('RTENSOR_WARPS', '8'))
        rt_cuda_set_budget(int(_env('RTENSOR_BUDGET_MB', '64')) << 20)
    except ValueError:
        pass
    for opcode in range(NOPCODES):
        for slot in range(NPARAMS):
            if not slot_used(opcode, slot):
                continue
            opcodes = []
            opcodes.append(opcode)
            lefts = []
            lefts.append(0)
            rights = []
            rights.append(1 if ARITY[opcode] == 2 else -1)
            params = []
            params.append(slot_param(opcode, slot))
            single_kernels.kernels[opcode * NPARAMS + slot] = build_kernel(
                ARITY[opcode], opcodes, lefts, rights, params)

@jit.oopspec("tensor.add(a, b, bcast)")
def tensor_add(a, b, bcast):
    return eval_op(ADD, a, b, bcast)

@jit.oopspec("tensor.mul(a, b, bcast)")
def tensor_mul(a, b, bcast):
    return eval_op(MUL, a, b, bcast)

@jit.oopspec("tensor.relu(a)")
def tensor_relu(a):
    return eval_op(RELU, a, NULLTENSOR, 0)

@jit.oopspec("tensor.sum(a, axis)")
def tensor_sum(a, axis):
    return eval_op(SUM, a, NULLTENSOR, axis)

@jit.oopspec("tensor.relugrad(y, g, bcast)")
def tensor_relugrad(y, g, bcast):
    return eval_op(RELUGRAD, y, g, bcast)

@jit.oopspec("tensor.sub(a, b, bcast)")
def tensor_sub(a, b, bcast):
    return eval_op(SUB, a, b, bcast)

@jit.oopspec("tensor.div(a, b, bcast)")
def tensor_div(a, b, bcast):
    return eval_op(DIV, a, b, bcast)

@jit.oopspec("tensor.exp(a)")
def tensor_exp(a):
    return eval_op(EXP, a, NULLTENSOR, 0)

@jit.oopspec("tensor.sqrt(a)")
def tensor_sqrt(a):
    return eval_op(SQRT, a, NULLTENSOR, 0)

@jit.oopspec("tensor.maxr(a, axis)")
def tensor_maxr(a, axis):
    return eval_op(MAXR, a, NULLTENSOR, axis)

@jit.oopspec("tensor.eqmask(a, b, bcast)")
def tensor_eqmask(a, b, bcast):
    return eval_op(EQMASK, a, b, bcast)

@jit.oopspec("tensor.ndim(a)")
def tensor_ndim(a):
    return len(a.shape)

@jit.oopspec("tensor.size(a)")
def tensor_size(a):
    return a.size

@jit.oopspec("tensor.shape(a, axis)")
def tensor_shape(a, axis):
    return a.shape[axis]


class SizePolicy(object):
    _immutable_fields_ = ['static?', 'static_cols?']
    def __init__(self):
        self.static = True
        self.seen = []
        self.static_cols = True
        self.seen_cols = []
policy = SizePolicy()
MAX_STATIC_SIZES = 3

@jit.elidable
def note_size(n):
    if n not in policy.seen:
        policy.seen.append(n)
        if len(policy.seen) > MAX_STATIC_SIZES:
            policy.static = False
    return n

@jit.elidable
def note_cols(c):
    if c not in policy.seen_cols:
        policy.seen_cols.append(c)
        if len(policy.seen_cols) > MAX_STATIC_SIZES:
            policy.static_cols = False
    return c

def cols_of(t):
    c = tensor_shape(t, 1)
    if policy.static_cols:
        c = jit.promote(c)
        note_cols(c)
    return c

def size(t):
    n = tensor_size(t)
    if policy.static:
        n = jit.promote(n)
        note_size(n)
    return n

def bcast_of(big, small, m):
    if m == 1:
        return BC_R_SCALAR
    if tensor_ndim(big) == 2:
        rows = tensor_shape(big, 0)
        c = tensor_shape(big, 1)
        if m == rows and tensor_ndim(small) == 2 and tensor_shape(small, 1) == 1:
            cols_of(big)
            return BC_R_COL
        if m == c:
            return BC_R_ROW
        if m == rows:
            cols_of(big)
            return BC_R_COL
    raise ValueError("shape mismatch")

def flip_bcast(p):
    if p == BC_R_ROW:
        return BC_L_ROW
    if p == BC_R_SCALAR:
        return BC_L_SCALAR
    return BC_L_COL

def bcast(a, b):
    na = size(a)
    nb = size(b)
    if na == nb:
        return BC_NONE
    if na > nb:
        return bcast_of(a, b, nb)
    return flip_bcast(bcast_of(b, a, na))

def add(a, b):
    return tensor_add(a, b, bcast(a, b))

def mul(a, b):
    return tensor_mul(a, b, bcast(a, b))

def add_(a, b):
    return add(a, b)

def mul_(a, b):
    return mul(a, b)

def relu(a):
    return tensor_relu(a)

def relugrad(y, g):
    return tensor_relugrad(y, g, bcast(y, g))

def sum(a, axis=AXIS_ALL):
    if axis == 1 and tensor_ndim(a) > 1:
        cols_of(a)
    return tensor_sum(a, axis)

def sub(a, b):
    return tensor_sub(a, b, bcast(a, b))

def div(a, b):
    return tensor_div(a, b, bcast(a, b))

def exp(a):
    return tensor_exp(a)

def sqrt(a):
    return tensor_sqrt(a)

def max(a, axis=AXIS_ALL):
    if axis == 1 and tensor_ndim(a) > 1:
        cols_of(a)
    return tensor_maxr(a, axis)

def reshape(a, shape_list):
    n = 1
    for d in shape_list:
        n *= d
    a = tensor_force(a)
    if n != a.size:
        raise ValueError("shape mismatch")
    shape = lltype.malloc(SHAPEARRAY, len(shape_list))
    for i in range(len(shape_list)):
        shape[i] = shape_list[i]
    return view(a, shape)

def view(a, shape):
    r = lltype.malloc(TENSOR)
    r.size = a.size
    r.shape = shape
    r.dptr = a.dptr
    r.host = a.host
    r.extra = lltype.nullptr(TENSORARRAY)
    r.buf = a.buf
    return r

class Ones(object):
    def __init__(self):
        self.n = -1
        self.t = NULLTENSOR
        self.one = NULLTENSOR
ones = Ones()

def _make_ones(n):
    t = new_tensor(n)
    for i in range(n):
        t.host[i] = 1.0
    dev(t)
    return t

@jit.unroll_safe
def ones_like(a):
    n = tensor_size(a)
    nd = tensor_ndim(a)
    shape = lltype.malloc(SHAPEARRAY, nd)
    for i in range(nd):
        shape[i] = tensor_shape(a, i)
    if n == 1:
        if not ones.one:
            ones.one = _make_ones(1)
        return view(ones.one, shape)
    if ones.n != n:
        ones.t = _make_ones(n)
        ones.n = n
    return view(ones.t, shape)

class ScalarCache(object):
    def __init__(self):
        self.tensors = {}
scalars = ScalarCache()


@jit.elidable
def scalar(value):
    t = scalars.tensors.get(value, NULLTENSOR)
    if not t:
        t = from_list([value])
        dev(t)
        scalars.tensors[value] = t
    return t


def matmul_shape(a, b):
    if tensor_ndim(a) != 2 or tensor_ndim(b) != 2:
        raise ValueError("shape mismatch")
    inner = tensor_shape(a, 1)
    if inner != tensor_shape(b, 0):
        raise ValueError("shape mismatch")
    return tensor_shape(a, 0), tensor_shape(b, 1), inner

def matmul(a, b, transpose_b=False):
    if transpose_b:
        rows, cols, inner = matmul_shape_t(a, b)
        return tensor_matmul(a, b, rows, cols, inner, 0, 1)
    rows, cols, inner = matmul_shape(a, b)
    return tensor_matmul(a, b, rows, cols, inner, 0, 0)

def matmul_shape_t(a, b):
    if tensor_ndim(a) != 2 or tensor_ndim(b) != 2:
        raise ValueError("shape mismatch")
    inner = tensor_shape(a, 1)
    if inner != tensor_shape(b, 1):
        raise ValueError("shape mismatch")
    return tensor_shape(a, 0), tensor_shape(b, 0), inner

def matmul_cpu(a, b, rows, cols, inner, ta, tb):
    ha = host(a)
    hb = host(b)
    shape = lltype.malloc(SHAPEARRAY, 2)
    shape[0] = rows
    shape[1] = cols
    r = new_tensor(rows * cols, shape)
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
def tensor_matmul(a, b, rows, cols, inner, ta, tb):
    if gpu_enabled():
        dptr_a = dev(a)
        dptr_b = dev(b)
        if dptr_a != 0 and dptr_b != 0:
            n = rows * cols
            collect_if_needed(n)
            outptr = rt_cuda_alloc(n, 0)
            if outptr != 0:
                ok = rffi.cast(lltype.Signed, rt_cuda_matmul(
                    dptr_a, dptr_b, outptr, rows, inner, cols, ta, tb)) != 0
                if ok:
                    shape = lltype.malloc(SHAPEARRAY, 2)
                    shape[0] = rows
                    shape[1] = cols
                    return device_tensor(n, outptr, shape)
                rt_cuda_free(outptr, n)
    return matmul_cpu(a, b, rows, cols, inner, ta, tb)

def item(a):
    return tensor_item(a)

def tensor_output(t, k):
    return t.extra[k]

@jit.dont_look_inside
def tensor_force(a):
    return a

@jit.elidable
def tensor_item(a):
    return host(a)[0]

class KernelCache(object):
    def __init__(self):
        self.kernels = {}
kernel_cache = KernelCache()

def cached_kernel(key):
    return kernel_cache.kernels.get(key, lltype.nullptr(KERNEL))

def cache_kernel(key, kernel):
    kernel_cache.kernels[key] = kernel

def new_kernel(ninputs, nnodes):
    kernel = lltype.malloc(KERNEL)
    kernel.ninputs = ninputs
    kernel.nodes = lltype.malloc(NODEARRAY, nnodes)
    kernel.fn = kernel.sumroot = kernel.threads = kernel.shared = kernel.nextra = 0
    kernel.rowmode = 0
    kernel.n = 0
    kernel.cols = 0
    kernel.outputs = lltype.malloc(SHAPEARRAY, 0)
    return kernel

def add_output(kernel, node):
    old = kernel.outputs
    new = lltype.malloc(SHAPEARRAY, len(old) + 1)
    for i in range(len(old)):
        new[i] = old[i]
    new[len(old)] = node
    kernel.outputs = new
    return len(old)

def next_pow2(c):
    t = 32
    while t < c:
        t *= 2
    return t

def row_tile(kernel):
    if kernel.cols > 0:
        t = next_pow2(kernel.cols)
        if t <= config.block:
            return t
    return config.block

def row_warps(tile):
    w = tile // 128
    if w < 1:
        return 1
    if w > 8:
        return 8
    return w

def kernel_key(kernel):
    rowmode = kernel_row_mode(kernel)
    parts = [str(kernel.ninputs), '0' if rowmode else str(kernel.n)]
    if rowmode:
        parts.append('r%d' % row_tile(kernel))
    for i in range(len(kernel.nodes)):
        node = kernel.nodes[i]
        parts.append('%d:%d:%d:%d' % (node.opcode, node.a, node.b, node.p))
    for i in range(len(kernel.outputs)):
        parts.append('o%d' % kernel.outputs[i])
    return ','.join(parts)

def compile_or_reuse(kernel):
    key = kernel_key(kernel)
    cached = cached_kernel(key)
    if cached:
        kernel.fn = cached.fn
        kernel.threads = cached.threads
        kernel.shared = cached.shared
        kernel.nextra = cached.nextra
        kernel.sumroot = cached.sumroot
        kernel.rowmode = cached.rowmode
        return kernel
    finish_kernel(kernel)
    cache_kernel(key, kernel)
    return kernel

def set_node(kernel, i, opcode, a, b, p):
    node = kernel.nodes[i]
    node.opcode = opcode
    node.a = a
    node.b = b
    node.p = p

def finish_kernel(kernel):
    n = len(kernel.nodes)
    kernel.sumroot = int(n > 0 and is_reduction(kernel.nodes[n - 1].opcode))
    kernel.rowmode = int(kernel_row_mode(kernel))
    kernel.fn = compile_gpu(kernel)
    return kernel

def build_kernel(ninputs, opcodes, lefts, rights, params):
    kernel = new_kernel(ninputs, len(opcodes))
    for i in range(len(opcodes)):
        set_node(kernel, i, opcodes[i], lefts[i], rights[i], params[i])
    return finish_kernel(kernel)

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
        r = launch_gpu(kernel, values)
        if r:
            return r
    nodes = kernel.nodes
    for i in range(len(nodes)):
        node = nodes[i]
        opcode = node.opcode
        assert opcode >= 0
        right = values[node.b] if node.b >= 0 else NULLTENSOR
        values.append(eval_op(opcode, values[node.a], right, node.p))
    result = values[len(values) - 1]
    nout = len(kernel.outputs)
    if nout > 0:
        result.extra = lltype.malloc(TENSORARRAY, nout)
        for k in range(nout):
            result.extra[k] = values[kernel.outputs[k]]
    return result

def to_tile_ir(kernel, name, n):
    tile = 'tile<%dxf64>' % n
    params = ', '.join(['%%in%d: !cuda_tile.ptr<f64>' % i
                        for i in range(kernel.ninputs)] +
                       ['%out: !cuda_tile.ptr<f64>'])
    lines = ['cuda_tile.module {',
             '  cuda_tile.entry @%s(%s) {' % (name, params)]
    for i in range(kernel.ninputs):
        lines.append('    %%v%d = cuda_tile.load_ptr_tko %%in%d : %s'
                     % (i, i, tile))
    result = tile
    nodes = kernel.nodes
    for i in range(len(nodes)):
        node = nodes[i]
        v = kernel.ninputs + i
        if node.opcode == ADD:
            lines.append('    %%v%d = arith.addf %%v%d, %%v%d : %s'
                         % (v, node.a, node.b, tile))
        elif node.opcode == MUL:
            lines.append('    %%v%d = arith.mulf %%v%d, %%v%d : %s'
                         % (v, node.a, node.b, tile))
        elif node.opcode == RELU:
            lines.append('    %%v%d = arith.maximumf %%v%d, %%zero : %s'
                         % (v, node.a, tile))
        elif node.opcode == RELUGRAD:
            lines.append('    %%v%d = cuda_tile.select %%v%d, %%v%d, %%zero : %s'
                         % (v, node.a, node.b, tile))
        else:
            result = 'tile<1xf64>'
            lines.append('    %%v%d = cuda_tile.reduce add %%v%d : %s -> %s'
                         % (v, node.a, tile, result))
    lines.append('    cuda_tile.store_ptr_tko %%out, %%v%d : %s'
                 % (kernel.ninputs + len(nodes) - 1, result))
    lines.append('    cuda_tile.return')
    lines.append('  }')
    lines.append('}')
    return '\n'.join(lines)

class Config(object):
    block = 4096
    num_warps = 8
config = Config()
DOUBLEARRAY = rffi.CArray(rffi.DOUBLE)
SIGNEDARRAY = rffi.CArray(lltype.Signed)
_here = os.path.dirname(os.path.abspath(__file__))
CUDA_HOME = os.environ.get('CUDA_HOME', '/usr/local/cuda')
eci = ExternalCompilationInfo(
    include_dirs=[os.path.join(CUDA_HOME, 'include')],
    separate_module_files=[os.path.join(_here, 'rtensor_cuda.c')],
    post_include_bits=["""
RPY_EXTERN int rt_cuda_available(void);
RPY_EXTERN long rt_cuda_load(const char *ptx, const char *name);
RPY_EXTERN long rt_cuda_alloc(long n, long zero);
RPY_EXTERN long rt_cuda_upload(double *host, long n);
RPY_EXTERN int rt_cuda_download(long dptr, double *host, long n);
RPY_EXTERN void rt_cuda_reset(void);
RPY_EXTERN void rt_cuda_free(long dptr, long n);
RPY_EXTERN void rt_cuda_set_budget(long bytes);
RPY_EXTERN long rt_cuda_launch_count(void);
RPY_EXTERN int rt_cuda_needs_gc(long n);
RPY_EXTERN void rt_cuda_sync(void);
RPY_EXTERN int rt_cuda_launch(long fn, long *inputs, int ninputs, long n,
                              long *outs, int nouts, int threads,
                              long elems_per_block, int shared, int nextra,
                              long cols);
RPY_EXTERN int rt_cuda_matmul(long a, long b, long c, long rows, long inner,
                              long cols, long ta, long tb);
RPY_EXTERN int rt_cuda_bmm(long a, long b, long c, long batch, long rows,
                           long inner, long cols, long ta, long tb);
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
                                 [rffi.DOUBLEP, lltype.Signed],
                                 lltype.Signed, compilation_info=eci,
                                releasegil=False)
rt_cuda_download = rffi.llexternal('rt_cuda_download',
                                   [lltype.Signed, rffi.DOUBLEP, lltype.Signed],
                                   rffi.INT, compilation_info=eci,
                                releasegil=False)
rt_cuda_reset = rffi.llexternal('rt_cuda_reset', [], lltype.Void,
                                compilation_info=eci,
                                releasegil=False)
rt_cuda_sync = rffi.llexternal('rt_cuda_sync', [], lltype.Void,
                               compilation_info=eci,
                                releasegil=False)
rt_cuda_matmul = rffi.llexternal(
    'rt_cuda_matmul',
    [lltype.Signed, lltype.Signed, lltype.Signed, lltype.Signed,
     lltype.Signed, lltype.Signed, lltype.Signed, lltype.Signed],
    rffi.INT, compilation_info=eci,
                                releasegil=False)
rt_cuda_bmm = rffi.llexternal(
    'rt_cuda_bmm',
    [lltype.Signed, lltype.Signed, lltype.Signed, lltype.Signed,
     lltype.Signed, lltype.Signed, lltype.Signed, lltype.Signed,
     lltype.Signed],
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

def upload(hostarray):
    if not gpu_enabled():
        return 0
    n = len(hostarray)
    buf = lltype.malloc(DOUBLEARRAY, n, flavor='raw')
    for i in range(n):
        buf[i] = hostarray[i]
    dptr = rt_cuda_upload(buf, n)
    lltype.free(buf, flavor='raw')
    return dptr

def download(dptr, hostarray):
    n = len(hostarray)
    buf = lltype.malloc(DOUBLEARRAY, n, flavor='raw')
    rt_cuda_download(dptr, buf, n)
    for i in range(n):
        hostarray[i] = buf[i]
    lltype.free(buf, flavor='raw')



rt_cuda_free = rffi.llexternal('rt_cuda_free', [lltype.Signed, lltype.Signed],
                               lltype.Void, compilation_info=eci,
                               releasegil=False)

rt_cuda_set_budget = rffi.llexternal('rt_cuda_set_budget', [lltype.Signed],
                                     lltype.Void, compilation_info=eci,
                                     releasegil=False)
rt_cuda_needs_gc = rffi.llexternal('rt_cuda_needs_gc', [lltype.Signed], rffi.INT,
                                   compilation_info=eci, releasegil=False)

rt_cuda_launch_count = rffi.llexternal('rt_cuda_launch_count', [],
                                       lltype.Signed, compilation_info=eci,
                                       releasegil=False)

def launch_count():
    return rt_cuda_launch_count()

def reset_device():
    scalars.tensors.clear()
    ones.n = -1
    ones.t = NULLTENSOR
    ones.one = NULLTENSOR
    rt_cuda_reset()

def collect_if_needed(n):
    if rffi.cast(lltype.Signed, rt_cuda_needs_gc(n)) != 0:
        rgc.collect(0)
        if rffi.cast(lltype.Signed, rt_cuda_needs_gc(n)) != 0:
            rgc.collect()

def sync_device():
    rt_cuda_sync()

def set_mode(modes, v, m):
    if modes[v] < 0:
        modes[v] = m
        return True
    return modes[v] == m

def all_modes(kernel):
    nin = kernel.ninputs
    nodes = kernel.nodes
    modes = [-1] * (nin + len(nodes))
    for k in range(len(nodes) - 1, -1, -1):
        node = nodes[k]
        m = modes[nin + k]
        if m < 0:
            m = 0
        ma = m
        mb = m
        if is_reduction(node.opcode):
            ma = 0
        elif ARITY[node.opcode] == 2:
            if node.p == BC_R_ROW:
                mb = 1
            elif node.p == BC_R_SCALAR:
                mb = 2
            elif node.p == BC_R_COL:
                mb = 3
            elif node.p == BC_L_ROW:
                ma = 1
            elif node.p == BC_L_SCALAR:
                ma = 2
            elif node.p == BC_L_COL:
                ma = 3
        if not set_mode(modes, node.a, ma):
            return []
        if node.b >= 0 and not set_mode(modes, node.b, mb):
            return []
    for i in range(len(modes)):
        if modes[i] < 0:
            modes[i] = 0
    return modes

def input_modes(kernel):
    modes = all_modes(kernel)
    nin = kernel.ninputs
    if len(modes) != nin + len(kernel.nodes):
        return []
    result = []
    for i in range(nin):
        result.append(modes[i])
    return result

def row_mode(kernel, modes):
    nin = kernel.ninputs
    nodes = kernel.nodes
    if len(modes) != nin + len(nodes):
        return False
    for k in range(len(nodes)):
        node = nodes[k]
        if is_reduction(node.opcode) and node.p == 1:
            if k == len(nodes) - 1 or modes[nin + k] == 3:
                return True
    for i in range(nin):
        if modes[i] == 3:
            return True
    return False

def kernel_row_mode(kernel):
    return row_mode(kernel, all_modes(kernel))

def to_ttir(kernel, name):
    modes = all_modes(kernel)
    if len(modes) != kernel.ninputs + len(kernel.nodes):
        return ''
    if row_mode(kernel, modes):
        return to_ttir_row(kernel, name, modes)
    return to_ttir_flat(kernel, name)

def _elementwise(lines, node, v, T, I1):
    if node.opcode == ADD:
        lines.append('    %%v%d = arith.addf %%v%d, %%v%d : %s'
                     % (v, node.a, node.b, T))
    elif node.opcode == MUL:
        lines.append('    %%v%d = arith.mulf %%v%d, %%v%d : %s'
                     % (v, node.a, node.b, T))
    elif node.opcode == RELU:
        lines.append('    %%c%d = arith.cmpf ogt, %%v%d, %%zero : %s'
                     % (v, node.a, T))
        lines.append('    %%v%d = arith.select %%c%d, %%v%d, %%zero : %s, %s'
                     % (v, v, node.a, I1, T))
    elif node.opcode == RELUGRAD:
        lines.append('    %%c%d = arith.cmpf ogt, %%v%d, %%zero : %s'
                     % (v, node.a, T))
        lines.append('    %%v%d = arith.select %%c%d, %%v%d, %%zero : %s, %s'
                     % (v, v, node.b, I1, T))
    elif node.opcode == SUB:
        lines.append('    %%v%d = arith.subf %%v%d, %%v%d : %s'
                     % (v, node.a, node.b, T))
    elif node.opcode == DIV:
        lines.append('    %%v%d = arith.divf %%v%d, %%v%d : %s'
                     % (v, node.a, node.b, T))
    elif node.opcode == EXP:
        lines.append('    %%v%d = math.exp %%v%d : %s' % (v, node.a, T))
    elif node.opcode == SQRT:
        lines.append('    %%v%d = math.sqrt %%v%d : %s' % (v, node.a, T))
    elif node.opcode == EQMASK:
        lines.append('    %%c%d = arith.cmpf oeq, %%v%d, %%v%d : %s'
                     % (v, node.a, node.b, T))
        lines.append('    %%v%d = arith.select %%c%d, %%one, %%zero : %s, %s'
                     % (v, v, I1, T))
    else:
        return False
    return True

def to_ttir_row(kernel, name, modes):
    nodes = kernel.nodes
    nin = kernel.ninputs
    last = len(nodes) - 1
    if last < 0:
        return ''
    BLOCK = row_tile(kernel)
    T = 'tensor<%dxf64>' % BLOCK
    P = 'tensor<%dx!tt.ptr<f64>>' % BLOCK
    I32 = 'tensor<%dxi32>' % BLOCK
    I64 = 'tensor<%dxi64>' % BLOCK
    I1 = 'tensor<%dxi1>' % BLOCK
    isred = [False] * (nin + len(nodes))
    for k in range(len(nodes)):
        node = nodes[k]
        if not is_reduction(node.opcode):
            continue
        if node.p == 1:
            if k != last and modes[nin + k] != 3:
                return ''
        elif node.p == AXIS_ALL:
            if k != last or node.opcode != SUM:
                return ''
        else:
            return ''
        isred[nin + k] = True
    for k in range(len(kernel.outputs)):
        if isred[kernel.outputs[k]]:
            return ''
    params = ['%%in%d: !tt.ptr<f64>' % i for i in range(nin)]
    params.append('%out: !tt.ptr<f64>')
    for k in range(len(kernel.outputs)):
        params.append('%%out%d: !tt.ptr<f64>' % k)
    lines = ['module {',
             '  tt.func public @%s(%s, %%n: i64, %%c: i64) '
             'attributes {noinline = false} {' % (name, ', '.join(params)),
             '    %%zero = arith.constant dense<0.0> : %s' % T,
             '    %%one = arith.constant dense<1.0> : %s' % T,
             '    %pid = tt.get_program_id x : i32',
             '    %rowi = arith.extsi %pid : i32 to i64',
             '    %%range = tt.make_range {end = %d : i32, start = 0 : i32} '
             ': %s' % (BLOCK, I32),
             '    %%ar = arith.extsi %%range : %s to %s' % (I32, I64),
             '    %%cs = tt.splat %%c : i64 -> %s' % I64,
             '    %%mask = arith.cmpi slt, %%ar, %%cs : %s' % I64,
             '    %base = arith.muli %rowi, %c : i64',
             '    %%bases = tt.splat %%base : i64 -> %s' % I64,
             '    %%offs = arith.addi %%bases, %%ar : %s' % I64]
    for i in range(nin):
        if modes[i] == 2 or modes[i] == 3:
            src = '%%in%d' % i
            if modes[i] == 3:
                lines.append('    %%sp%d = tt.addptr %%in%d, %%pid : '
                             '!tt.ptr<f64>, i32' % (i, i))
                src = '%%sp%d' % i
            lines.append('    %%sv%d = tt.load %s : !tt.ptr<f64>' % (i, src))
            lines.append('    %%v%d = tt.splat %%sv%d : f64 -> %s' % (i, i, T))
            continue
        offs = '%offs'
        if modes[i] == 1:
            offs = '%ar'
        lines.append('    %%p%d = tt.splat %%in%d : !tt.ptr<f64> -> %s'
                     % (i, i, P))
        lines.append('    %%q%d = tt.addptr %%p%d, %s : %s, %s'
                     % (i, i, offs, P, I64))
        lines.append('    %%v%d = tt.load %%q%d, %%mask, %%zero : %s'
                     % (i, i, P))
    for k in range(len(nodes)):
        node = nodes[k]
        v = nin + k
        if _elementwise(lines, node, v, T, I1):
            continue
        if node.opcode == MAXR:
            lines.append('    %%ninf%d = arith.constant dense<%s> : %s'
                         % (v, NEG_INF_BITS, T))
            init = '%%ninf%d' % v
            combine = 'maximumf'
        else:
            init = '%zero'
            combine = 'addf'
        lines.append('    %%rm%d = arith.select %%mask, %%v%d, %s : %s, %s'
                     % (v, node.a, init, I1, T))
        lines.append('    %%rs%d = "tt.reduce"(%%rm%d) <{axis = 0 : i32}> ({'
                     % (v, v))
        lines.append('    ^bb0(%%x%d: f64, %%y%d: f64):' % (v, v))
        lines.append('      %%rr%d = arith.%s %%x%d, %%y%d : f64'
                     % (v, combine, v, v))
        lines.append('      tt.reduce.return %%rr%d : f64' % v)
        lines.append('    }) : (%s) -> f64' % T)
        lines.append('    %%v%d = tt.splat %%rs%d : f64 -> %s' % (v, v, T))
        if k != last:
            continue
        if node.p == 1:
            lines.append('    %po = tt.addptr %out, %pid : !tt.ptr<f64>, i32')
            lines.append('    tt.store %%po, %%rs%d : !tt.ptr<f64>' % v)
        else:
            lines.append('    %true = arith.constant true')
            lines.append('    %%o = tt.atomic_rmw fadd, acq_rel, gpu, %%out, '
                         '%%rs%d, %%true : (!tt.ptr<f64>, f64, i1) -> f64' % v)
    if not isred[nin + last]:
        lines.append('    %%po = tt.splat %%out : !tt.ptr<f64> -> %s' % P)
        lines.append('    %%qo = tt.addptr %%po, %%offs : %s, %s' % (P, I64))
        lines.append('    tt.store %%qo, %%v%d, %%mask : %s'
                     % (nin + last, P))
    for k in range(len(kernel.outputs)):
        lines.append('    %%po%d = tt.splat %%out%d : !tt.ptr<f64> -> %s'
                     % (k, k, P))
        lines.append('    %%qo%d = tt.addptr %%po%d, %%offs : %s, %s'
                     % (k, k, P, I64))
        lines.append('    tt.store %%qo%d, %%v%d, %%mask : %s'
                     % (k, kernel.outputs[k], P))
    lines.append('    tt.return')
    lines.append('  }')
    lines.append('}')
    return '\n'.join(lines) + '\n'

def to_ttir_flat(kernel, name):
    nodes = kernel.nodes
    nin = kernel.ninputs
    BLOCK = config.block
    masked = kernel.n == 0 or kernel.n % BLOCK != 0
    T = 'tensor<%dxf64>' % BLOCK
    P = 'tensor<%dx!tt.ptr<f64>>' % BLOCK
    I32 = 'tensor<%dxi32>' % BLOCK
    I64 = 'tensor<%dxi64>' % BLOCK
    I1 = 'tensor<%dxi1>' % BLOCK
    modes = input_modes(kernel)
    if len(modes) != nin:
        return ''
    axis = AXIS_ALL
    if kernel.sumroot:
        axis = nodes[len(nodes) - 1].p
    need_mod = axis == 0
    need_div = axis == 1
    need_zero_off = False
    for i in range(nin):
        if modes[i] == 1:
            need_mod = True
        elif modes[i] == 2:
            need_zero_off = True
        elif modes[i] == 3:
            need_div = True
    params = ['%%in%d: !tt.ptr<f64>' % i for i in range(nin)]
    params.append('%out: !tt.ptr<f64>')
    for k in range(len(kernel.outputs)):
        params.append('%%out%d: !tt.ptr<f64>' % k)
    lines = ['module {',
             '  tt.func public @%s(%s, %%n: i64, %%c: i64) '
             'attributes {noinline = false} {' % (name, ', '.join(params)),
             '    %%zero = arith.constant dense<0.0> : %s' % T,
             '    %%one = arith.constant dense<1.0> : %s' % T,
             '    %%bs = arith.constant %d : i32' % BLOCK,
             '    %pid = tt.get_program_id x : i32',
             '    %start = arith.muli %pid, %bs : i32',
             '    %%range = tt.make_range {end = %d : i32, start = 0 : i32} '
             ': %s' % (BLOCK, I32),
             '    %%starts = tt.splat %%start : i32 -> %s' % I32,
             '    %%offs = arith.addi %%starts, %%range : %s' % I32,
             '    %%offs64 = arith.extsi %%offs : %s to %s' % (I32, I64),
             '    %%ns = tt.splat %%n : i64 -> %s' % I64,
             '    %%mask = arith.cmpi slt, %%offs64, %%ns : %s' % I64]
    if need_mod or need_div:
        lines.append('    %%cs = tt.splat %%c : i64 -> %s' % I64)
    if need_mod:
        lines.append('    %%offsm = arith.remsi %%offs64, %%cs : %s' % I64)
    if need_div:
        lines.append('    %%offsd = arith.divsi %%offs64, %%cs : %s' % I64)
    if need_zero_off:
        lines.append('    %%zoffs = arith.constant dense<0> : %s' % I32)
    if masked:
        tmask = '%mask'
    else:
        tmask = '%tmask'
        lines.append('    %%tmask = arith.constant dense<true> : %s' % I1)
    for i in range(nin):
        lines.append('    %%p%d = tt.splat %%in%d : !tt.ptr<f64> -> %s' % (i, i, P))
        if modes[i] == 1:
            lines.append('    %%q%d = tt.addptr %%p%d, %%offsm : %s, %s'
                         % (i, i, P, I64))
        elif modes[i] == 2:
            lines.append('    %%q%d = tt.addptr %%p%d, %%zoffs : %s, %s'
                         % (i, i, P, I32))
        elif modes[i] == 3:
            lines.append('    %%q%d = tt.addptr %%p%d, %%offsd : %s, %s'
                         % (i, i, P, I64))
        else:
            lines.append('    %%q%d = tt.addptr %%p%d, %%offs : %s, %s'
                         % (i, i, P, I32))
        if masked:
            lines.append('    %%v%d = tt.load %%q%d, %%mask, %%zero : %s' % (i, i, P))
        else:
            lines.append('    %%v%d = tt.load %%q%d : %s' % (i, i, P))
    last = nin + len(nodes) - 1
    for k in range(len(nodes)):
        node = nodes[k]
        v = nin + k
        if _elementwise(lines, node, v, T, I1):
            continue
        if k != len(nodes) - 1 or not is_reduction(node.opcode):
            return ''
        if node.opcode == MAXR:
            if axis != AXIS_ALL or kernel.n == 0 or kernel.n > BLOCK:
                return ''
            src = '%%v%d' % node.a
            if masked:
                lines.append('    %%ninf = arith.constant dense<%s> : %s'
                             % (NEG_INF_BITS, T))
                lines.append('    %%m%d = arith.select %%mask, %%v%d, %%ninf '
                             ': %s, %s' % (v, node.a, I1, T))
                src = '%%m%d' % v
            lines.append('    %%v%d = "tt.reduce"(%s) <{axis = 0 : i32}> ({'
                         % (v, src))
            lines.append('    ^bb0(%x: f64, %y: f64):')
            lines.append('      %r = arith.maximumf %x, %y : f64')
            lines.append('      tt.reduce.return %r : f64')
            lines.append('    }) : (%s) -> f64' % T)
            lines.append('    tt.store %%out, %%v%d : !tt.ptr<f64>' % v)
        else:
            if axis == 0 or axis == 1:
                offs = '%offsm'
                if axis == 1:
                    offs = '%offsd'
                lines.append('    %%po = tt.splat %%out : !tt.ptr<f64> -> %s' % P)
                lines.append('    %%qo = tt.addptr %%po, %s : %s, %s'
                             % (offs, P, I64))
                lines.append('    %%v%d = tt.atomic_rmw fadd, acq_rel, gpu, '
                             '%%qo, %%v%d, %s : (%s, %s, %s) -> %s'
                             % (v, node.a, tmask, P, T, I1, T))
            else:
                lines.append('    %%v%d = "tt.reduce"(%%v%d) <{axis = 0 : i32}> ({'
                             % (v, node.a))
                lines.append('    ^bb0(%x: f64, %y: f64):')
                lines.append('      %r = arith.addf %x, %y : f64')
                lines.append('      tt.reduce.return %r : f64')
                lines.append('    }) : (%s) -> f64' % T)
                lines.append('    %true = arith.constant true')
                lines.append('    %%o = tt.atomic_rmw fadd, acq_rel, gpu, %%out, %%v%d, '
                             '%%true : (!tt.ptr<f64>, f64, i1) -> f64' % v)
    if not kernel.sumroot:
        lines.append('    %%po = tt.splat %%out : !tt.ptr<f64> -> %s' % P)
        lines.append('    %%qo = tt.addptr %%po, %%offs : %s, %s' % (P, I32))
        if masked:
            lines.append('    tt.store %%qo, %%v%d, %%mask : %s' % (last, P))
        else:
            lines.append('    tt.store %%qo, %%v%d : %s' % (last, P))
    for k in range(len(kernel.outputs)):
        lines.append('    %%po%d = tt.splat %%out%d : !tt.ptr<f64> -> %s' % (k, k, P))
        lines.append('    %%qo%d = tt.addptr %%po%d, %%offs : %s, %s'
                     % (k, k, P, I32))
        if masked:
            lines.append('    tt.store %%qo%d, %%v%d, %%mask : %s'
                         % (k, kernel.outputs[k], P))
        else:
            lines.append('    tt.store %%qo%d, %%v%d : %s' % (k, kernel.outputs[k], P))
    lines.append('    tt.return')
    lines.append('  }')
    lines.append('}')
    return '\n'.join(lines) + '\n'

class Counter(object):
    n = 0
counter = Counter()

def _write(path, data):
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0644)
    os.write(fd, data)
    os.close(fd)

def _read(path):
    fd = os.open(path, os.O_RDONLY, 0)
    chunks = []
    while True:
        chunk = os.read(fd, 65536)
        if not chunk:
            break
        chunks.append(chunk)
    os.close(fd)
    return ''.join(chunks)

def compile_gpu(kernel):
    if not gpu_enabled():
        return 0
    try:
        return _compile_gpu(kernel)
    except (OSError, ValueError, IndexError):
        return 0

def _compile_gpu(kernel):
    name = 'rtensor_k%d' % counter.n
    counter.n += 1
    src = to_ttir(kernel, name)
    if not src:
        return 0
    warps = config.num_warps
    if kernel.rowmode and kernel.cols > 0:
        warps = row_warps(row_tile(kernel))
    base = _env('TMPDIR', '/tmp') + '/' + name
    _write(base + '.ttir', src)
    cmd = '%s -P %s %s.ttir %s.ptx %s.meta %s %d' % (
        _env('RTENSOR_PYTHON', 'python3'), _here + '/rtensor_triton.py',
        base, base, base, _env('RTENSOR_CC', '86'), warps)
    if os.system(cmd) != 0:
        return 0
    words = _read(base + '.meta').strip().split(' ')
    kernel.threads = int(words[0])
    kernel.shared = int(words[1])
    kernel.nextra = int(words[2])
    ptx = _read(base + '.ptx')
    p_ptx = rffi.str2charp(ptx)
    p_name = rffi.str2charp(name)
    fn = rt_cuda_load(p_ptx, p_name)
    rffi.free_charp(p_ptx)
    rffi.free_charp(p_name)
    return fn

def needs_zero(kernel):
    if not kernel.sumroot:
        return 0
    root = kernel.nodes[len(kernel.nodes) - 1]
    if root.opcode != SUM:
        return 0
    if kernel.rowmode and root.p != AXIS_ALL:
        return 0
    return 1


def launch_gpu(kernel, inputs):
    nin = len(inputs)
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
    collect_if_needed(n)
    dptrs = lltype.malloc(SIGNEDARRAY, nin, flavor='raw')
    outs = lltype.malloc(SIGNEDARRAY, nout, flavor='raw')
    ok = True
    for k in range(nin):
        dptrs[k] = dev(inputs[k])
        if dptrs[k] == 0:
            ok = False
    outs[0] = rt_cuda_alloc(outlen, needs_zero(kernel)) if ok else 0
    if outs[0] == 0:
        ok = False
    for k in range(1, nout):
        outs[k] = rt_cuda_alloc(n, 0) if ok else 0
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
        result = device_tensor(outlen, outs[0], shape)
        if nout > 1:
            result.extra = lltype.malloc(TENSORARRAY, nout - 1)
            for k in range(1, nout):
                result.extra[k - 1] = device_tensor(n, outs[k], inputs[big].shape)
    lltype.free(dptrs, flavor='raw')
    lltype.free(outs, flavor='raw')
    return result


GA_IM2COL, GA_COL2CHW, GA_MAXPOOL = 0, 1, 2
GA_HEADSPLIT, GA_HEADMERGE = 3, 4


def _shape2(rows, cols):
    shape = lltype.malloc(SHAPEARRAY, 2)
    shape[0] = rows
    shape[1] = cols
    return shape


def view2(a, rows, cols):
    return view(a, _shape2(rows, cols))


def column(rows):
    return new_tensor(rows, _shape2(rows, 1))


class GatherKernel(object):
    def __init__(self, fn, threads, shared, nextra):
        self.fn = fn
        self.threads = threads
        self.shared = shared
        self.nextra = nextra


class GatherCache(object):
    def __init__(self):
        self.kernels = {}
gather_cache = GatherCache()


class Emitter(object):
    def __init__(self):
        self.lines = []
        self.k = 0

    def tmp(self):
        self.k += 1
        return '%%t%d' % self.k

    def add(self, line):
        self.lines.append(line)


def _gconst(e, v, ty):
    r = e.tmp()
    e.add('    %s = arith.constant dense<%d> : %s' % (r, v, ty))
    return r


def _gbin(e, op, a, b, ty):
    r = e.tmp()
    e.add('    %s = arith.%s %s, %s : %s' % (r, op, a, b, ty))
    return r


def _gcmp(e, pred, a, b, ty):
    r = e.tmp()
    e.add('    %s = arith.cmpi %s, %s, %s : %s' % (r, pred, a, b, ty))
    return r


def _gdiv(e, a, v, ty):
    return _gbin(e, 'divsi', a, _gconst(e, v, ty), ty)


def _gmod(e, a, v, ty):
    return _gbin(e, 'remsi', a, _gconst(e, v, ty), ty)


def _gmul(e, a, v, ty):
    return _gbin(e, 'muli', a, _gconst(e, v, ty), ty)


def _gaddc(e, a, v, ty):
    return _gbin(e, 'addi', a, _gconst(e, v, ty), ty)


def _gather_index(e, op, params, I64, I1):
    off = '%offs64'
    srcs = []
    if op == GA_IM2COL:
        n, c, h, w, k, pad = (params[0], params[1], params[2], params[3],
                              params[4], params[5])
        hw = h * w
        kk = k * k
        ckk = c * kk
        row = _gdiv(e, off, ckk, I64)
        col = _gmod(e, off, ckk, I64)
        img = _gdiv(e, row, hw, I64)
        pos = _gmod(e, row, hw, I64)
        ph = _gdiv(e, pos, w, I64)
        pw = _gmod(e, pos, w, I64)
        ch = _gdiv(e, col, kk, I64)
        rk = _gmod(e, col, kk, I64)
        ih = _gaddc(e, _gbin(e, 'addi', ph, _gdiv(e, rk, k, I64), I64),
                    -pad, I64)
        iw = _gaddc(e, _gbin(e, 'addi', pw, _gmod(e, rk, k, I64), I64),
                    -pad, I64)
        zero = _gconst(e, 0, I64)
        ok = _gbin(e, 'andi',
                   _gbin(e, 'andi', _gcmp(e, 'sge', ih, zero, I64),
                         _gcmp(e, 'slt', ih, _gconst(e, h, I64), I64), I1),
                   _gbin(e, 'andi', _gcmp(e, 'sge', iw, zero, I64),
                         _gcmp(e, 'slt', iw, _gconst(e, w, I64), I64), I1), I1)
        base = _gbin(e, 'addi', _gmul(e, img, c * hw, I64),
                     _gmul(e, ch, hw, I64), I64)
        src = _gbin(e, 'addi', base,
                    _gbin(e, 'addi', _gmul(e, ih, w, I64), iw, I64), I64)
        srcs.append(src)
        return srcs, ok
    if op == GA_COL2CHW:
        n, hw, o = params[0], params[1], params[2]
        img = _gdiv(e, off, o * hw, I64)
        rem = _gmod(e, off, o * hw, I64)
        ch = _gdiv(e, rem, hw, I64)
        pos = _gmod(e, rem, hw, I64)
        row = _gbin(e, 'addi', _gmul(e, img, hw, I64), pos, I64)
        srcs.append(_gbin(e, 'addi', _gmul(e, row, o, I64), ch, I64))
        return srcs, ''
    if op == GA_HEADSPLIT or op == GA_HEADMERGE:
        rows, dh, heads = params[0], params[1], params[2]
        if op == GA_HEADSPLIT:
            ostride, sstride = rows * dh, heads * dh
        else:
            ostride, sstride = heads * dh, rows * dh
        oi = _gdiv(e, off, ostride, I64)
        rem = _gmod(e, off, ostride, I64)
        ii = _gdiv(e, rem, dh, I64)
        ci = _gmod(e, rem, dh, I64)
        srcs.append(_gbin(e, 'addi',
                          _gbin(e, 'addi', _gmul(e, ii, sstride, I64),
                                _gmul(e, oi, dh, I64), I64), ci, I64))
        return srcs, ''
    n, c, h, w = params[0], params[1], params[2], params[3]
    oh = h // 2
    ow = w // 2
    pw = _gmod(e, off, ow, I64)
    q = _gdiv(e, off, ow, I64)
    ph = _gmod(e, q, oh, I64)
    q2 = _gdiv(e, q, oh, I64)
    ch = _gmod(e, q2, c, I64)
    img = _gdiv(e, q2, c, I64)
    base = _gbin(e, 'addi', _gmul(e, img, c * h * w, I64),
                 _gmul(e, ch, h * w, I64), I64)
    base = _gbin(e, 'addi', base, _gmul(e, ph, 2 * w, I64), I64)
    base = _gbin(e, 'addi', base, _gmul(e, pw, 2, I64), I64)
    srcs.append(base)
    srcs.append(_gaddc(e, base, 1, I64))
    srcs.append(_gaddc(e, base, w, I64))
    srcs.append(_gaddc(e, base, w + 1, I64))
    return srcs, ''


def to_ttir_gather(op, params, name):
    BLOCK = config.block
    T = 'tensor<%dxf64>' % BLOCK
    P = 'tensor<%dx!tt.ptr<f64>>' % BLOCK
    I32 = 'tensor<%dxi32>' % BLOCK
    I64 = 'tensor<%dxi64>' % BLOCK
    I1 = 'tensor<%dxi1>' % BLOCK
    e = Emitter()
    e.add('module {')
    e.add('  tt.func public @%s(%%in: !tt.ptr<f64>, %%out: !tt.ptr<f64>, '
          '%%n: i64, %%c: i64) attributes {noinline = false} {' % name)
    e.add('    %%zero = arith.constant dense<0.0> : %s' % T)
    e.add('    %%bs = arith.constant %d : i32' % BLOCK)
    e.add('    %pid = tt.get_program_id x : i32')
    e.add('    %start = arith.muli %pid, %bs : i32')
    e.add('    %%range = tt.make_range {end = %d : i32, start = 0 : i32} : %s'
          % (BLOCK, I32))
    e.add('    %%starts = tt.splat %%start : i32 -> %s' % I32)
    e.add('    %%offs = arith.addi %%starts, %%range : %s' % I32)
    e.add('    %%offs64 = arith.extsi %%offs : %s to %s' % (I32, I64))
    e.add('    %%ns = tt.splat %%n : i64 -> %s' % I64)
    e.add('    %%mask = arith.cmpi slt, %%offs64, %%ns : %s' % I64)
    srcs, ok = _gather_index(e, op, params, I64, I1)
    mask = '%mask'
    if ok:
        mask = _gbin(e, 'andi', mask, ok, I1)
    e.add('    %%pin = tt.splat %%in : !tt.ptr<f64> -> %s' % P)
    vals = []
    for i in range(len(srcs)):
        q = e.tmp()
        e.add('    %s = tt.addptr %%pin, %s : %s, %s' % (q, srcs[i], P, I64))
        v = e.tmp()
        e.add('    %s = tt.load %s, %s, %%zero : %s' % (v, q, mask, P))
        vals.append(v)
    acc = vals[0]
    for i in range(1, len(vals)):
        acc = _gbin(e, 'maximumf', acc, vals[i], T)
    e.add('    %%pout = tt.splat %%out : !tt.ptr<f64> -> %s' % P)
    e.add('    %%qout = tt.addptr %%pout, %%offs : %s, %s' % (P, I32))
    e.add('    tt.store %%qout, %s, %%mask : %s' % (acc, P))
    e.add('    tt.return')
    e.add('  }')
    e.add('}')
    return '\n'.join(e.lines) + '\n'


def gather_key(op, params):
    parts = ['g%d' % op]
    for i in range(len(params)):
        parts.append(str(params[i]))
    return ','.join(parts)


def gather_kernel(op, params):
    key = gather_key(op, params)
    k = gather_cache.kernels.get(key, None)
    if k is None:
        k = _gather_compile(op, params)
        gather_cache.kernels[key] = k
    return k


def _gather_compile(op, params):
    if not gpu_enabled():
        return GatherKernel(0, 0, 0, 0)
    try:
        return _gather_compile_gpu(op, params)
    except (OSError, ValueError, IndexError):
        return GatherKernel(0, 0, 0, 0)


def _gather_compile_gpu(op, params):
    name = 'rtensor_g%d' % counter.n
    counter.n += 1
    src = to_ttir_gather(op, params, name)
    base = _env('TMPDIR', '/tmp') + '/' + name
    _write(base + '.ttir', src)
    cmd = '%s -P %s %s.ttir %s.ptx %s.meta %s %d' % (
        _env('RTENSOR_PYTHON', 'python3'), _here + '/rtensor_triton.py',
        base, base, base, _env('RTENSOR_CC', '86'), config.num_warps)
    if os.system(cmd) != 0:
        return GatherKernel(0, 0, 0, 0)
    words = _read(base + '.meta').strip().split(' ')
    threads = int(words[0])
    shared = int(words[1])
    nextra = int(words[2])
    ptx = _read(base + '.ptx')
    p_ptx = rffi.str2charp(ptx)
    p_name = rffi.str2charp(name)
    fn = rt_cuda_load(p_ptx, p_name)
    rffi.free_charp(p_ptx)
    rffi.free_charp(p_name)
    return GatherKernel(fn, threads, shared, nextra)


def gather_gpu(op, params, x, outn, shape):
    kernel = gather_kernel(op, params)
    if kernel.fn == 0:
        return NULLTENSOR
    dptr = dev(x)
    if dptr == 0:
        return NULLTENSOR
    collect_if_needed(outn)
    ins = lltype.malloc(SIGNEDARRAY, 1, flavor='raw')
    outs = lltype.malloc(SIGNEDARRAY, 1, flavor='raw')
    ins[0] = dptr
    outs[0] = rt_cuda_alloc(outn, 0)
    ok = outs[0] != 0
    if ok:
        ok = rffi.cast(lltype.Signed, rt_cuda_launch(
            kernel.fn, ins, rffi.cast(rffi.INT, 1), outn, outs,
            rffi.cast(rffi.INT, 1), rffi.cast(rffi.INT, kernel.threads),
            config.block, rffi.cast(rffi.INT, kernel.shared),
            rffi.cast(rffi.INT, kernel.nextra), 0)) != 0
    result = NULLTENSOR
    if ok:
        result = device_tensor(outn, outs[0], shape)
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
    r = new_tensor(rows * cols, _shape2(rows, cols))
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
def im2col(x, c, h, w, k, pad):
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
    r = new_tensor(n * o * hw, _shape2(n, o * hw))
    hr = r.host
    for i in range(n * o * hw):
        img = i // (o * hw)
        rem = i % (o * hw)
        idx = (img * hw + rem % hw) * o + rem // hw
        assert idx >= 0
        hr[i] = hy[idx]
    return r


@jit.dont_look_inside
def col2chw(y, n, hw, o):
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
    r = new_tensor(outn, _shape2(n, c * oh * ow))
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
def maxpool2(x, c, h, w):
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
    r = new_tensor(rows * dh * heads, _shape2(heads * rows, dh))
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
    r = new_tensor(rows * dh * heads, _shape2(rows, heads * dh))
    hr = r.host
    for i in range(rows * dh * heads):
        ri = i // (heads * dh)
        rem = i % (heads * dh)
        idx = rem // dh * (rows * dh) + ri * dh + rem % dh
        assert idx >= 0
        hr[i] = hx[idx]
    return r


@jit.dont_look_inside
def head_split(x, rows, dh, heads):
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
def head_merge(x, rows, dh, heads):
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
    r = new_tensor(batch * rows * cols, _shape2(batch * rows, cols))
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
def tensor_bmm(a, b, batch, rows, cols, inner, ta, tb):
    if gpu_enabled():
        dptr_a = dev(a)
        dptr_b = dev(b)
        if dptr_a != 0 and dptr_b != 0:
            n = batch * rows * cols
            collect_if_needed(n)
            outptr = rt_cuda_alloc(n, 0)
            if outptr != 0:
                ok = rffi.cast(lltype.Signed, rt_cuda_bmm(
                    dptr_a, dptr_b, outptr, batch, rows, inner, cols,
                    ta, tb)) != 0
                if ok:
                    return device_tensor(n, outptr,
                                         _shape2(batch * rows, cols))
                rt_cuda_free(outptr, n)
    return bmm_cpu(a, b, batch, rows, cols, inner, ta, tb)
