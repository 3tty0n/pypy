from rpython.rlib import jit
from rpython.rtyper.lltypesystem import lltype
from rpython.rtensor.core import (ADD, AXIS_ALL, BC_L_COL, BC_L_ROW, BC_L_SCALAR, BC_NONE, BC_R_COL, BC_R_ROW, BC_R_SCALAR, DIV, EQMASK, EXP, MAXR, MUL, NDTYPES, NULLTENSOR, RELU, RELUGRAD, SHAPEARRAY, SQRT, SUB, SUM, TENSOR, TENSORARRAY, _shape2, cols, new_tensor, note_cols, note_dtype, note_size, policy)
from rpython.rtensor.device import (host)
from rpython.rtensor.runtime import (_make_ones, eval_op, ones, tensor_assign, tensor_matmul)

def astype(t, dtype):
    if t.dtype == dtype:
        return t
    note_dtype(dtype)
    h = host(t)
    r = new_tensor(t.size, t.shape, dtype)
    for i in range(t.size):
        r.host[i] = h[i]
    return r

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
    r.dtype = a.dtype
    r.buf = a.buf
    return r


@jit.unroll_safe
def ones_like(a):
    n = tensor_size(a)
    nd = tensor_ndim(a)
    dtype = tensor_dtype(a)
    shape = lltype.malloc(SHAPEARRAY, nd)
    for i in range(nd):
        shape[i] = tensor_shape(a, i)
    if n == 1:
        if not ones.one[dtype]:
            ones.one[dtype] = _make_ones(1, dtype)
        return view(ones.one[dtype], shape)
    key = n * NDTYPES + dtype
    t = ones.cache.get(key, NULLTENSOR)
    if not t:
        t = _make_ones(n, dtype)
        ones.cache[key] = t
    return view(t, shape)

def view2(a, rows, cols):
    return view(a, _shape2(rows, cols))



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

@jit.oopspec("tensor.dtype(a)")
def tensor_dtype(a):
    return a.dtype


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
    if tensor_dtype(a) != tensor_dtype(b):
        raise ValueError("dtype mismatch")
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
    return assign(a, add(a, b))

def mul_(a, b):
    return assign(a, mul(a, b))

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

def assign(dst, src):
    if (tensor_size(dst) != tensor_size(src) or
            tensor_dtype(dst) != tensor_dtype(src)):
        raise ValueError("shape mismatch")
    return tensor_assign(dst, src)

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
