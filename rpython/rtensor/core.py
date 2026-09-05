from rpython.rlib import jit
from rpython.rlib.rfloat import INFINITY
from rpython.rtyper.lltypesystem import lltype
from rpython.rtyper.rclass import OBJECTPTR

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
                              ('dtype', lltype.Signed),
                              ('buf', OBJECTPTR)))
NULLTENSOR = lltype.nullptr(TENSOR)

F64, F32, F16 = 0, 1, 2
NDTYPES = 3
DTYPE_BYTES = [8, 4, 2]
DTYPE_NAMES = ['float64', 'float32', 'float16']
STORE_TYPE = ['f64', 'f32', 'f16']
COMP_TYPE = ['f64', 'f32', 'f32']
COMP_NEG_INF = ['0xFFF0000000000000', '0xFF800000', '0xFF800000']

def dtype_of_name(name):
    for i in range(NDTYPES):
        if DTYPE_NAMES[i] == name:
            return i
    raise ValueError("unknown dtype")

def nbytes(n, dtype):
    return n * DTYPE_BYTES[dtype]

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
                         ('dtype', lltype.Signed),
                         ('outputs', lltype.Ptr(SHAPEARRAY)))
KERNELPTR = lltype.Ptr(KERNEL)

def _shape1(n):
    shape = lltype.malloc(SHAPEARRAY, 1)
    shape[0] = n
    return shape

def new_tensor(n, shape=lltype.nullptr(SHAPEARRAY), dtype=F64):
    t = lltype.malloc(TENSOR)
    t.size = n
    t.shape = shape if shape else _shape1(n)
    t.dptr = 0
    t.host = lltype.malloc(HOSTARRAY, n)
    t.extra = lltype.nullptr(TENSORARRAY)
    t.dtype = dtype
    t.buf = lltype.nullptr(OBJECTPTR.TO)
    return t

def zeros(shape_list, dtype=F64):
    n = 1
    for d in shape_list:
        n *= d
    shape = lltype.malloc(SHAPEARRAY, len(shape_list))
    for i in range(len(shape_list)):
        shape[i] = shape_list[i]
    note_dtype(dtype)
    return new_tensor(n, shape, dtype)

def from_list(values, dtype=F64):
    note_dtype(dtype)
    t = new_tensor(len(values), lltype.nullptr(SHAPEARRAY), dtype)
    for i in range(len(values)):
        t.host[i] = values[i]
    return t

def cols(t):
    nd = len(t.shape)
    if nd > 1:
        c = t.shape[nd - 1]
        if c > 0:
            return c
    return 1

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

class SizePolicy(object):
    _immutable_fields_ = ['static?', 'static_cols?', 'dtype?']
    def __init__(self):
        self.static = True
        self.seen = []
        self.static_cols = True
        self.seen_cols = []
        self.dtype = F64
policy = SizePolicy()

def note_dtype(dtype):
    if policy.dtype != dtype:
        policy.dtype = dtype
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

class Config(object):
    block = 4096
    num_warps = 8
config = Config()
GA_IM2COL, GA_COL2CHW, GA_MAXPOOL = 0, 1, 2
GA_HEADSPLIT, GA_HEADMERGE = 3, 4


def _shape2(rows, cols):
    shape = lltype.malloc(SHAPEARRAY, 2)
    shape[0] = rows
    shape[1] = cols
    return shape


def column(rows, dtype=F64):
    return new_tensor(rows, _shape2(rows, 1), dtype)

