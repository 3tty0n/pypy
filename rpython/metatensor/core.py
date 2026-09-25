import os

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
                              ('buf', OBJECTPTR),
                              ('lazy', OBJECTPTR)))
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
# A permutation of its operand's elements: rot_half, head_split, head_merge.
# The param packs which one and its sizes (gather_param below), so the node
# is fully described by (opcode, param) like every other node.
GATHER = 11
# Elementwise math.  UNARY's param is the function, a U_* below.  BINARY's
# param packs the function above the broadcast mode, p = bcast + NPARAMS * fn
# with fn a B_*, so p % NPARAMS is the BC_* of any binary node and the kernel
# key still names the code.
UNARY, BINARY = 12, 13
NOPCODES = 14
ARITY = [2, 2, 1, 1, 2, 2, 2, 1, 1, 1, 2, 1, 1, 2]
NAMES = ['add', 'mul', 'relu', 'sum', 'relugrad',
         'sub', 'div', 'exp', 'sqrt', 'maxr', 'eqmask', 'gather',
         'unary', 'binary']
HAS_PARAM = [True, True, False, True, True,
             True, True, False, False, True, True, True, True, True]
U_TANH, U_SIGMOID, U_LOG, U_ABS, U_SIN, U_COS, U_ERF, U_FLOOR, U_NEG = range(9)
UNARY_NAMES = ['tanh', 'sigmoid', 'log', 'abs', 'sin', 'cos', 'erf', 'floor',
               'neg']
B_MAX, B_MIN, B_POW, B_LT, B_LE, B_GT, B_GE, B_EQ, B_NE = range(9)
# keep(c, x): x where c != 0 (KEEP_NZ) or where c == 0 (KEEP_Z), else 0.  A
# select, so an inf or NaN in the lanes it drops stays dropped; where(c, a, b)
# is keep_nz(c, a) + keep_z(c, b).
B_KEEP_NZ, B_KEEP_Z = 9, 10
BINARY_NAMES = ['maximum', 'minimum', 'pow', 'lt', 'le', 'gt', 'ge', 'eq',
                'ne', 'keep_nz', 'keep_z']
# Hard ceiling on a fused kernel's input slots.  It is a compile-time
# constant because the tensor.launch oopspec has that many tensor arguments;
# how many of them the fusion pass is actually allowed to use is the runtime
# knob below, so a MAX_INPUTS sensitivity sweep needs no retranslation.
MAX_INPUTS_LIMIT = 8
DEFAULT_MAX_INPUTS = 6

class _Knobs(object):
    max_inputs = 0
_knobs = _Knobs()

class _LazyKnob(object):
    # Quasi-immutable: every fusible operation asks whether execution is
    # deferred, and with the pass on the answer is a compile-time constant,
    # so the question costs one guard_not_invalidated per trace and nothing
    # per operation.
    _immutable_fields_ = ['on?']
    on = False
lazy_knob = _LazyKnob()

class _DrainKnob(object):
    # Which way a value leaves a fused region when something downstream needs
    # it and the region has already been launched.
    #
    #   off  keep the descriptor: the launched kernel gains one output, so
    #        the device buffers, the input set and the layout all stay, and
    #        only the signature is recompiled.
    #   on   drain to canonical form: forget that the region ran, walk the
    #        operation graph back to its leaves and record a fresh kernel.
    #
    # Read once at device init, so the optimizer's branch on it is a constant
    # for the whole process.
    _immutable_fields_ = ['on?']
    on = False
drain_knob = _DrainKnob()

class _GatherKnob(object):
    # METATENSOR_NO_GATHER_FUSION=1: rot_half, head_split and head_merge run
    # as standalone gather kernels, as they did before GATHER was a fusion
    # node - the ablation of that node on the same binary.
    _immutable_fields_ = ['off?']
    off = False
gather_knob = _GatherKnob()

def init_gather():
    value = os.environ.get('METATENSOR_NO_GATHER_FUSION')
    gather_knob.off = value is not None and value != '' and value != '0'

def init_lazy():
    """Read METATENSOR_LAZY once, at device init."""
    value = os.environ.get('METATENSOR_LAZY')
    lazy_knob.on = value is not None and value != '' and value != '0'

def init_drain():
    """Read METATENSOR_DRAIN once, at device init."""
    value = os.environ.get('METATENSOR_DRAIN')
    drain_knob.on = value is not None and value != '' and value != '0'

def max_inputs():
    """Leaves a fused kernel may take, from RTENSOR_MAX_INPUTS (4..8).

    Default 6, which is what it was before the knob existed.  The value is
    returned from the local, not re-read from the cache, so the answer is
    right on the first call as well as on every later one."""
    n = _knobs.max_inputs
    if n != 0:
        return n
    n = DEFAULT_MAX_INPUTS
    value = os.environ.get('RTENSOR_MAX_INPUTS')
    if value is not None and len(value) > 0:
        try:
            n = int(value)
        except ValueError:
            n = DEFAULT_MAX_INPUTS
        if n < 4:
            n = 4
        elif n > MAX_INPUTS_LIMIT:
            n = MAX_INPUTS_LIMIT
    _knobs.max_inputs = n
    return n

BC_NONE, BC_R_ROW, BC_R_SCALAR, BC_L_ROW, BC_L_SCALAR = 0, 1, 2, 3, 4
BC_R_COL, BC_L_COL = 5, 6
NPARAMS = 7

def bc_mode(opcode, p):
    if ARITY[opcode] != 2:
        return BC_NONE
    return p % NPARAMS

def binary_fn(p):
    return p // NPARAMS
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
                         ('modes', lltype.Signed),
                         ('outmodes', lltype.Signed),
                         ('nouts', lltype.Signed),
                         ('consts', lltype.Ptr(HOSTARRAY)),
                         ('outputs', lltype.Ptr(SHAPEARRAY)),
                         ('wfn', lltype.Signed),
                         ('wthreads', lltype.Signed),
                         ('wshared', lltype.Signed),
                         ('wnextra', lltype.Signed))
KERNELPTR = lltype.Ptr(KERNEL)
NO_CONSTS = lltype.malloc(HOSTARRAY, 0, immortal=True)

def nvals(kernel):
    """Scalar leaves live in the same value-index space as the inputs, right
    after them, but are literals in the generated code instead of pointers."""
    return kernel.ninputs + len(kernel.consts)

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
    t.lazy = lltype.nullptr(OBJECTPTR.TO)
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
    if opcode == GATHER or opcode == UNARY or opcode == BINARY:
        return False
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
    wide = 8192
    wide_warps = 16
    flat = 4096
    num_warps = 8
    cc = None
config = Config()
GA_IM2COL, GA_COL2CHW, GA_MAXPOOL = 0, 1, 2
GA_HEADSPLIT, GA_HEADMERGE = 3, 4
GA_ROWS = 5
GA_ROTHALF = 6
GA_IM2COL_NHWC = 7
GA_MAXPOOL_NHWC = 8
# Standalone only, never GATHER node kinds.
GA_STRIDED = 9
GA_TAKE = 10
GA_POOL = 11
GA_IM2COL2 = 12
GA_IM2COL_T = 13
POOL_AVG, POOL_ADAPTIVE, POOL_DEPTHWISE = 0, 1, 2

def conv_out(h, k, stride, pad, dil):
    return (h + 2 * pad - dil * (k - 1) - 1) // stride + 1

# kind in the low 4 bits, dh in the next 20, heads above.  rows is not stored:
# it is size // (dh * heads), and leaving it out keeps one kernel per layout
# rather than one per sequence length.
def gather_param(kind, dh, heads):
    return kind + ((dh + (heads << 20)) << 4)

def gather_kind(p):
    return p & 15

def gather_dh(p):
    return (p >> 4) & 0xFFFFF

def gather_heads(p):
    return p >> 24

def gather_changes_shape(p):
    k = gather_kind(p)
    return k == GA_HEADSPLIT or k == GA_HEADMERGE

def gather_shape(p, n, like):
    """The shape of a gather's result, given its operand's size and shape."""
    k = gather_kind(p)
    dh = gather_dh(p)
    heads = gather_heads(p)
    if k == GA_HEADSPLIT:
        return _shape2(n // dh, dh)
    if k == GA_HEADMERGE:
        return _shape2(n // (heads * dh), heads * dh)
    return like


def _shape2(rows, cols):
    shape = lltype.malloc(SHAPEARRAY, 2)
    shape[0] = rows
    shape[1] = cols
    return shape


def column(rows, dtype=F64):
    return new_tensor(rows, _shape2(rows, 1), dtype)

