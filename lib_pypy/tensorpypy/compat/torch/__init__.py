"""The part of torch's API the ported suite models use, over _metatensor.

A ported model's source is the upstream file with Python 2 syntax; it keeps
`import torch`, and the harness puts lib_pypy/tensorpypy/compat on sys.path
so that import lands here.  Every choice about how an op runs lives in this
package and applies to every model alike.

A Tensor is a _metatensor tensor holding its elements contiguously in
logical (row-major) order, plus the logical shape.  The engine's kernels see
2-D views of it; which 2-D view an op needs is decided here.  An op whose
operands do not fit a view the engine has raises NotImplementedError naming
the case, rather than computing something else.

Two refinements of that picture:

  perm   transpose/permute do not move data: the tensor keeps its physical
         (contiguous) storage and records, for each logical dim, which
         physical dim it is.  Ops that understand a permutation (attention
         reading q/k/v as [B, S, H, D] storage viewed [B, H, S, D]) take it
         as is; every other op asks for dense() and gets NotImplementedError
         if the permutation would need a copy the engine does not have.
  host   integer tensors (token ids, positions, masks) are host lists: they
         are index data, a few thousand elements computed from shapes and
         inputs, and uploading them only where they index device memory
         keeps their arithmetic off the device.
"""
import _metatensor

_float, _int = float, int

float32 = "float32"
float64 = "float64"
float16 = "float16"
bfloat16 = "bfloat16"
half = float16
double = float64
float = float32
int64 = "int64"
int32 = "int32"
int8 = "int8"
uint8 = "uint8"
long = int64
int = int32
bool = "bool"
__version__ = "2.14.0"


class _Flag(object):
    def __init__(self, value):
        self.allow_tf32 = value


class backends(object):
    class cuda(object):
        matmul = _Flag(False)

    class cudnn(object):
        allow_tf32 = True
        benchmark = False


_scalars = {}


def _scalar(v, dtype):
    key = (v, dtype)
    t = _scalars.get(key)
    if t is None:
        t = _metatensor.scalar(v, dtype)
        _scalars[key] = t
    return t


def _numel(shape):
    n = 1
    for d in shape:
        n *= d
    return n


def _norm_dim(d, nd):
    return d + nd if d < 0 else d


class Size(tuple):
    def numel(self):
        return _numel(self)


class Tensor(object):
    __slots__ = ("raw", "shape", "perm", "host", "_index")

    def __init__(self, raw, shape, perm=None, host=None):
        self.raw = raw
        self.shape = Size(shape)
        self.perm = perm
        self.host = host
        self._index = None

    # -- metadata ---------------------------------------------------------
    @property
    def dtype(self):
        if self.host is not None:
            return int64
        return self.raw.dtype

    @property
    def device(self):
        return "cuda"

    @property
    def is_host(self):
        return self.host is not None

    def dense(self):
        """This tensor with its elements contiguous in logical order."""
        if self.perm is None:
            return self
        raise NotImplementedError("materialising permutation %s of %s"
                                  % (self.perm, list(self.physical_shape())))

    def physical_shape(self):
        if self.perm is None:
            return tuple(self.shape)
        out = [0] * len(self.shape)
        for i, p in enumerate(self.perm):
            out[p] = self.shape[i]
        return tuple(out)

    @property
    def ndim(self):
        return len(self.shape)

    def dim(self):
        return len(self.shape)

    def size(self, d=None):
        if d is None:
            return self.shape
        return self.shape[_norm_dim(d, len(self.shape))]

    def numel(self):
        return _numel(self.shape)

    def __len__(self):
        return self.shape[0]

    def __repr__(self):
        return "tensor(shape=%s, dtype=%s)" % (list(self.shape), self.dtype)

    # -- views ------------------------------------------------------------
    def view(self, *shape):
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = tuple(shape[0])
        n = self.numel()
        shape = list(shape)
        if -1 in shape:
            i = shape.index(-1)
            rest = 1
            for j, d in enumerate(shape):
                if j != i:
                    rest *= d
            shape[i] = n // rest
        if _numel(shape) != n:
            raise ValueError("shape %s is invalid for %d elements"
                             % (shape, n))
        if self.host is not None:
            return Tensor(None, tuple(shape), host=self.host)
        return Tensor(self.dense().raw, tuple(shape))

    reshape = view

    def transpose(self, a, b):
        nd = len(self.shape)
        a, b = _norm_dim(a, nd), _norm_dim(b, nd)
        order = list(range(nd))
        order[a], order[b] = order[b], order[a]
        return self.permute(*order)

    def permute(self, *order):
        if len(order) == 1 and isinstance(order[0], (tuple, list)):
            order = tuple(order[0])
        nd = len(self.shape)
        order = [_norm_dim(d, nd) for d in order]
        if self.host is not None:
            raise NotImplementedError("permute of an index tensor")
        base = self.perm or tuple(range(nd))
        perm = tuple(base[d] for d in order)
        shape = tuple(self.shape[d] for d in order)
        if perm == tuple(range(nd)):
            perm = None
        return Tensor(self.raw, shape, perm)

    def expand(self, *shape):
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = tuple(shape[0])
        cur = (1,) * (len(shape) - len(self.shape)) + tuple(self.shape)
        out = tuple(c if s == -1 else s for s, c in zip(shape, cur))
        if out == cur:
            return self.view(out)
        if self.host is not None:
            return _host_expand(self, cur, out)
        raise NotImplementedError("expand %s -> %s" % (list(cur), list(out)))

    def expand_as(self, other):
        return self.expand(other.shape)

    def __getitem__(self, idx):
        if self.host is not None:
            return _host_getitem(self, idx)
        raise NotImplementedError("indexing a device tensor with %r"
                                  % (idx,))

    def flatten(self, start_dim=0, end_dim=-1):
        nd = len(self.shape)
        s = _norm_dim(start_dim, nd)
        e = _norm_dim(end_dim, nd)
        if nd == 0:
            return self.view(1)
        mid = _numel(self.shape[s:e + 1])
        return self.view(self.shape[:s] + (mid,) + self.shape[e + 1:])

    def contiguous(self):
        return self.dense()

    def detach(self):
        return self

    def to(self, *args, **kwargs):
        for a in args + tuple(kwargs.values()):
            if a in (float32, float64, float16) and a != self.dtype:
                return Tensor(self.raw.astype(a), self.shape)
        return self

    def float(self):
        return self.to(float32)

    def unsqueeze(self, d):
        d = _norm_dim(d, len(self.shape) + 1)
        return self.view(self.shape[:d] + (1,) + self.shape[d:])

    def squeeze(self, d=None):
        if d is None:
            return self.view(tuple(x for x in self.shape if x != 1))
        d = _norm_dim(d, len(self.shape))
        if self.shape[d] != 1:
            return self
        return self.view(self.shape[:d] + self.shape[d + 1:])

    # -- host -------------------------------------------------------------
    def tolist(self):
        if self.host is not None:
            return list(self.host)
        return self.dense().raw.tolist()

    def index_tensor(self, dtype):
        """The device copy of an index tensor, made once."""
        if self._index is None or self._index.dtype != dtype:
            self._index = _metatensor._tensor_flat(
                [_float(v) for v in self.host], [len(self.host)], False, dtype)
        return self._index

    def item(self):
        return self.raw.item()

    def force(self):
        return Tensor(self.raw.force(), self.shape)

    # -- elementwise ------------------------------------------------------
    def __add__(self, other):
        return _binary(self, other, "add")

    __radd__ = __add__
    __iadd__ = __add__

    def __mul__(self, other):
        return _binary(self, other, "mul")

    __rmul__ = __mul__
    __imul__ = __mul__

    def __sub__(self, other):
        return _binary(self, other, "sub")

    __isub__ = __sub__

    def __rsub__(self, other):
        return _binary(_as_tensor(other, self.dtype), self, "sub")

    def __div__(self, other):
        return _binary(self, other, "div")

    __truediv__ = __div__
    __itruediv__ = __div__
    __idiv__ = __div__

    def __rdiv__(self, other):
        return _binary(_as_tensor(other, self.dtype), self, "div")

    __rtruediv__ = __rdiv__

    def __neg__(self):
        return _binary(self, -1.0, "mul")

    def add(self, other, alpha=1):
        if alpha != 1:
            other = other * alpha
        return self + other

    add_ = add

    def mul(self, other):
        return self * other

    mul_ = mul

    def relu(self):
        return Tensor(self.raw.relu(), self.shape, self.perm)

    relu_ = relu

    def exp(self):
        return Tensor(self.raw.exp(), self.shape, self.perm)

    def sqrt(self):
        return Tensor(self.raw.sqrt(), self.shape, self.perm)

    def mean(self, dim=None, keepdim=False):
        if dim is None:
            return Tensor(self.raw.sum(), ()) * (1.0 / self.numel())
        nd = len(self.shape)
        n = _numel([self.shape[_norm_dim(d, nd)] for d in _dims(dim)])
        return _reduce_last(self, dim, keepdim, "sum") * (1.0 / n)

    def sum(self, dim=None, keepdim=False):
        if dim is None:
            return Tensor(self.raw.sum(), ())
        return _reduce_last(self, dim, keepdim, "sum")

    def amax(self, dim=-1, keepdim=False):
        return _reduce_last(self, dim, keepdim, "max")

    def softmax(self, dim=-1):
        return _rowwise(self, dim, "softmax")

    def matmul(self, other):
        return matmul(self, other)

    __matmul__ = matmul


def _dims(dim):
    return list(dim) if isinstance(dim, (tuple, list)) else [dim]


def _reduce_last(x, dim, keepdim, how):
    """Reduce over trailing dimensions: a row reduction of the [rows, cols]
    view with cols = the product of the reduced dims."""
    x = x.dense()
    nd = len(x.shape)
    ds = sorted(_norm_dim(d, nd) for d in _dims(dim))
    if ds != list(range(nd - len(ds), nd)):
        raise NotImplementedError("reduction over non-trailing dims %s of %s"
                                  % (ds, list(x.shape)))
    cols = _numel(x.shape[nd - len(ds):])
    rows = x.numel() // cols
    t2 = x.raw.reshape([rows, cols])
    r = t2.sum(1) if how == "sum" else t2.max(1)
    keep = x.shape[:nd - len(ds)]
    shape = keep + (1,) * len(ds) if keepdim else keep
    return Tensor(r, shape)


def _rowwise(x, dim, how):
    x = x.dense()
    nd = len(x.shape)
    if _norm_dim(dim, nd) != nd - 1:
        raise NotImplementedError("%s over dim %d of %d" % (how, dim, nd))
    cols = x.shape[-1]
    t2 = x.raw.reshape([x.numel() // cols, cols])
    return Tensor(getattr(t2, how)(), x.shape)


def _as_tensor(v, dtype):
    if isinstance(v, Tensor):
        return v
    return Tensor(_scalar(_float(v), dtype), ())


def _broadcast_shape(a, b):
    n = max(len(a), len(b))
    a = (1,) * (n - len(a)) + tuple(a)
    b = (1,) * (n - len(b)) + tuple(b)
    out = []
    for x, y in zip(a, b):
        if x != y and x != 1 and y != 1:
            raise ValueError("shapes %s and %s do not broadcast" % (a, b))
        out.append(max(x, y))
    return tuple(out), a, b


def _binary(a, b, op):
    if not isinstance(b, Tensor):
        if a.host is not None:
            return _host_binary(a, b, op)
        s = _scalar(_float(b), a.dtype)
        return Tensor(getattr(a.raw, op)(s), a.shape, a.perm)
    if not isinstance(a, Tensor):
        a = _as_tensor(a, b.dtype)
    if a.host is not None or b.host is not None:
        return _host_binary(a, b, op)
    if a.shape == b.shape and a.perm == b.perm:
        # elementwise ops do not care how the elements are laid out, only
        # that both operands lay them out the same way
        return Tensor(getattr(a.raw, op)(b.raw), a.shape, a.perm)
    a, b = a.dense(), b.dense()
    if a.shape == b.shape:
        return Tensor(getattr(a.raw, op)(b.raw), a.shape)
    shape, pa, pb = _broadcast_shape(a.shape, b.shape)
    big, small, flip = (a, b, False)
    ps = pb
    if _numel(pa) < _numel(pb):
        big, small, flip, ps = b, a, True, pa
    if _numel(big.shape) != _numel(shape):
        raise NotImplementedError("broadcast of both operands: %s, %s"
                                  % (list(a.shape), list(b.shape)))
    m = small.numel()
    nd = len(shape)
    if m == 1:
        bt = big.raw.reshape([_numel(shape)])
        st = small.raw.reshape([1]) if small.shape != () else small.raw
        view = None
    else:
        # small's non-1 dims must be a contiguous block [lo, hi) of shape
        lo = 0
        while ps[lo] == 1:
            lo += 1
        hi = nd
        while ps[hi - 1] == 1:
            hi -= 1
        for i in range(lo, hi):
            if ps[i] != shape[i]:
                raise NotImplementedError(
                    "broadcast %s against %s" % (list(ps), list(shape)))
        inner = _numel(shape[hi:])
        if inner == 1:
            # trailing block: row broadcast over a [rows, m] view
            bt = big.raw.reshape([_numel(shape) // m, m])
            st = small.raw.reshape([m])
        elif lo == 0:
            # leading block: column broadcast over [m, inner]
            bt = big.raw.reshape([m, inner])
            st = small.raw.reshape([m, 1])
        else:
            raise NotImplementedError(
                "broadcast %s against %s" % (list(ps), list(shape)))
    if flip:
        if op in ("add", "mul"):
            r = getattr(bt, op)(st)
        else:
            return _binary_flipped(small, big, op, shape)
    else:
        r = getattr(bt, op)(st)
    return Tensor(r, shape)


def _binary_flipped(small, big, op, shape):
    # small - big == -(big - small); small / big == 1 / (big / small)
    if op == "sub":
        return -_binary(big, small, "sub")
    if op == "div":
        return 1.0 / _binary(big, small, "div")
    raise NotImplementedError(op)


def matmul(a, b):
    """a @ b for a [..., n] and b a 2-D [n, m]."""
    a, b = a.dense(), b.dense()
    if len(b.shape) != 2:
        raise NotImplementedError("matmul with a %d-D right operand"
                                  % len(b.shape))
    n = a.shape[-1]
    rows = a.numel() // n
    r = a.raw.reshape([rows, n]).matmul(b.raw.reshape([n, b.shape[1]]), False,
                                      backends.cuda.matmul.allow_tf32)
    return Tensor(r, a.shape[:-1] + (b.shape[1],))


def relu(x, inplace=False):
    return x.relu()


def flatten(x, start_dim=0, end_dim=-1):
    return x.flatten(start_dim, end_dim)


def add(a, b):
    return a + b


def mean(x, dim=None, keepdim=False):
    return x.mean(dim, keepdim)


def softmax(x, dim=-1):
    return x.softmax(dim)


def from_flat(values, shape, dtype):
    return Tensor(_metatensor._tensor_flat(values, list(shape) or [1], False,
                                           dtype), tuple(shape))


def zeros(*shape, **kw):
    if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
        shape = tuple(shape[0])
    dtype = kw.get("dtype") or float32
    return Tensor(_metatensor.zeros(list(shape), False, dtype), shape)


# -- index tensors on the host ---------------------------------------------

def _host(values, shape):
    return Tensor(None, tuple(shape), host=list(values))


def _strides(shape):
    st, acc = [], 1
    for d in reversed(shape):
        st.append(acc)
        acc *= d
    return list(reversed(st))


def _host_getitem(x, idx):
    if not isinstance(idx, tuple):
        idx = (idx,)
    if Ellipsis in idx:
        i = idx.index(Ellipsis)
        fill = len(x.shape) - (len(idx) - 1 - idx.count(None))
        idx = idx[:i] + (slice(None),) * fill + idx[i + 1:]
    shape, src = list(x.shape), _strides(x.shape)
    axes = []                  # (start, step, count, stride) per output dim
    base, d = 0, 0
    out_shape = []
    for it in idx:
        if it is None:
            axes.append((0, 0, 1, 0))
            out_shape.append(1)
            continue
        if isinstance(it, slice):
            start, stop, step = it.indices(shape[d])
            n = max(0, (stop - start + (step - (1 if step > 0 else -1)))
                    // step)
            axes.append((start, step, n, src[d]))
            out_shape.append(n)
        else:
            i = _int(it)
            base += (i + shape[d] if i < 0 else i) * src[d]
        d += 1
    for k in range(d, len(shape)):
        axes.append((0, 1, shape[k], src[k]))
        out_shape.append(shape[k])
    vals = []

    def walk(level, off):
        if level == len(axes):
            vals.append(x.host[off])
            return
        start, step, n, stride = axes[level]
        for j in range(n):
            walk(level + 1, off + (start + j * step) * stride)
    walk(0, base)
    return _host(vals, out_shape)


def _host_expand(x, cur, out):
    src = _strides(cur)
    vals = []

    def walk(level, off):
        if level == len(out):
            vals.append(x.host[off])
            return
        for j in range(out[level]):
            walk(level + 1, off + (0 if cur[level] == 1 else j * src[level]))
    walk(0, 0)
    return _host(vals, out)


_HOST_OPS = {
    "add": lambda p, q: p + q, "sub": lambda p, q: p - q,
    "mul": lambda p, q: p * q, "div": lambda p, q: p / _float(q),
}


def _host_binary(a, b, op):
    if isinstance(b, Tensor) and b.host is None or \
            isinstance(a, Tensor) and a.host is None:
        raise NotImplementedError("%s of an index tensor and a device "
                                  "tensor" % op)
    f = _HOST_OPS[op]
    if not isinstance(b, Tensor):
        return _host([f(v, b) for v in a.host], a.shape)
    if a.shape != b.shape:
        shape = _broadcast_shape(a.shape, b.shape)[0]
        a, b = a.expand(shape), b.expand(shape)
    return _host([f(p, q) for p, q in zip(a.host, b.host)], a.shape)


def _is_int_dtype(dtype):
    return dtype in (int64, "int32", bool)


def arange(start, end=None, step=1, dtype=None, device=None):
    if end is None:
        start, end = 0, start
    vals = list(range(start, end, step))
    if dtype is None or _is_int_dtype(dtype):
        return _host(vals, (len(vals),))
    return from_flat([_float(v) for v in vals], (len(vals),), dtype)


def tensor(data, dtype=None, device=None):
    def shape_of(v):
        return (len(v),) + shape_of(v[0]) if isinstance(v, (list, tuple)) \
            else ()

    def flat_of(v, out):
        if isinstance(v, (list, tuple)):
            for e in v:
                flat_of(e, out)
        else:
            out.append(v)
        return out
    shape, vals = shape_of(data), flat_of(data, [])
    if dtype is None:
        dtype = float32 if any(isinstance(v, _float) for v in vals) \
            else int64
    if _is_int_dtype(dtype):
        return _host([_int(v) for v in vals], shape)
    return from_flat([_float(v) for v in vals], shape, dtype)


class finfo(object):
    _MIN = {float32: -3.4028234663852886e+38,
            float64: -1.7976931348623157e+308, float16: -65504.0}

    def __init__(self, dtype):
        self.min = self._MIN[dtype]
        self.max = -self.min


class jit(object):
    @staticmethod
    def is_tracing():
        return False

    @staticmethod
    def is_scripting():
        return False


class no_grad(object):
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def __call__(self, fn):
        return fn


inference_mode = no_grad


def is_tensor(x):
    return isinstance(x, Tensor)


from torch import nn  # noqa: E402
from torch.nn import functional  # noqa: E402,F401
