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
"""
import _metatensor

float32 = "float32"
float64 = "float64"
float16 = "float16"
int64 = "int64"
long = int64
bool = "bool"


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
    __slots__ = ("t", "shape")

    def __init__(self, t, shape):
        self.t = t
        self.shape = Size(shape)

    # -- metadata ---------------------------------------------------------
    @property
    def dtype(self):
        return self.t.dtype

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
        return Tensor(self.t, tuple(shape))

    reshape = view

    def flatten(self, start_dim=0, end_dim=-1):
        nd = len(self.shape)
        s = _norm_dim(start_dim, nd)
        e = _norm_dim(end_dim, nd)
        if nd == 0:
            return self.view(1)
        mid = _numel(self.shape[s:e + 1])
        return self.view(self.shape[:s] + (mid,) + self.shape[e + 1:])

    def contiguous(self):
        return self

    def detach(self):
        return self

    def to(self, *args, **kwargs):
        for a in args + tuple(kwargs.values()):
            if a in (float32, float64, float16) and a != self.dtype:
                return Tensor(self.t.astype(a), self.shape)
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
        return self.t.tolist()

    def item(self):
        return self.t.item()

    def force(self):
        return Tensor(self.t.force(), self.shape)

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
        return Tensor(self.t.relu(), self.shape)

    relu_ = relu

    def exp(self):
        return Tensor(self.t.exp(), self.shape)

    def sqrt(self):
        return Tensor(self.t.sqrt(), self.shape)

    def mean(self, dim=None, keepdim=False):
        if dim is None:
            return Tensor(self.t.sum(), ()) * (1.0 / self.numel())
        nd = len(self.shape)
        n = _numel([self.shape[_norm_dim(d, nd)] for d in _dims(dim)])
        return _reduce_last(self, dim, keepdim, "sum") * (1.0 / n)

    def sum(self, dim=None, keepdim=False):
        if dim is None:
            return Tensor(self.t.sum(), ())
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
    nd = len(x.shape)
    ds = sorted(_norm_dim(d, nd) for d in _dims(dim))
    if ds != list(range(nd - len(ds), nd)):
        raise NotImplementedError("reduction over non-trailing dims %s of %s"
                                  % (ds, list(x.shape)))
    cols = _numel(x.shape[nd - len(ds):])
    rows = x.numel() // cols
    t2 = x.t.reshape([rows, cols])
    r = t2.sum(1) if how == "sum" else t2.max(1)
    keep = x.shape[:nd - len(ds)]
    shape = keep + (1,) * len(ds) if keepdim else keep
    return Tensor(r, shape)


def _rowwise(x, dim, how):
    nd = len(x.shape)
    if _norm_dim(dim, nd) != nd - 1:
        raise NotImplementedError("%s over dim %d of %d" % (how, dim, nd))
    cols = x.shape[-1]
    t2 = x.t.reshape([x.numel() // cols, cols])
    return Tensor(getattr(t2, how)(), x.shape)


def _as_tensor(v, dtype):
    if isinstance(v, Tensor):
        return v
    return Tensor(_scalar(float(v), dtype), ())


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
        s = _scalar(float(b), a.dtype)
        return Tensor(getattr(a.t, op)(s), a.shape)
    if not isinstance(a, Tensor):
        a = _as_tensor(a, b.dtype)
    if a.shape == b.shape:
        return Tensor(getattr(a.t, op)(b.t), a.shape)
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
        bt = big.t.reshape([_numel(shape)])
        st = small.t.reshape([1]) if small.shape != () else small.t
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
            bt = big.t.reshape([_numel(shape) // m, m])
            st = small.t.reshape([m])
        elif lo == 0:
            # leading block: column broadcast over [m, inner]
            bt = big.t.reshape([m, inner])
            st = small.t.reshape([m, 1])
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
    if len(b.shape) != 2:
        raise NotImplementedError("matmul with a %d-D right operand"
                                  % len(b.shape))
    n = a.shape[-1]
    rows = a.numel() // n
    r = a.t.reshape([rows, n]).matmul(b.t.reshape([n, b.shape[1]]), False,
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
