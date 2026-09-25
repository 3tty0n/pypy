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
  col    split() along the last dim does not copy either: each piece is a
         window of columns (offset, row stride, and how many trailing
         logical dims the window spans) into the same [rows, ld] storage.
         view() may regroup the window's dims, transpose may reorder them,
         and attention reads q/k/v straight out of a packed projection.
  host   integer tensors (token ids, positions, masks) are host lists: they
         are index data, a few thousand elements computed from shapes and
         inputs, and uploading them only where they index device memory
         keeps their arithmetic off the device.
"""
import _metatensor

_float, _int = float, int
_all, _any, _abs, _max, _min, _sum = all, any, abs, max, min, sum

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


def _scalar(v, dtype):
    # _metatensor.scalar promotes the value, so the trace sees a constant
    # and the fusion pass writes it into the kernel as a literal; a cache
    # here would hand the trace a dict lookup instead, and every constant
    # would take one of the fused kernel's few input slots
    return _metatensor.scalar(v, dtype)


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
    __slots__ = ("raw", "shape", "perm", "host", "_index", "col")

    def __init__(self, raw, shape, perm=None, host=None, col=None):
        self.raw = raw
        self.col = col
        # a plain tuple: comparing two Size (tuple subclass) instances is a
        # user-level __eq__ the JIT leaves as a residual call, and a residual
        # call that may force ends every fused region still open
        self.shape = shape if type(shape) is tuple else tuple(shape)
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
        if self.col is not None:
            raise NotImplementedError("materialising a column window %s"
                                      % (self.col,))
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
            return Size(self.shape)
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
        if self.col is not None and self.perm is None:
            return self._window_view(tuple(shape))
        return Tensor(self.dense().raw, tuple(shape))

    reshape = view

    def _window_view(self, shape):
        # the rows (every dim before the window) must stay the rows; the
        # window's own dims may be regrouped
        off, ld, ndw = self.col
        rows = _numel(self.shape[:len(self.shape) - ndw])
        acc, k = 1, len(shape)
        while k > 0 and acc < self.numel() // rows:
            k -= 1
            acc *= shape[k]
        if acc != self.numel() // rows or _numel(shape[:k]) != rows:
            raise NotImplementedError("view %s of a column window over %s"
                                      % (list(shape), list(self.shape)))
        return Tensor(self.raw, shape, col=(off, ld, len(shape) - k))

    def split(self, split_size, dim=0):
        nd = len(self.shape)
        dim = _norm_dim(dim, nd)
        if self.host is not None or dim != nd - 1 or self.perm is not None \
                or self.col is not None:
            raise NotImplementedError("split along dim %d of %s"
                                      % (dim, list(self.shape)))
        ld = self.shape[-1]
        sizes = [split_size] * (ld // split_size) if isinstance(
            split_size, _int) else list(split_size)
        if _sum(sizes) != ld:
            raise ValueError("split sizes %s for %d" % (sizes, ld))
        out, off = [], 0
        for w in sizes:
            out.append(Tensor(self.raw, self.shape[:-1] + (w,),
                              col=(off, ld, 1)))
            off += w
        return tuple(out)

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
        if _same(perm, tuple(range(nd))):
            perm = None
        return Tensor(self.raw, shape, perm, col=self.col)

    def expand(self, *shape):
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = tuple(shape[0])
        cur = (1,) * (len(shape) - len(self.shape)) + tuple(self.shape)
        out = tuple(c if s == -1 else s for s, c in zip(shape, cur))
        if _same(out, cur):
            return self.view(out)
        if self.host is not None:
            return _host_expand(self, cur, out)
        raise NotImplementedError("expand %s -> %s" % (list(cur), list(out)))

    def expand_as(self, other):
        return self.expand(other.shape)

    def __getitem__(self, idx):
        if self.host is not None:
            return _host_getitem(self, idx)
        # indexing that selects everything is a view: full slices, None
        # (a new dim of 1) and a trailing Ellipsis
        items = idx if isinstance(idx, tuple) else (idx,)
        shape, d = [], 0
        for it in items:
            if it is None:
                shape.append(1)
            elif it is Ellipsis:
                shape.extend(self.shape[d:])
                d = len(self.shape)
            elif isinstance(it, slice) and d < len(self.shape) and \
                    it.indices(self.shape[d]) == (0, self.shape[d], 1):
                shape.append(self.shape[d])
                d += 1
            else:
                raise NotImplementedError("indexing a device tensor with %r"
                                          % (idx,))
        shape.extend(self.shape[d:])
        return self.view(tuple(shape)) if len(shape) != len(self.shape) \
            else self

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
    # The common cases - two tensors laid out alike, or a tensor and a
    # number - go straight to the engine from the operator itself.  Every
    # Python call the tracer inlines costs frame bookkeeping in the trace,
    # and a long elementwise expression (pyhpc's equation of state is ~300
    # of them) has to fit in one trace for the fusion pass to see it whole.
    def __add__(self, other):
        if self.host is None:
            if type(other) is Tensor:
                if other.host is None and (
                        other.shape is self.shape or
                        _same(self.shape, other.shape)) and (
                        other.perm is self.perm or
                        _same(self.perm, other.perm)) and (
                        other.col is self.col or
                        _same(self.col, other.col)):
                    return Tensor(self.raw.add(other.raw), self.shape,
                                  self.perm, col=self.col)
            elif type(other) is _float or type(other) is _int:
                return Tensor(self.raw.add(_num(other, self)), self.shape,
                              self.perm, col=self.col)
        return _binary(self, other, "add")

    __radd__ = __add__
    __iadd__ = __add__

    def __mul__(self, other):
        if self.host is None:
            if type(other) is Tensor:
                if other.host is None and (
                        other.shape is self.shape or
                        _same(self.shape, other.shape)) and (
                        other.perm is self.perm or
                        _same(self.perm, other.perm)) and (
                        other.col is self.col or
                        _same(self.col, other.col)):
                    return Tensor(self.raw.mul(other.raw), self.shape,
                                  self.perm, col=self.col)
            elif type(other) is _float or type(other) is _int:
                return Tensor(self.raw.mul(_num(other, self)), self.shape,
                              self.perm, col=self.col)
        return _binary(self, other, "mul")

    __rmul__ = __mul__
    __imul__ = __mul__

    def __sub__(self, other):
        if self.host is None:
            if type(other) is Tensor:
                if other.host is None and (
                        other.shape is self.shape or
                        _same(self.shape, other.shape)) and (
                        other.perm is self.perm or
                        _same(self.perm, other.perm)) and (
                        other.col is self.col or
                        _same(self.col, other.col)):
                    return Tensor(self.raw.sub(other.raw), self.shape,
                                  self.perm, col=self.col)
            elif type(other) is _float or type(other) is _int:
                return Tensor(self.raw.sub(_num(other, self)), self.shape,
                              self.perm, col=self.col)
        return _binary(self, other, "sub")

    __isub__ = __sub__

    def __rsub__(self, other):
        if self.host is None and (type(other) is _float or
                                  type(other) is _int):
            return Tensor(_num(other, self).sub(self.raw), self.shape,
                          self.perm, col=self.col)
        return _binary(_as_tensor(other, self.dtype), self, "sub")

    def __div__(self, other):
        if self.host is None:
            if type(other) is Tensor:
                if other.host is None and (
                        other.shape is self.shape or
                        _same(self.shape, other.shape)) and (
                        other.perm is self.perm or
                        _same(self.perm, other.perm)) and (
                        other.col is self.col or
                        _same(self.col, other.col)):
                    return Tensor(self.raw.div(other.raw), self.shape,
                                  self.perm, col=self.col)
            elif type(other) is _float or type(other) is _int:
                return Tensor(self.raw.div(_num(other, self)), self.shape,
                              self.perm, col=self.col)
        return _binary(self, other, "div")

    __truediv__ = __div__
    __itruediv__ = __div__
    __idiv__ = __div__

    def __rdiv__(self, other):
        if self.host is None and (type(other) is _float or
                                  type(other) is _int):
            return Tensor(_num(other, self).div(self.raw), self.shape,
                          self.perm, col=self.col)
        return _binary(_as_tensor(other, self.dtype), self, "div")

    __rtruediv__ = __rdiv__

    def __neg__(self):
        return self * -1.0

    def add(self, other, alpha=1):
        if alpha != 1:
            other = other * alpha
        return self + other

    add_ = add

    def mul(self, other):
        return self * other

    mul_ = mul

    def relu(self):
        return Tensor(self.raw.relu(), self.shape, self.perm, col=self.col)

    relu_ = relu

    def exp(self):
        return Tensor(self.raw.exp(), self.shape, self.perm, col=self.col)

    def _unary(self, name):
        return Tensor(self.raw.unary(name), self.shape, self.perm,
                      col=self.col)

    def tanh(self):
        return self._unary("tanh")

    def sigmoid(self):
        return self._unary("sigmoid")

    def log(self):
        return self._unary("log")

    def abs(self):
        return self._unary("abs")

    __abs__ = abs

    def sin(self):
        return self._unary("sin")

    def cos(self):
        return self._unary("cos")

    def erf(self):
        return self._unary("erf")

    def floor(self):
        return self._unary("floor")

    def rsqrt(self):
        return 1.0 / self.sqrt()

    def reciprocal(self):
        return 1.0 / self

    def __pow__(self, other):
        return _binary(self, other, "pow")

    def __rpow__(self, other):
        return _binary(_as_tensor(other, self.dtype), self, "pow")

    pow = __pow__

    def maximum(self, other):
        return _binary(self, other, "maximum")

    def minimum(self, other):
        return _binary(self, other, "minimum")

    def __lt__(self, other):
        return _binary(self, other, "lt")

    def __le__(self, other):
        return _binary(self, other, "le")

    def __gt__(self, other):
        return _binary(self, other, "gt")

    def __ge__(self, other):
        return _binary(self, other, "ge")

    def eq(self, other):
        return _binary(self, other, "eq")

    def ne(self, other):
        return _binary(self, other, "ne")

    __eq__ = eq
    __ne__ = ne
    __hash__ = object.__hash__

    # -- index-tensor reductions and casts ----------------------------------
    def _host_only(self, what):
        if self.host is None:
            raise NotImplementedError("%s of a device tensor" % what)

    def all(self):
        self._host_only("all")
        return _all(self.host)

    def any(self):
        self._host_only("any")
        return _any(self.host)

    def cumsum(self, dim, dtype=None):
        self._host_only("cumsum")
        nd = len(self.shape)
        dim = _norm_dim(dim, nd)
        st = _strides(self.shape)
        out = list(self.host)
        n = self.shape[dim]
        for flat in range(len(out)):
            if (flat // st[dim]) % n:
                out[flat] += out[flat - st[dim]]
        return _host(out, self.shape)

    def long(self):
        if self.host is None:
            raise NotImplementedError("long() of a device tensor")
        return self

    int = long

    def type_as(self, other):
        if self.host is not None and other.host is not None:
            return self
        raise NotImplementedError("type_as across host and device")

    def clamp(self, min=None, max=None):
        x = self
        if min is not None:
            x = x.maximum(min)
        if max is not None:
            x = x.minimum(max)
        return x

    clamp_min = lambda self, v: self.clamp(min=v)
    clamp_max = lambda self, v: self.clamp(max=v)

    def sqrt(self):
        return Tensor(self.raw.sqrt(), self.shape, self.perm, col=self.col)

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


def _num(v, like):
    return _metatensor.scalar(_float(v), like.raw.dtype)


def _same(p, q):
    """p == q for two shapes or permutations, as an element loop: the tracer
    unrolls it into int compares, where tuple == would be a residual call
    that may force, and a forcing call closes every open fused region."""
    if p is q:
        return True
    if p is None or q is None or len(p) != len(q):
        return False
    for i in range(len(p)):
        if p[i] != q[i]:
            return False
    return True


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
    n = _max(len(a), len(b))
    a = (1,) * (n - len(a)) + tuple(a)
    b = (1,) * (n - len(b)) + tuple(b)
    out = []
    for x, y in zip(a, b):
        if x != y and x != 1 and y != 1:
            raise ValueError("shapes %s and %s do not broadcast" % (a, b))
        out.append(_max(x, y))
    return tuple(out), a, b


class _engine(object):
    """The engine call behind each binary op name.  _apply looks the name up
    as a class attribute, which the JIT folds for a constant name; a string
    comparison or a dict lookup would stay in every trace."""
    add = staticmethod(lambda x, y: x.add(y))
    mul = staticmethod(lambda x, y: x.mul(y))
    sub = staticmethod(lambda x, y: x.sub(y))
    div = staticmethod(lambda x, y: x.div(y))
    pow = staticmethod(lambda x, y: x.binary("pow", y))
    maximum = staticmethod(lambda x, y: x.binary("maximum", y))
    minimum = staticmethod(lambda x, y: x.binary("minimum", y))
    lt = staticmethod(lambda x, y: x.binary("lt", y))
    le = staticmethod(lambda x, y: x.binary("le", y))
    gt = staticmethod(lambda x, y: x.binary("gt", y))
    ge = staticmethod(lambda x, y: x.binary("ge", y))
    eq = staticmethod(lambda x, y: x.binary("eq", y))
    ne = staticmethod(lambda x, y: x.binary("ne", y))


def _apply(x, op, y):
    return getattr(_engine, op)(x, y)


def _binary(a, b, op):
    if not isinstance(b, Tensor):
        if a.host is not None:
            return _host_binary(a, b, op)
        s = _scalar(_float(b), a.dtype)
        return Tensor(_apply(a.raw, op, s), a.shape, a.perm, col=a.col)
    if not isinstance(a, Tensor):
        a = _as_tensor(a, b.dtype)
    if a.host is not None or b.host is not None:
        return _host_binary(a, b, op)
    if _same(a.shape, b.shape) and _same(a.perm, b.perm) and \
            _same(a.col, b.col):
        # elementwise ops do not care how the elements are laid out, only
        # that both operands lay them out the same way
        return Tensor(_apply(a.raw, op, b.raw), a.shape, a.perm, col=a.col)
    a, b = a.dense(), b.dense()
    if _same(a.shape, b.shape):
        return Tensor(_apply(a.raw, op, b.raw), a.shape)
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
    # the engine broadcasts either operand (its BC_L_* modes), so the
    # operands go in their own order
    r = _apply(st, op, bt) if flip else _apply(bt, op, st)
    return Tensor(r, shape)


def matmul(a, b):
    """a @ b for a [..., n] and b a 2-D [n, m]."""
    a, b = a.dense(), b.dense()
    if len(b.shape) != 2:
        raise NotImplementedError("matmul with a %d-D right operand"
                                  % len(b.shape))
    n = a.shape[-1]
    rows = a.numel() // n
    r = _as2d(a, rows, n).matmul(_as2d(b, n, b.shape[1]), False,
                                 backends.cuda.matmul.allow_tf32)
    return Tensor(r, a.shape[:-1] + (b.shape[1],))


def _as2d(x, rows, cols):
    """x's storage as [rows, cols], without a new view when it already is
    one: a view of a weight not yet on the device uploads a copy of its own,
    and the weight, never uploaded itself, is copied again next time."""
    if len(x.shape) == 2 and x.shape[0] == rows:
        return x.raw
    return x.raw.reshape([rows, cols])


def relu(x, inplace=False):
    return x.relu()


def addmm(bias, a, b, beta=1, alpha=1):
    """bias + a @ b, for b in [in, out] as GPT-2's Conv1D keeps it."""
    if beta != 1 or alpha != 1:
        raise NotImplementedError("addmm with beta/alpha")
    return matmul(a, b) + bias


def empty(*shape, **kw):
    return zeros(*shape, **kw)


def tanh(x):
    return x.tanh()


def sigmoid(x):
    return x.sigmoid()


def log(x):
    return x.log()


def exp(x):
    return x.exp()


def sqrt(x):
    return x.sqrt()


def abs(x):
    return x.abs()


def sin(x):
    return x.sin()


def cos(x):
    return x.cos()


def rsqrt(x):
    return x.rsqrt()


def reciprocal(x):
    return x.reciprocal()


def pow(x, y):
    return x ** y if isinstance(x, Tensor) else _as_tensor(x, y.dtype) ** y


def maximum(a, b):
    return a.maximum(b)


def minimum(a, b):
    return a.minimum(b)


def clamp(x, min=None, max=None):
    return x.clamp(min, max)


def where(cond, a, b):
    if not isinstance(a, Tensor) and not isinstance(b, Tensor):
        raise NotImplementedError("where with two scalars")
    dt = a.dtype if isinstance(a, Tensor) else b.dtype
    a = _as_tensor(a, dt)
    b = _as_tensor(b, dt)
    shape = _broadcast_shape(cond.shape,
                             _broadcast_shape(a.shape, b.shape)[0])[0]
    for t in (cond, a, b):
        if not (_same(t.shape, shape) or t.shape == ()) or \
                t.perm is not None:
            raise NotImplementedError("where over %s, %s, %s"
                                      % (list(cond.shape), list(a.shape),
                                         list(b.shape)))
    return Tensor(cond.raw.where(a.raw, b.raw), shape)


def cumsum(x, dim, dtype=None):
    return x.cumsum(dim, dtype)


def all(x):
    return x.all()


def any(x):
    return x.any()


def eq(a, b):
    return a.eq(b)


def ne(a, b):
    return a.ne(b)


def lt(a, b):
    return a < b


def gt(a, b):
    return a > b


def le(a, b):
    return a <= b


def ge(a, b):
    return a >= b


def mul(a, b):
    return a * b


def sub(a, b):
    return a - b


def div(a, b):
    return a / b


def ones_like(x, dtype=None):
    return zeros_like(x, dtype) + 1.0


def zeros_like(x, dtype=None):
    return zeros(*x.shape, dtype=dtype or x.dtype)


def empty_like(x, dtype=None):
    return zeros_like(x, dtype)


def flatten(x, start_dim=0, end_dim=-1):
    return x.flatten(start_dim, end_dim)


def add(a, b):
    return a + b


def mean(x, dim=None, keepdim=False):
    return x.mean(dim, keepdim)


def softmax(x, dim=-1):
    return x.softmax(dim)


_shapes = {}


def canonical_shape(shape):
    """One tuple object per distinct shape, so the elementwise fast path can
    compare shapes by identity: ops hand their operand's shape object on."""
    shape = tuple(shape)
    return _shapes.setdefault(shape, shape)


def from_flat(values, shape, dtype):
    return Tensor(_metatensor._tensor_flat(values, list(shape) or [1], False,
                                           dtype), canonical_shape(shape))


def zeros(*shape, **kw):
    if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
        shape = tuple(shape[0])
    dtype = kw.get("dtype") or float32
    if _is_int_dtype(dtype):
        return _host([0] * _numel(shape), shape)
    return Tensor(_metatensor.zeros(list(shape), False, dtype), shape)


def gather(x, dim, index):
    """torch.gather; index tensors only (the embedding-side lookups upstream
    runs on its integer buffers)."""
    if x.host is None or index.host is None:
        raise NotImplementedError("gather on device tensors")
    nd = len(x.shape)
    dim = _norm_dim(dim, nd)
    xs, ist = _strides(x.shape), _strides(index.shape)
    out = []
    for flat in range(index.numel()):
        pos, rem = [], flat
        for s in ist:
            pos.append(rem // s)
            rem %= s
        pos[dim] = index.host[flat]
        out.append(x.host[sum(p * s for p, s in zip(pos, xs))])
    return _host(out, index.shape)


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
            n = _max(0, (stop - start + (step - (1 if step > 0 else -1)))
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
    "lt": lambda p, q: _int(p < q), "le": lambda p, q: _int(p <= q),
    "gt": lambda p, q: _int(p > q), "ge": lambda p, q: _int(p >= q),
    "eq": lambda p, q: _int(p == q), "ne": lambda p, q: _int(p != q),
    "maximum": _max, "minimum": _min, "pow": lambda p, q: p ** q,
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
        dtype = float32 if _any(isinstance(v, _float) for v in vals) \
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
