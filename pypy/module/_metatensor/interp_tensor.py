from rpython.metatensor import core, device, kernels, nn, ops, runtime
from pypy.interpreter.baseobjspace import W_Root
from pypy.interpreter.typedef import TypeDef, GetSetProperty
from pypy.interpreter.gateway import interp2app, unwrap_spec
from pypy.interpreter.error import oefmt


def _floats_w(space, w_list):
    return [space.float_w(w_item) for w_item in space.listview(w_list)]


def _ints_w(space, w_list):
    return [space.int_w(w_item) for w_item in space.listview(w_list)]


def _shape_w(space, shape):
    return space.newtuple([space.newint(d) for d in shape])


def _mismatch(space):
    return oefmt(space.w_ValueError, "shape mismatch")


def _wrap(t):
    return W_Tensor(nn.Tensor(t))


class W_Tensor(W_Root):
    _immutable_fields_ = ['tensor']

    def __init__(self, tensor):
        self.tensor = tensor

    def _other(self, space, w_other):
        other = space.interp_w(W_Tensor, w_other)
        return other.tensor

    def descr_add(self, space, w_other):
        try:
            return W_Tensor(self.tensor.add(self._other(space, w_other)))
        except ValueError:
            raise _mismatch(space)

    def descr_mul(self, space, w_other):
        try:
            return W_Tensor(self.tensor.mul(self._other(space, w_other)))
        except ValueError:
            raise _mismatch(space)

    def descr_add_(self, space, w_other):
        try:
            self.tensor.add_(self._other(space, w_other))
        except ValueError:
            raise oefmt(space.w_ValueError, "in-place op not allowed")
        return self

    def descr_mul_(self, space, w_other):
        try:
            self.tensor.mul_(self._other(space, w_other))
        except ValueError:
            raise oefmt(space.w_ValueError, "in-place op not allowed")
        return self

    def descr_sub(self, space, w_other):
        try:
            return W_Tensor(self.tensor.sub(self._other(space, w_other)))
        except ValueError:
            raise _mismatch(space)

    def descr_div(self, space, w_other):
        try:
            return W_Tensor(self.tensor.div(self._other(space, w_other)))
        except ValueError:
            raise _mismatch(space)

    def descr_exp(self, space):
        return W_Tensor(self.tensor.exp())

    def descr_sqrt(self, space):
        return W_Tensor(self.tensor.sqrt())

    @unwrap_spec(axis=int)
    def descr_max(self, space, axis=-1):
        return W_Tensor(self.tensor.max(axis))

    def _rows_cols(self, space):
        t = self.tensor.t
        if ops.tensor_ndim(t) != 2:
            raise _mismatch(space)
        return ops.tensor_shape(t, 0), ops.tensor_shape(t, 1)

    def _rowwise(self, space):
        rows, cols = self._rows_cols(space)
        return cols

    def _weight(self, space, w_other, cols):
        other = self._other(space, w_other)
        if (ops.tensor_size(other.t) != cols or
                ops.tensor_dtype(other.t) != ops.tensor_dtype(self.tensor.t)):
            raise _mismatch(space)
        return other

    def descr_softmax(self, space):
        self._rowwise(space)
        return W_Tensor(nn.softmax(self.tensor))

    @unwrap_spec(eps=float)
    def descr_layer_norm(self, space, w_gamma, w_beta, eps=1e-5):
        cols = self._rowwise(space)
        gamma = self._weight(space, w_gamma, cols)
        beta = self._weight(space, w_beta, cols)
        return W_Tensor(nn.layernorm(self.tensor, gamma, beta, eps))

    @unwrap_spec(eps=float)
    def descr_rms_norm(self, space, w_gamma, eps=1e-5):
        cols = self._rowwise(space)
        gamma = self._weight(space, w_gamma, cols)
        return W_Tensor(nn.rmsnorm(self.tensor, gamma, eps))

    def descr_gelu(self, space):
        return W_Tensor(nn.gelu(self.tensor))

    def descr_silu(self, space):
        return W_Tensor(nn.silu(self.tensor))

    def descr_relu(self, space):
        return W_Tensor(self.tensor.relu())

    @unwrap_spec(axis=int)
    def descr_sum(self, space, axis=-1):
        return W_Tensor(self.tensor.sum(axis))

    def descr_item(self, space):
        return space.newfloat(self.tensor.item())

    @unwrap_spec(transpose_b=bool)
    def descr_matmul(self, space, w_other, transpose_b=False):
        try:
            return W_Tensor(self.tensor.matmul(self._other(space, w_other),
                                               transpose_b))
        except ValueError:
            raise _mismatch(space)

    def descr_reshape(self, space, w_shape):
        shape = _ints_w(space, w_shape)
        try:
            return W_Tensor(self.tensor.reshape(shape))
        except ValueError:
            raise _mismatch(space)

    @unwrap_spec(heads=int, dcols=int, off_a=int, off_b=int)
    def descr_attn_scores(self, space, w_other, heads, dcols=-1, off_a=0,
                          off_b=0):
        t = self.tensor.t
        other = self._other(space, w_other)
        if (heads <= 0 or ops.tensor_ndim(t) != 2 or
                ops.tensor_ndim(other.t) != 2):
            raise _mismatch(space)
        rows = ops.tensor_shape(t, 0)
        lda = ops.tensor_shape(t, 1)
        ldb = ops.tensor_shape(other.t, 1)
        d = lda if dcols <= 0 else dcols
        if (d % heads != 0 or off_a < 0 or off_b < 0 or
                off_a + d > lda or off_b + d > ldb or
                ops.tensor_shape(other.t, 0) != rows or
                ops.tensor_dtype(other.t) != ops.tensor_dtype(t)):
            raise _mismatch(space)
        return W_Tensor(self.tensor.attn_scores(
            other, heads, rows, d // heads, lda, ldb, off_a, off_b))

    @unwrap_spec(heads=int, dcols=int, off_b=int)
    def descr_attn_context(self, space, w_other, heads, dcols=-1, off_b=0):
        t = self.tensor.t
        other = self._other(space, w_other)
        if (heads <= 0 or ops.tensor_ndim(t) != 2 or
                ops.tensor_ndim(other.t) != 2):
            raise _mismatch(space)
        rows = ops.tensor_shape(t, 1)
        ldb = ops.tensor_shape(other.t, 1)
        d = ldb if dcols <= 0 else dcols
        if (ops.tensor_shape(t, 0) != heads * rows or d % heads != 0 or
                off_b < 0 or off_b + d > ldb or
                ops.tensor_shape(other.t, 0) != rows or
                ops.tensor_dtype(other.t) != ops.tensor_dtype(t)):
            raise _mismatch(space)
        return W_Tensor(self.tensor.attn_context(
            other, heads, rows, d // heads, ldb, off_b))

    @unwrap_spec(dh=int)
    def descr_rot_half(self, space, dh):
        if dh <= 1 or dh % 2 != 0:
            raise _mismatch(space)
        rows, cols = self._rows_cols(space)
        if cols % dh != 0:
            raise _mismatch(space)
        return W_Tensor(self.tensor.rot_half(dh))

    @unwrap_spec(heads=int)
    def descr_head_split(self, space, heads):
        if heads <= 0:
            raise _mismatch(space)
        rows, d = self._rows_cols(space)
        if d % heads != 0:
            raise _mismatch(space)
        return W_Tensor(self.tensor.head_split(rows, d // heads, heads))

    @unwrap_spec(heads=int)
    def descr_head_merge(self, space, heads):
        if heads <= 0:
            raise _mismatch(space)
        hr, cols = self._rows_cols(space)
        if hr % heads != 0:
            raise _mismatch(space)
        return W_Tensor(self.tensor.head_merge(hr // heads, cols, heads))

    @unwrap_spec(batch=int, transpose_b=bool)
    def descr_bmm(self, space, w_other, batch, transpose_b=False):
        a = self.tensor.t
        b = self._other(space, w_other).t
        if (batch <= 0 or ops.tensor_ndim(a) != 2 or
                ops.tensor_ndim(b) != 2 or
                ops.tensor_shape(a, 0) % batch != 0 or
                ops.tensor_shape(b, 0) % batch != 0):
            raise _mismatch(space)
        rows = ops.tensor_shape(a, 0) // batch
        inner = ops.tensor_shape(a, 1)
        if transpose_b:
            cols = ops.tensor_shape(b, 0) // batch
            ok = ops.tensor_shape(b, 1) == inner
        else:
            cols = ops.tensor_shape(b, 1)
            ok = ops.tensor_shape(b, 0) // batch == inner
        if not ok:
            raise _mismatch(space)
        return W_Tensor(self.tensor.bmm(
            self._other(space, w_other), batch, rows, cols, inner,
            1 if transpose_b else 0))

    def descr_tolist(self, space):
        t = self.tensor.t
        h = device.host(t)
        n = ops.tensor_size(t)
        return space.newlist([space.newfloat(h[i]) for i in range(n)])

    def descr_take(self, space, w_idx):
        t = self.tensor.t
        idx = self._other(space, w_idx).t
        trows, cols = self._rows_cols(space)
        if ops.tensor_dtype(idx) != ops.tensor_dtype(t):
            raise _mismatch(space)
        rows = ops.tensor_size(idx)
        return _wrap(runtime.rowgather(t, idx, rows, cols))

    @unwrap_spec(c=int, h=int, w=int, k=int, pad=int, stride=int)
    def descr_im2col(self, space, c, h, w, k=3, pad=1, stride=1):
        t = self.tensor.t
        self._conv_check(space, c, h, w, k, pad, stride)
        return _wrap(runtime.im2col(t, c, h, w, k, pad, stride))

    @unwrap_spec(c=int, h=int, w=int, k=int, pad=int, stride=int)
    def descr_im2col_nhwc(self, space, c, h, w, k=3, pad=1, stride=1):
        t = self.tensor.t
        self._conv_check(space, c, h, w, k, pad, stride)
        return _wrap(runtime.im2col_nhwc(t, c, h, w, k, pad, stride))

    def _conv_check(self, space, c, h, w, k, pad, stride):
        t = self.tensor.t
        if (c <= 0 or h <= 0 or w <= 0 or k <= 0 or pad < 0 or stride <= 0 or
                h + 2 * pad < k or w + 2 * pad < k or
                ops.tensor_size(t) % (c * h * w) != 0):
            raise _mismatch(space)

    @unwrap_spec(c=int, h=int, w=int, k=int, stride=int, pad=int)
    def descr_maxpool2(self, space, c, h, w, k=2, stride=2, pad=0):
        t = self.tensor.t
        self._conv_check(space, c, h, w, k, pad, stride)
        return _wrap(runtime.maxpool2(t, c, h, w, k, stride, pad))

    @unwrap_spec(c=int, h=int, w=int, k=int, stride=int, pad=int)
    def descr_maxpool2_nhwc(self, space, c, h, w, k=2, stride=2, pad=0):
        t = self.tensor.t
        self._conv_check(space, c, h, w, k, pad, stride)
        return _wrap(runtime.maxpool2_nhwc(t, c, h, w, k, stride,
                                           pad))

    @unwrap_spec(c=int, h=int, w=int, k=int, stride=int, pad=int)
    def descr_conv2d(self, space, w_weight, c, h, w, w_bias=None, k=3,
                     stride=1, pad=1):
        weight = self._other(space, w_weight).t
        t = self.tensor.t
        self._conv_check(space, c, h, w, k, pad, stride)
        if (ops.tensor_ndim(weight) != 2 or
                ops.tensor_shape(weight, 0) != c * k * k):
            raise _mismatch(space)
        o = ops.tensor_shape(weight, 1)
        ohw = ((h + 2 * pad - k) // stride + 1) * ((w + 2 * pad - k) //
                                                   stride + 1)
        rows = ops.tensor_size(t) // (c * h * w)
        y = runtime.tensor_matmul(runtime.im2col(t, c, h, w, k, pad, stride),
                                  weight, rows * ohw, o, c * k * k, 0, 0)
        if w_bias is not None and not space.is_none(w_bias):
            y = ops.tensor_add(y, self._other(space, w_bias).t,
                                   core.BC_R_ROW)
        return _wrap(runtime.col2chw(y, rows, ohw, o))

    def descr_detach(self, space):
        return W_Tensor(nn.Tensor(self.tensor.t, self.tensor.requires_grad))

    def descr_backward(self, space):
        self.tensor.backward()

    def descr_zero_grad(self, space):
        self.tensor.grad = None

    def descr_shape(self, space):
        t = self.tensor.t
        shape = [ops.tensor_shape(t, i) for i in range(ops.tensor_ndim(t))]
        return _shape_w(space, shape)

    def descr_size(self, space):
        return space.newint(ops.tensor_size(self.tensor.t))

    def descr_grad(self, space):
        g = self.tensor.grad
        if g is None:
            return space.w_None
        return W_Tensor(g)

    def descr_requires_grad(self, space):
        return space.newbool(self.tensor.requires_grad)

    def descr_dtype(self, space):
        dtype = ops.tensor_dtype(self.tensor.t)
        return space.newtext(core.DTYPE_NAMES[dtype])

    def descr_astype(self, space, w_dtype):
        dtype = _dtype_w(space, w_dtype)
        return W_Tensor(nn.Tensor(
            ops.astype(self.tensor.t, dtype), self.tensor.requires_grad))

    def descr_repr(self, space):
        t = self.tensor.t
        shape = [ops.tensor_shape(t, i) for i in range(ops.tensor_ndim(t))]
        parts = [str(d) for d in shape]
        return space.newtext("Tensor(shape=(%s))" % ", ".join(parts))


W_Tensor.typedef = TypeDef(
    'Tensor',
    add=interp2app(W_Tensor.descr_add),
    mul=interp2app(W_Tensor.descr_mul),
    add_=interp2app(W_Tensor.descr_add_),
    mul_=interp2app(W_Tensor.descr_mul_),
    sub=interp2app(W_Tensor.descr_sub),
    div=interp2app(W_Tensor.descr_div),
    exp=interp2app(W_Tensor.descr_exp),
    sqrt=interp2app(W_Tensor.descr_sqrt),
    max=interp2app(W_Tensor.descr_max),
    relu=interp2app(W_Tensor.descr_relu),
    softmax=interp2app(W_Tensor.descr_softmax),
    layer_norm=interp2app(W_Tensor.descr_layer_norm),
    rms_norm=interp2app(W_Tensor.descr_rms_norm),
    gelu=interp2app(W_Tensor.descr_gelu),
    silu=interp2app(W_Tensor.descr_silu),
    sum=interp2app(W_Tensor.descr_sum),
    item=interp2app(W_Tensor.descr_item),
    matmul=interp2app(W_Tensor.descr_matmul),
    reshape=interp2app(W_Tensor.descr_reshape),
    attn_scores=interp2app(W_Tensor.descr_attn_scores),
    attn_context=interp2app(W_Tensor.descr_attn_context),
    rot_half=interp2app(W_Tensor.descr_rot_half),
    head_split=interp2app(W_Tensor.descr_head_split),
    head_merge=interp2app(W_Tensor.descr_head_merge),
    bmm=interp2app(W_Tensor.descr_bmm),
    take=interp2app(W_Tensor.descr_take),
    tolist=interp2app(W_Tensor.descr_tolist),
    im2col=interp2app(W_Tensor.descr_im2col),
    im2col_nhwc=interp2app(W_Tensor.descr_im2col_nhwc),
    maxpool2=interp2app(W_Tensor.descr_maxpool2),
    maxpool2_nhwc=interp2app(W_Tensor.descr_maxpool2_nhwc),
    conv2d=interp2app(W_Tensor.descr_conv2d),
    detach=interp2app(W_Tensor.descr_detach),
    backward=interp2app(W_Tensor.descr_backward),
    zero_grad=interp2app(W_Tensor.descr_zero_grad),
    __add__=interp2app(W_Tensor.descr_add),
    __mul__=interp2app(W_Tensor.descr_mul),
    __sub__=interp2app(W_Tensor.descr_sub),
    __div__=interp2app(W_Tensor.descr_div),
    __truediv__=interp2app(W_Tensor.descr_div),
    __repr__=interp2app(W_Tensor.descr_repr),
    shape=GetSetProperty(W_Tensor.descr_shape),
    size=GetSetProperty(W_Tensor.descr_size),
    grad=GetSetProperty(W_Tensor.descr_grad),
    requires_grad=GetSetProperty(W_Tensor.descr_requires_grad),
    dtype=GetSetProperty(W_Tensor.descr_dtype),
    astype=interp2app(W_Tensor.descr_astype),
)


def _dtype_w(space, w_dtype):
    try:
        dtype = core.dtype_of_name(space.text_w(w_dtype))
    except ValueError:
        raise oefmt(space.w_ValueError, "unknown dtype")
    ensure_dtype(dtype)
    return dtype


class DeviceState(object):
    ready = False
device_state = DeviceState()

def ensure_device():
    if not device_state.ready:
        device_state.ready = True
        kernels.init_device()

def ensure_dtype(dtype):
    ensure_device()
    kernels.init_dtype(dtype)

@unwrap_spec(requires_grad=bool)
def tensor_flat(space, w_data, w_shape, requires_grad=False,
                w_dtype=None):
    ensure_device()
    dtype = core.F64
    if w_dtype is not None and not space.is_none(w_dtype):
        dtype = _dtype_w(space, w_dtype)
    values = _floats_w(space, w_data)
    shape = _ints_w(space, w_shape)
    n = 1
    for d in shape:
        n *= d
    if n != len(values):
        raise _mismatch(space)
    t = core.from_list(values, dtype)
    if len(shape) != 1 or shape[0] != n:
        try:
            t = ops.reshape(t, shape)
        except ValueError:
            raise _mismatch(space)
    return W_Tensor(nn.Tensor(t, requires_grad))


@unwrap_spec(requires_grad=bool)
def zeros(space, w_shape, requires_grad=False, w_dtype=None):
    ensure_device()
    dtype = core.F64
    if w_dtype is not None and not space.is_none(w_dtype):
        dtype = _dtype_w(space, w_dtype)
    shape = _ints_w(space, w_shape)
    t = core.zeros(shape, dtype)
    h = device.host(t)
    for i in range(t.size):
        h[i] = 0.0
    return W_Tensor(nn.Tensor(t, requires_grad))


def kernel_count(space):
    return space.newint(kernels.counter.n)


def launch_count(space):
    return space.newint(device.launch_count())
