from rpython.rlib import rtensor, rtensor_nn
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
            raise oefmt(space.w_ValueError, "shape mismatch")

    def descr_mul(self, space, w_other):
        try:
            return W_Tensor(self.tensor.mul(self._other(space, w_other)))
        except ValueError:
            raise oefmt(space.w_ValueError, "shape mismatch")

    def descr_add_(self, space, w_other):
        return self.descr_add(space, w_other)

    def descr_mul_(self, space, w_other):
        return self.descr_mul(space, w_other)

    def descr_sub(self, space, w_other):
        try:
            return W_Tensor(self.tensor.sub(self._other(space, w_other)))
        except ValueError:
            raise oefmt(space.w_ValueError, "shape mismatch")

    def descr_div(self, space, w_other):
        try:
            return W_Tensor(self.tensor.div(self._other(space, w_other)))
        except ValueError:
            raise oefmt(space.w_ValueError, "shape mismatch")

    def descr_exp(self, space):
        return W_Tensor(self.tensor.exp())

    def descr_sqrt(self, space):
        return W_Tensor(self.tensor.sqrt())

    @unwrap_spec(axis=int)
    def descr_max(self, space, axis=-1):
        return W_Tensor(self.tensor.max(axis))

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
            raise oefmt(space.w_ValueError, "shape mismatch")

    def descr_reshape(self, space, w_shape):
        shape = _ints_w(space, w_shape)
        try:
            return W_Tensor(self.tensor.reshape(shape))
        except ValueError:
            raise oefmt(space.w_ValueError, "shape mismatch")

    @unwrap_spec(c=int, h=int, w=int, k=int, pad=int)
    def descr_im2col(self, space, c, h, w, k=3, pad=1):
        t = self.tensor.t
        if c <= 0 or h <= 0 or w <= 0 or k <= 0 or pad < 0:
            raise oefmt(space.w_ValueError, "shape mismatch")
        if rtensor.tensor_size(t) % (c * h * w) != 0:
            raise oefmt(space.w_ValueError, "shape mismatch")
        return W_Tensor(rtensor_nn.Tensor(rtensor.im2col(t, c, h, w, k, pad)))

    @unwrap_spec(c=int, h=int, w=int)
    def descr_maxpool2(self, space, c, h, w):
        t = self.tensor.t
        if c <= 0 or h <= 1 or w <= 1:
            raise oefmt(space.w_ValueError, "shape mismatch")
        if rtensor.tensor_size(t) % (c * h * w) != 0:
            raise oefmt(space.w_ValueError, "shape mismatch")
        return W_Tensor(rtensor_nn.Tensor(rtensor.maxpool2(t, c, h, w)))

    @unwrap_spec(c=int, h=int, w=int)
    def descr_conv2d(self, space, w_weight, c, h, w, w_bias=None):
        weight = self._other(space, w_weight).t
        t = self.tensor.t
        hw = h * w
        if c <= 0 or h <= 0 or w <= 0 or rtensor.tensor_size(t) % (c * hw) != 0:
            raise oefmt(space.w_ValueError, "shape mismatch")
        if (rtensor.tensor_ndim(weight) != 2 or
                rtensor.tensor_shape(weight, 0) != c * 9):
            raise oefmt(space.w_ValueError, "shape mismatch")
        o = rtensor.tensor_shape(weight, 1)
        rows = rtensor.tensor_size(t) // (c * hw)
        y = rtensor.tensor_matmul(rtensor.im2col(t, c, h, w, 3, 1), weight,
                                  rows * hw, o, c * 9, 0, 0)
        if w_bias is not None and not space.is_none(w_bias):
            y = rtensor.tensor_add(y, self._other(space, w_bias).t,
                                   rtensor.BC_R_ROW)
        return W_Tensor(rtensor_nn.Tensor(rtensor.col2chw(y, rows, hw, o)))

    def descr_detach(self, space):
        return W_Tensor(rtensor_nn.Tensor(self.tensor.t, self.tensor.requires_grad))

    def descr_backward(self, space):
        self.tensor.backward()

    def descr_zero_grad(self, space):
        self.tensor.grad = None

    def descr_shape(self, space):
        t = self.tensor.t
        shape = [rtensor.tensor_shape(t, i) for i in range(rtensor.tensor_ndim(t))]
        return _shape_w(space, shape)

    def descr_size(self, space):
        return space.newint(rtensor.tensor_size(self.tensor.t))

    def descr_grad(self, space):
        g = self.tensor.grad
        if g is None:
            return space.w_None
        return W_Tensor(g)

    def descr_requires_grad(self, space):
        return space.newbool(self.tensor.requires_grad)

    def descr_repr(self, space):
        t = self.tensor.t
        shape = [rtensor.tensor_shape(t, i) for i in range(rtensor.tensor_ndim(t))]
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
    sum=interp2app(W_Tensor.descr_sum),
    item=interp2app(W_Tensor.descr_item),
    matmul=interp2app(W_Tensor.descr_matmul),
    reshape=interp2app(W_Tensor.descr_reshape),
    im2col=interp2app(W_Tensor.descr_im2col),
    maxpool2=interp2app(W_Tensor.descr_maxpool2),
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
)


class DeviceState(object):
    ready = False
device_state = DeviceState()

def ensure_device():
    if not device_state.ready:
        device_state.ready = True
        rtensor.init_device()

@unwrap_spec(requires_grad=bool)
def tensor_flat(space, w_data, w_shape, requires_grad=False):
    ensure_device()
    values = _floats_w(space, w_data)
    shape = _ints_w(space, w_shape)
    n = 1
    for d in shape:
        n *= d
    if n != len(values):
        raise oefmt(space.w_ValueError, "shape mismatch")
    t = rtensor.from_list(values)
    if len(shape) != 1 or shape[0] != n:
        try:
            t = rtensor.reshape(t, shape)
        except ValueError:
            raise oefmt(space.w_ValueError, "shape mismatch")
    return W_Tensor(rtensor_nn.Tensor(t, requires_grad))


@unwrap_spec(requires_grad=bool)
def zeros(space, w_shape, requires_grad=False):
    ensure_device()
    shape = _ints_w(space, w_shape)
    t = rtensor.zeros(shape)
    h = rtensor.host(t)
    for i in range(t.size):
        h[i] = 0.0
    return W_Tensor(rtensor_nn.Tensor(t, requires_grad))
