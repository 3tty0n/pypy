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

    def descr_relu(self, space):
        return W_Tensor(self.tensor.relu())

    @unwrap_spec(axis=int)
    def descr_sum(self, space, axis=-1):
        return W_Tensor(self.tensor.sum(axis))

    def descr_item(self, space):
        return space.newfloat(self.tensor.item())

    def descr_matmul(self, space, w_other):
        try:
            return W_Tensor(self.tensor.matmul(self._other(space, w_other)))
        except ValueError:
            raise oefmt(space.w_ValueError, "shape mismatch")

    def descr_reshape(self, space, w_shape):
        shape = _ints_w(space, w_shape)
        try:
            return W_Tensor(self.tensor.reshape(shape))
        except ValueError:
            raise oefmt(space.w_ValueError, "shape mismatch")

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
    relu=interp2app(W_Tensor.descr_relu),
    sum=interp2app(W_Tensor.descr_sum),
    item=interp2app(W_Tensor.descr_item),
    matmul=interp2app(W_Tensor.descr_matmul),
    reshape=interp2app(W_Tensor.descr_reshape),
    detach=interp2app(W_Tensor.descr_detach),
    backward=interp2app(W_Tensor.descr_backward),
    zero_grad=interp2app(W_Tensor.descr_zero_grad),
    __add__=interp2app(W_Tensor.descr_add),
    __mul__=interp2app(W_Tensor.descr_mul),
    __repr__=interp2app(W_Tensor.descr_repr),
    shape=GetSetProperty(W_Tensor.descr_shape),
    size=GetSetProperty(W_Tensor.descr_size),
    grad=GetSetProperty(W_Tensor.descr_grad),
    requires_grad=GetSetProperty(W_Tensor.descr_requires_grad),
)


@unwrap_spec(requires_grad=bool)
def tensor_flat(space, w_data, w_shape, requires_grad=False):
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
    shape = _ints_w(space, w_shape)
    t = rtensor.zeros(shape)
    h = rtensor.host(t)
    for i in range(t.size):
        h[i] = 0.0
    return W_Tensor(rtensor_nn.Tensor(t, requires_grad))
