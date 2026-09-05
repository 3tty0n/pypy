from rpython.rlib import rtensor, jit


class Tensor(object):
    def __init__(self, t):
        self.t = t

    def add(self, other):
        return Tensor(rtensor.add(self.t, other.t))

    def mul(self, other):
        return Tensor(rtensor.mul(self.t, other.t))

    def relu(self):
        return Tensor(rtensor.relu(self.t))

    def sum(self, axis=rtensor.AXIS_ALL):
        return Tensor(rtensor.sum(self.t, axis))

    def item(self):
        return rtensor.item(self.t)

    def matmul(self, other):
        return Tensor(rtensor.matmul(self.t, other.t))

    def reshape(self, shape_list):
        return Tensor(rtensor.reshape(self.t, shape_list))

    def add_(self, other):
        return Tensor(rtensor.add_(self.t, other.t))

    def mul_(self, other):
        return Tensor(rtensor.mul_(self.t, other.t))

    def shape(self, axis):
        return rtensor.tensor_shape(self.t, axis)

    def size(self):
        return rtensor.tensor_size(self.t)


class Linear(object):
    def __init__(self, weight, bias):
        self.weight = weight
        self.bias = bias

    def forward(self, x):
        return x.matmul(self.weight).add(self.bias).relu()


class MLP(object):
    def __init__(self, layers):
        self.layers = layers

    @jit.unroll_safe
    def forward(self, x):
        for i in range(len(self.layers)):
            x = self.layers[i].forward(x)
        return x
