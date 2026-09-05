import _tensor


class Param(object):
    def __init__(self, tensor):
        self.tensor = tensor


class Linear(object):
    def __init__(self, weight, bias):
        self.weight = Param(weight)
        self.bias = Param(bias)

    def __call__(self, x):
        return x.matmul(self.weight.tensor).add(self.bias.tensor).relu()


class MLP(object):
    def __init__(self, layers):
        self.layers = layers

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        params = []
        for layer in self.layers:
            params.append(layer.weight)
            params.append(layer.bias)
        return params


def sgd_step(params, lr):
    neg_lr = _tensor.tensor([-lr])
    for p in params:
        t = p.tensor
        g = t.grad
        if g is not None:
            p.tensor = t.add(g.mul(neg_lr)).detach()
