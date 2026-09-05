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


def softmax(x):
    rows = x.shape[0]
    m = x.max(1).reshape([rows, 1])
    e = x.sub(m).exp()
    s = e.sum(1).reshape([rows, 1])
    return e.div(s)


def layernorm(x, gamma, beta, eps=1e-5):
    rows, cols = x.shape
    inv = _tensor.tensor([1.0 / cols])
    mean = x.sum(1).mul(inv).reshape([rows, 1])
    d = x.sub(mean)
    var = d.mul(d).sum(1).mul(inv).reshape([rows, 1])
    denom = var.add(_tensor.tensor([eps])).sqrt()
    return d.div(denom).mul(gamma).add(beta)


def attention(q, k, v):
    import math
    scale = _tensor.tensor([1.0 / math.sqrt(q.shape[1])])
    scores = q.matmul(k, True).mul(scale)
    return softmax(scores).matmul(v)


class Head(object):
    def __init__(self, wq, wk, wv, wo):
        self.wq = wq
        self.wk = wk
        self.wv = wv
        self.wo = wo

    def __call__(self, x):
        return attention(x.matmul(self.wq), x.matmul(self.wk),
                         x.matmul(self.wv)).matmul(self.wo)


class TransformerBlock(object):
    def __init__(self, heads, gamma1, beta1, gamma2, beta2, mlp, eps=1e-5):
        self.heads = heads
        self.gamma1 = gamma1
        self.beta1 = beta1
        self.gamma2 = gamma2
        self.beta2 = beta2
        self.mlp = mlp
        self.eps = eps

    def __call__(self, x):
        h = layernorm(x, self.gamma1, self.beta1, self.eps)
        a = self.heads[0](h)
        for head in self.heads[1:]:
            a = a.add(head(h))
        x = x.add(a)
        return x.add(self.mlp(layernorm(x, self.gamma2, self.beta2, self.eps)))


class Conv2d(object):
    def __init__(self, weight, bias, c, h, w):
        self.weight = Param(weight)
        self.bias = Param(bias)
        self.c = c
        self.h = h
        self.w = w

    def __call__(self, x):
        return x.conv2d(self.weight.tensor, self.c, self.h, self.w,
                        self.bias.tensor)


class BatchNorm2d(object):
    def __init__(self, c, gamma=None, beta=None, mean=None, var=None,
                 eps=1e-5):
        self.c = c
        self.gamma = gamma if gamma is not None else [1.0] * c
        self.beta = beta if beta is not None else [0.0] * c
        self.mean = mean if mean is not None else [0.0] * c
        self.var = var if var is not None else [1.0] * c
        self.eps = eps
        self.cache = {}

    def _params(self, rows):
        if rows not in self.cache:
            import math
            a = []
            b = []
            for i in range(rows):
                ci = i % self.c
                g = self.gamma[ci] / math.sqrt(self.var[ci] + self.eps)
                a.append(g)
                b.append(self.beta[ci] - self.mean[ci] * g)
            self.cache[rows] = (_tensor.tensor(a).reshape([rows, 1]),
                                _tensor.tensor(b).reshape([rows, 1]))
        return self.cache[rows]

    def __call__(self, x, hw):
        rows = x.size // hw
        scale, shift = self._params(rows)
        y = x.reshape([rows, hw]).mul(scale).add(shift)
        return y.reshape([rows // self.c, self.c * hw])


class MaxPool2d(object):
    def __init__(self, c, h, w):
        self.c = c
        self.h = h
        self.w = w

    def __call__(self, x):
        return x.maxpool2(self.c, self.h, self.w)


class CNN(object):
    def __init__(self, conv, bn, pool, fc):
        self.conv = conv
        self.bn = bn
        self.pool = pool
        self.fc = fc

    def __call__(self, x):
        y = self.bn(self.conv(x), self.pool.h * self.pool.w)
        return self.fc(self.pool(y.relu()))
