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
