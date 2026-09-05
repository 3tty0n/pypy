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


_scalars = {}


def _scalar(v, dtype="float64"):
    key = (v, dtype)
    t = _scalars.get(key)
    if t is None:
        t = _tensor.tensor([v], None, False, dtype)
        _scalars[key] = t
    return t


def softmax(x):
    rows = x.shape[0]
    m = x.max(1).reshape([rows, 1])
    e = x.sub(m).exp()
    s = e.sum(1).reshape([rows, 1])
    return e.div(s)


def layernorm(x, gamma, beta, eps=1e-5):
    rows, cols = x.shape
    dt = x.dtype
    inv = _scalar(1.0 / cols, dt)
    mean = x.sum(1).mul(inv).reshape([rows, 1])
    d = x.sub(mean)
    var = d.mul(d).sum(1).mul(inv).reshape([rows, 1])
    denom = var.add(_scalar(eps, dt)).sqrt()
    return d.div(denom).mul(gamma).add(beta)


class MultiHead(object):
    def __init__(self, wq, wk, wv, wo, heads):
        self.wq = wq
        self.wk = wk
        self.wv = wv
        self.wo = wo
        self.heads = heads

    def __call__(self, x):
        import math
        h = self.heads
        dh = x.shape[1] // h
        q = x.matmul(self.wq).head_split(h)
        k = x.matmul(self.wk).head_split(h)
        v = x.matmul(self.wv).head_split(h)
        s = q.bmm(k, h, True).mul(_scalar(1.0 / math.sqrt(dh), x.dtype))
        c = softmax(s).bmm(v, h)
        return c.head_merge(h).matmul(self.wo)


class TransformerBlock(object):
    def __init__(self, attn, gamma1, beta1, gamma2, beta2, mlp, eps=1e-5):
        self.attn = attn
        self.gamma1 = gamma1
        self.beta1 = beta1
        self.gamma2 = gamma2
        self.beta2 = beta2
        self.mlp = mlp
        self.eps = eps

    def __call__(self, x):
        h = layernorm(x, self.gamma1, self.beta1, self.eps)
        x = x.add(self.attn(h))
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


def gelu(x):
    import math
    dt = x.dtype
    c = _scalar(math.sqrt(2.0 / math.pi), dt)
    x3 = x.mul(x).mul(x).mul(_scalar(0.044715, dt))
    z = x.add(x3).mul(c).mul(_scalar(2.0, dt))
    one = _scalar(1.0, dt)
    tanh = one.sub(_scalar(2.0, dt).div(z.exp().add(one)))
    return x.mul(_scalar(0.5, dt)).mul(tanh.add(one))


class CausalSelfAttention(object):
    def __init__(self, wq, bq, wk, bk, wv, bv, wo, bo, heads, mask):
        self.wq = wq
        self.bq = bq
        self.wk = wk
        self.bk = bk
        self.wv = wv
        self.bv = bv
        self.wo = wo
        self.bo = bo
        self.heads = heads
        self.mask = mask

    def __call__(self, x):
        import math
        h = self.heads
        dh = x.shape[1] // h
        q = x.matmul(self.wq).add(self.bq).head_split(h)
        k = x.matmul(self.wk).add(self.bk).head_split(h)
        v = x.matmul(self.wv).add(self.bv).head_split(h)
        s = q.bmm(k, h, True).mul(
            _scalar(1.0 / math.sqrt(dh), x.dtype)).add(self.mask)
        c = softmax(s).bmm(v, h)
        return c.head_merge(h).matmul(self.wo).add(self.bo)


class GPT2MLP(object):
    def __init__(self, wfc, bfc, wproj, bproj):
        self.wfc = wfc
        self.bfc = bfc
        self.wproj = wproj
        self.bproj = bproj

    def __call__(self, x):
        h = gelu(x.matmul(self.wfc).add(self.bfc))
        return h.matmul(self.wproj).add(self.bproj)


class GPT2Block(object):
    def __init__(self, attn, g1, b1, g2, b2, mlp, eps=1e-5):
        self.attn = attn
        self.g1 = g1
        self.b1 = b1
        self.g2 = g2
        self.b2 = b2
        self.mlp = mlp
        self.eps = eps

    def __call__(self, x):
        x = x.add(self.attn(layernorm(x, self.g1, self.b1, self.eps)))
        return x.add(self.mlp(layernorm(x, self.g2, self.b2, self.eps)))


class GPT2(object):
    def __init__(self, wte, blocks, gf, bf, eps=1e-5):
        self.wte = wte
        self.blocks = blocks
        self.gf = gf
        self.bf = bf
        self.eps = eps

    def __call__(self, idx, pos):
        x = self.wte.take(idx).add(pos)
        for block in self.blocks:
            x = block(x)
        x = layernorm(x, self.gf, self.bf, self.eps)
        return x.matmul(self.wte, True)


def rmsnorm(x, gamma, eps=1e-5):
    rows, cols = x.shape
    dt = x.dtype
    inv = _scalar(1.0 / cols, dt)
    denom = x.mul(x).sum(1).mul(inv).add(_scalar(eps, dt)).sqrt()
    return x.div(denom.reshape([rows, 1])).mul(gamma)


def silu(x):
    dt = x.dtype
    one = _scalar(1.0, dt)
    return x.div(one.add(x.mul(_scalar(-1.0, dt)).exp()))


def rope(x, cos, sin, p):
    return x.mul(cos).add(x.matmul(p).mul(sin))


class LlamaAttention(object):
    def __init__(self, wq, wk, wv, wo, heads, mask, cos, sin, p):
        self.wq = wq
        self.wk = wk
        self.wv = wv
        self.wo = wo
        self.heads = heads
        self.mask = mask
        self.cos = cos
        self.sin = sin
        self.p = p

    def __call__(self, x):
        import math
        h = self.heads
        dh = x.shape[1] // h
        q = rope(x.matmul(self.wq), self.cos, self.sin, self.p).head_split(h)
        k = rope(x.matmul(self.wk), self.cos, self.sin, self.p).head_split(h)
        v = x.matmul(self.wv).head_split(h)
        s = q.bmm(k, h, True).mul(
            _scalar(1.0 / math.sqrt(dh), x.dtype)).add(self.mask)
        c = softmax(s).bmm(v, h)
        return c.head_merge(h).matmul(self.wo)


class LlamaMLP(object):
    def __init__(self, wgate, wup, wdown):
        self.wgate = wgate
        self.wup = wup
        self.wdown = wdown

    def __call__(self, x):
        return silu(x.matmul(self.wgate)).mul(x.matmul(self.wup)).matmul(
            self.wdown)


class LlamaBlock(object):
    def __init__(self, attn, g1, g2, mlp, eps=1e-5):
        self.attn = attn
        self.g1 = g1
        self.g2 = g2
        self.mlp = mlp
        self.eps = eps

    def __call__(self, x):
        x = x.add(self.attn(rmsnorm(x, self.g1, self.eps)))
        return x.add(self.mlp(rmsnorm(x, self.g2, self.eps)))


class Llama(object):
    def __init__(self, wte, blocks, gf, head=None, eps=1e-5):
        self.wte = wte
        self.blocks = blocks
        self.gf = gf
        self.head = head
        self.eps = eps

    def __call__(self, idx):
        x = self.wte.take(idx)
        for block in self.blocks:
            x = block(x)
        x = rmsnorm(x, self.gf, self.eps)
        if self.head is None:
            return x.matmul(self.wte, True)
        return x.matmul(self.head, True)
