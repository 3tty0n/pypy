from . import take
from .functional import layer_norm, rms_norm, softmax, _scalar


class Param(object):
    def __init__(self, owner, name):
        self.owner = owner
        self.name = name

    @property
    def tensor(self):
        return getattr(self.owner, self.name)

    @tensor.setter
    def tensor(self, value):
        setattr(self.owner, self.name, value)

    @property
    def grad(self):
        return self.tensor.grad

    def zero_grad(self):
        self.tensor.zero_grad()


class Module(object):
    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def parameters(self):
        params = []
        for name, v in self.__dict__.items():
            if isinstance(v, Module):
                params.extend(v.parameters())
            elif isinstance(v, list):
                for item in v:
                    if isinstance(item, Module):
                        params.extend(item.parameters())
            elif getattr(v, "requires_grad", False):
                params.append(Param(self, name))
        return params


class Linear(Module):
    def __init__(self, weight, bias):
        self.weight = weight
        self.bias = bias

    def forward(self, x):
        return x.matmul(self.weight).add(self.bias).relu()


class LayerNorm(Module):
    def __init__(self, gamma, beta, eps=1e-5):
        self.gamma = gamma
        self.beta = beta
        self.eps = eps

    def forward(self, x):
        return layer_norm(x, self.gamma, self.beta, self.eps)


class RMSNorm(Module):
    def __init__(self, gamma, eps=1e-5):
        self.gamma = gamma
        self.eps = eps

    def forward(self, x):
        return rms_norm(x, self.gamma, self.eps)


class Embedding(Module):
    def __init__(self, weight):
        self.weight = weight

    def forward(self, idx):
        return take(self.weight, idx, axis=0)


class MultiheadAttention(Module):
    def __init__(self, wq, wk, wv, wo, heads):
        self.wq = wq
        self.wk = wk
        self.wv = wv
        self.wo = wo
        self.heads = heads

    def forward(self, x):
        import math
        h = self.heads
        dh = x.shape[1] // h
        q = x.matmul(self.wq).head_split(h)
        k = x.matmul(self.wk).head_split(h)
        v = x.matmul(self.wv).head_split(h)
        s = q.bmm(k, h, True).mul(_scalar(1.0 / math.sqrt(dh), x.dtype))
        c = softmax(s).bmm(v, h)
        return c.head_merge(h).matmul(self.wo)


class Conv2d(Module):
    def __init__(self, weight, bias, c, h, w, k=3, stride=1, pad=1,
                 nhwc=False):
        self.nhwc = nhwc
        self.weight = weight
        self.bias = bias
        self.c = c
        self.h = h
        self.w = w
        self.k = k
        self.stride = stride
        self.pad = pad
        self.oh = (h + 2 * pad - k) // stride + 1
        self.ow = (w + 2 * pad - k) // stride + 1

    def forward(self, x):
        if not self.nhwc:
            return x.conv2d(self.weight, self.c, self.h, self.w, self.bias,
                            self.k, self.stride, self.pad)
        if self.k == 1 and self.stride == 1 and self.pad == 0:
            cols = x
        else:
            cols = x.im2col_nhwc(self.c, self.h, self.w, self.k, self.pad,
                                 self.stride)
        # conv GEMM: tensor cores in TF32, matching cuDNN's default for a
        # convolution.  Plain matmuls elsewhere stay in strict fp32.
        y = cols.matmul(self.weight, False, True)
        if self.bias is not None:
            y = y.add(self.bias)
        return y


class BatchNorm2d(Module):
    def __init__(self, c, gamma=None, beta=None, mean=None, var=None,
                 eps=1e-5, dtype=None, nhwc=False):
        self.nhwc = nhwc
        self.c = c
        self.gamma = gamma if gamma is not None else [1.0] * c
        self.beta = beta if beta is not None else [0.0] * c
        self.mean = mean if mean is not None else [0.0] * c
        self.var = var if var is not None else [1.0] * c
        self.eps = eps
        self.dtype = dtype
        self.cache = {}

    def _params(self, rows, shape=None):
        import _metatensor
        if shape is None:
            shape = [rows, 1]
        if rows not in self.cache:
            import math
            a = []
            b = []
            for i in range(rows):
                ci = i % self.c
                g = self.gamma[ci] / math.sqrt(self.var[ci] + self.eps)
                a.append(g)
                b.append(self.beta[ci] - self.mean[ci] * g)
            self.cache[rows] = (
                _metatensor.tensor(a, shape, False, self.dtype),
                _metatensor.tensor(b, shape, False, self.dtype))
        return self.cache[rows]

    def forward(self, x, hw):
        if self.nhwc:
            scale, shift = self._params(self.c, [1, self.c])
            return x.mul(scale).add(shift)
        rows = x.size // hw
        scale, shift = self._params(rows)
        y = x.reshape([rows, hw]).mul(scale).add(shift)
        return y.reshape([rows // self.c, self.c * hw])


class MaxPool2d(Module):
    def __init__(self, c, h, w, k=2, stride=2, pad=0, nhwc=False):
        self.nhwc = nhwc
        self.c = c
        self.h = h
        self.w = w
        self.k = k
        self.stride = stride
        self.pad = pad
        self.oh = (h + 2 * pad - k) // stride + 1
        self.ow = (w + 2 * pad - k) // stride + 1

    def forward(self, x):
        if self.nhwc:
            return x.maxpool2_nhwc(self.c, self.h, self.w, self.k,
                                   self.stride, self.pad)
        return x.maxpool2(self.c, self.h, self.w, self.k, self.stride,
                          self.pad)
