import torch
from torch import Tensor


def relu(x, inplace=False):
    return x.relu()


def gelu(x, approximate="none"):
    if approximate == "tanh":
        return Tensor(x.t.gelu(), x.shape)
    return Tensor(x.t.gelu_erf(), x.shape)


def leaky_relu(x, negative_slope=0.01, inplace=False):
    raise NotImplementedError("leaky_relu")


def linear(x, weight, bias=None):
    """x @ weight.T + bias, weight in torch's [out, in]."""
    n = x.shape[-1]
    out = weight.shape[0]
    rows = x.numel() // n
    y = x.t.reshape([rows, n]).matmul(weight.t, True,
                                      torch.backends.cuda.matmul.allow_tf32)
    if bias is not None:
        y = y.add(bias.t)
    return Tensor(y, x.shape[:-1] + (out,))


def adaptive_avg_pool2d(x, output_size):
    n, c, h, w = x.shape
    oh, ow = output_size
    if (oh, ow) == (h, w):
        return x
    if (oh, ow) == (1, 1):
        y = x.t.reshape([n * c, h * w]).sum(1)
        return Tensor(y, (n, c, 1, 1)) * (1.0 / (h * w))
    raise NotImplementedError("adaptive average pool %s -> %s"
                              % ((h, w), (oh, ow)))


def softmax(x, dim=-1, dtype=None):
    return x.softmax(dim)


def dropout(x, p=0.5, training=False, inplace=False):
    return x


def _transpose2(x):
    """A 2-D transpose, done once on the host: only parameters at load time
    come through here."""
    r, c = x.shape
    v = x.tolist()
    flat = [v[i * c + j] for j in range(c) for i in range(r)]
    return torch.from_flat(flat, (c, r), x.dtype)
