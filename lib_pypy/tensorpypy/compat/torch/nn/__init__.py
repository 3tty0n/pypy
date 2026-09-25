"""torch.nn modules, eval mode only.

A module is built from its constructor arguments alone, as upstream builds
it; its tensors arrive later through load_state_dict under their torch
names.  Whatever layout the engine wants a parameter in (a convolution's
filter as the [C*k*k, O] matrix im2col multiplies, batch norm's statistics
as one scale and one shift per channel) is made from the torch tensor at
load time, the way cuDNN transforms a filter, and never by the port.
"""
import math

import torch
from torch import Tensor
from torch.nn import functional as F
from torch.nn import init  # noqa: F401


class Parameter(object):
    """Placeholder until load_state_dict; upstream code only passes it to
    torch.nn.init, which does nothing here."""

    def __init__(self, *shape):
        self.shape = shape


class Module(object):
    training = False

    def __init__(self, *args, **kwargs):
        pass

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def eval(self):
        return self

    def train(self, mode=True):
        if mode:
            raise NotImplementedError("training")
        return self

    def to(self, *args, **kwargs):
        return self

    def named_children(self):
        for k, v in sorted(self.__dict__.items()):
            if isinstance(v, Module):
                yield k, v

    def children(self):
        for _, m in self.named_children():
            yield m

    def modules(self):
        yield self
        for m in self.children():
            for x in m.modules():
                yield x

    def _child(self, name):
        return getattr(self, name)

    def register_buffer(self, name, value, persistent=True):
        setattr(self, name, value)

    def load_state_dict(self, sd, strict=True):
        """sd maps torch names to Tensors.  Each module takes the entries
        named for it, then prepares whatever its forward reads."""
        by_module = {}
        for key, value in sd.items():
            path, _, leaf = key.rpartition(".")
            by_module.setdefault(path, {})[leaf] = value
        for path, entries in by_module.items():
            m = self
            if path:
                for part in path.split("."):
                    m = m._child(part)
            m._load(entries)
        for m in self.modules():
            m._prepare()
        return self

    def _load(self, entries):
        for k, v in entries.items():
            setattr(self, k, v)

    def _prepare(self):
        pass


class Sequential(Module):
    def __init__(self, *mods):
        if len(mods) == 1 and isinstance(mods[0], dict):
            mods = list(mods[0].values())
        self._mods = list(mods)

    def _child(self, name):
        return self._mods[int(name)]

    def named_children(self):
        for i, m in enumerate(self._mods):
            yield str(i), m

    def __getitem__(self, i):
        return self._mods[i]

    def __len__(self):
        return len(self._mods)

    def __iter__(self):
        return iter(self._mods)

    def append(self, m):
        self._mods.append(m)

    def forward(self, x):
        for m in self._mods:
            x = m(x)
        return x


class ModuleList(Sequential):
    def forward(self, *a):
        raise NotImplementedError("ModuleList is not callable")


class Identity(Module):
    def forward(self, x):
        return x


class Dropout(Module):
    def __init__(self, p=0.5, inplace=False):
        self.p = p

    def forward(self, x):
        return x


class ReLU(Module):
    def __init__(self, inplace=False):
        pass

    def forward(self, x):
        return x.relu()


class GELU(Module):
    def __init__(self, approximate="none"):
        self.approximate = approximate

    def forward(self, x):
        return F.gelu(x, self.approximate)


class Tanh(Module):
    def forward(self, x):
        return x.tanh()


class Sigmoid(Module):
    def forward(self, x):
        return x.sigmoid()


class LeakyReLU(Module):
    def __init__(self, negative_slope=0.01, inplace=False):
        self.negative_slope = negative_slope

    def forward(self, x):
        return F.leaky_relu(x, self.negative_slope)


class Linear(Module):
    def __init__(self, in_features, out_features, bias=True, device=None,
                 dtype=None):
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(out_features, in_features)
        self.bias = Parameter(out_features) if bias else None

    def forward(self, x):
        return F.linear(x, self.weight, self.bias)


def _pair(v):
    if isinstance(v, (tuple, list)):
        return tuple(v)
    return (v, v)


def _square(name, v):
    a, b = _pair(v)
    if a != b:
        raise NotImplementedError("non-square %s %s" % (name, (a, b)))
    return a


class Conv2d(Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True,
                 padding_mode="zeros", device=None, dtype=None):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)
        self.padding = _pair(padding)
        self.dilation = _pair(dilation)
        self.groups = groups
        self.weight = Parameter(out_channels, in_channels // groups,
                                *self.kernel_size)
        self.bias = Parameter(out_channels) if bias else None

    def _prepare(self):
        if self.groups != 1:
            raise NotImplementedError("grouped convolution")
        if self.dilation != (1, 1):
            raise NotImplementedError("dilated convolution")
        self.k = _square("kernel", self.kernel_size)
        self.s = _square("stride", self.stride)
        self.p = _square("padding", self.padding)
        w = self.weight
        o = w.shape[0]
        # [O, C, k, k] -> [C*k*k, O], the matrix im2col's columns multiply
        self.filter = F._transpose2(w.view(o, w.numel() // o))

    def forward(self, x):
        n, c, h, w = x.shape
        b = self.bias.t if isinstance(self.bias, Tensor) else None
        y = x.t.conv2d(self.filter.t, c, h, w, b, self.k, self.s, self.p)
        oh = (h + 2 * self.p - self.k) // self.s + 1
        ow = (w + 2 * self.p - self.k) // self.s + 1
        return Tensor(y, (n, self.out_channels, oh, ow))


class BatchNorm2d(Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True,
                 track_running_stats=True, device=None, dtype=None):
        self.num_features = num_features
        self.eps = eps
        self.weight = Parameter(num_features)
        self.bias = Parameter(num_features)
        self._rows = {}

    def _prepare(self):
        g = self.weight.tolist()
        b = self.bias.tolist()
        m = self.running_mean.tolist()
        v = self.running_var.tolist()
        self._scale = [g[i] / math.sqrt(v[i] + self.eps)
                       for i in range(self.num_features)]
        self._shift = [b[i] - m[i] * self._scale[i]
                       for i in range(self.num_features)]
        self._dtype = self.weight.dtype
        self._rows = {}

    def _per_row(self, n):
        # eval batch norm is x * scale + shift per channel; over the
        # [N*C, H*W] view that is a column broadcast of the per-channel
        # vectors repeated once per image
        r = self._rows.get(n)
        if r is None:
            r = (torch.from_flat(self._scale * n, [n * self.num_features, 1],
                                 self._dtype),
                 torch.from_flat(self._shift * n, [n * self.num_features, 1],
                                 self._dtype))
            self._rows[n] = r
        return r

    def forward(self, x):
        n, c, h, w = x.shape
        scale, shift = self._per_row(n)
        y = x.t.reshape([n * c, h * w]).mul(scale.t).add(shift.t)
        return Tensor(y, x.shape)


class MaxPool2d(Module):
    def __init__(self, kernel_size, stride=None, padding=0, dilation=1,
                 return_indices=False, ceil_mode=False):
        self.kernel_size = kernel_size
        self.stride = stride if stride is not None else kernel_size
        self.padding = padding
        if dilation != 1 or ceil_mode or return_indices:
            raise NotImplementedError("max pool dilation/ceil/indices")

    def forward(self, x):
        n, c, h, w = x.shape
        k = _square("kernel", self.kernel_size)
        s = _square("stride", self.stride)
        p = _square("padding", self.padding)
        y = x.t.maxpool2(c, h, w, k, s, p)
        return Tensor(y, (n, c, (h + 2 * p - k) // s + 1,
                          (w + 2 * p - k) // s + 1))


class AdaptiveAvgPool2d(Module):
    def __init__(self, output_size):
        self.output_size = _pair(output_size)

    def forward(self, x):
        return F.adaptive_avg_pool2d(x, self.output_size)


class Flatten(Module):
    def __init__(self, start_dim=1, end_dim=-1):
        self.start_dim = start_dim
        self.end_dim = end_dim

    def forward(self, x):
        return x.flatten(self.start_dim, self.end_dim)




class GroupNorm(Module):
    def __init__(self, num_groups, num_channels, eps=1e-5, affine=True):
        self.weight = Parameter(num_channels)
        self.bias = Parameter(num_channels)

    def forward(self, x):
        raise NotImplementedError("GroupNorm")
