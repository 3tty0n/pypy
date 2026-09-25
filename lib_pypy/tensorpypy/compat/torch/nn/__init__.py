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

    @property
    def data(self):
        return self

    def size(self, d=None):
        return self.shape if d is None else self.shape[d]

    def numel(self):
        return torch._numel(self.shape)

    def __getitem__(self, i):
        if not isinstance(i, int):
            raise NotImplementedError("Parameter placeholder indexed %r"
                                      % (i,))
        return Parameter(*self.shape[1:])

    def _init_op(self, *args, **kwargs):
        return self

    mul_ = div_ = add_ = zero_ = fill_ = copy_ = normal_ = uniform_ = \
        _init_op


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

    def named_modules(self, memo=None, prefix='', remove_duplicate=True):
        if memo is None:
            memo = set()
        if remove_duplicate and id(self) in memo:
            return
        memo.add(id(self))
        yield prefix, self
        for name, m in self.named_children():
            sub = prefix + ('.' if prefix else '') + name
            for x in m.named_modules(memo, sub, remove_duplicate):
                yield x

    def apply(self, fn):
        for m in self.children():
            m.apply(fn)
        fn(self)
        return self

    def _child(self, name):
        return getattr(self, name)

    def register_buffer(self, name, value, persistent=True):
        setattr(self, name, value)

    def add_module(self, name, module):
        setattr(self, name, module)

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
    """Children in order, under the names upstream gives them: positions,
    or the keys of an OrderedDict, or add_module's name."""

    def __init__(self, *mods):
        self._names, self._mods = [], []
        if len(mods) == 1 and isinstance(mods[0], dict):
            for k, m in mods[0].items():
                self.add_module(k, m)
        else:
            for m in mods:
                self.append(m)

    def add_module(self, name, module):
        if name in self._names:
            self._mods[self._names.index(name)] = module
        else:
            self._names.append(name)
            self._mods.append(module)

    def __setattr__(self, name, value):
        # as torch.nn.Module registers it: a module assigned as an attribute
        # of a Sequential is its next child, run in assignment order
        if isinstance(value, Module) and not name.startswith("_"):
            self.add_module(name, value)
        object.__setattr__(self, name, value)

    def _child(self, name):
        return self._mods[self._names.index(name)]

    def __getattr__(self, name):
        if not name.startswith("_") and name in self.__dict__.get(
                "_names", ()):
            return self._child(name)
        raise AttributeError(name)

    def named_children(self):
        return iter(zip(self._names, self._mods))

    def __getitem__(self, i):
        if isinstance(i, slice):
            s = type(self)()
            for n, m in zip(self._names[i], self._mods[i]):
                s.add_module(n, m)
            return s
        return self._mods[i]

    def __len__(self):
        return len(self._mods)

    def __iter__(self):
        return iter(self._mods)

    def append(self, m):
        self.add_module(str(len(self._mods)), m)
        return self

    def forward(self, x):
        for m in self._mods:
            x = m(x)
        return x


class ModuleList(Sequential):
    def __init__(self, modules=None):
        Sequential.__init__(self)
        for m in modules or []:
            self.append(m)

    def forward(self, *a):
        raise NotImplementedError("ModuleList is not callable")


class ModuleDict(Sequential):
    def __init__(self, modules=None):
        Sequential.__init__(self)
        for k, m in (modules or {}).items():
            self.add_module(k, m)

    def items(self):
        return list(zip(self._names, self._mods))

    def keys(self):
        return list(self._names)

    def values(self):
        return list(self._mods)

    def __getitem__(self, k):
        return self._child(k)

    def __iter__(self):
        return iter(self._names)

    def forward(self, *a):
        raise NotImplementedError("ModuleDict is not callable")


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


class Softmax(Module):
    def __init__(self, dim=None):
        self.dim = dim

    def forward(self, x):
        return x.softmax(self.dim)


class Sigmoid(Module):
    def forward(self, x):
        return x.sigmoid()


class LeakyReLU(Module):
    def __init__(self, negative_slope=0.01, inplace=False):
        self.negative_slope = negative_slope

    def forward(self, x):
        return F.leaky_relu(x, self.negative_slope)


class Embedding(Module):
    def __init__(self, num_embeddings, embedding_dim, padding_idx=None,
                 max_norm=None, norm_type=2.0, scale_grad_by_freq=False,
                 sparse=False, _weight=None, _freeze=False, device=None,
                 dtype=None):
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx
        self.weight = Parameter(num_embeddings, embedding_dim)

    def forward(self, idx):
        return F.embedding(idx, self.weight)


class LayerNorm(Module):
    def __init__(self, normalized_shape, eps=1e-5, elementwise_affine=True,
                 bias=True, device=None, dtype=None):
        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = tuple(normalized_shape)
        self.eps = eps
        self.weight = Parameter(*self.normalized_shape)
        self.bias = Parameter(*self.normalized_shape) if bias else None

    def forward(self, x):
        return F.layer_norm(x, self.normalized_shape, self.weight, self.bias,
                            self.eps)


class SiLU(Module):
    def __init__(self, inplace=False):
        pass

    def forward(self, x):
        return F.silu(x)


class Hardswish(Module):
    def __init__(self, inplace=False):
        pass

    def forward(self, x):
        return F.hardswish(x)


class ReLU6(Module):
    def __init__(self, inplace=False):
        pass

    def forward(self, x):
        return F.relu6(x)


class Hardsigmoid(Module):
    def __init__(self, inplace=False):
        pass

    def forward(self, x):
        return F.hardsigmoid(x)


class Hardtanh(Module):
    def __init__(self, min_val=-1.0, max_val=1.0, inplace=False):
        self.min_val, self.max_val = min_val, max_val

    def forward(self, x):
        return F.hardtanh(x, self.min_val, self.max_val)


class PReLU(Module):
    def __init__(self, num_parameters=1, init=0.25):
        self.weight = Parameter(num_parameters)

    def forward(self, x):
        raise NotImplementedError("PReLU")


class CrossEntropyLoss(Module):
    def __init__(self, weight=None, size_average=None, ignore_index=-100,
                 reduce=None, reduction="mean", label_smoothing=0.0):
        self.ignore_index = ignore_index
        self.reduction = reduction
        self.label_smoothing = label_smoothing

    def forward(self, input, target):
        return F.cross_entropy(input, target, ignore_index=self.ignore_index,
                               reduction=self.reduction,
                               label_smoothing=self.label_smoothing)


class MSELoss(Module):
    def __init__(self, size_average=None, reduce=None, reduction="mean"):
        self.reduction = reduction

    def forward(self, input, target):
        d = input - target
        return (d * d).mean()


class BCEWithLogitsLoss(Module):
    def __init__(self, weight=None, size_average=None, reduce=None,
                 reduction="mean", pos_weight=None):
        pass

    def forward(self, input, target):
        raise NotImplementedError("BCEWithLogitsLoss")


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
        self.padding = padding if isinstance(padding, str) \
            else _pair(padding)
        self.dilation = _pair(dilation)
        self.groups = groups
        self.padding_mode = padding_mode
        self.weight = Parameter(out_channels, in_channels // groups,
                                *self.kernel_size)
        self.bias = Parameter(out_channels) if bias else None

    def reset_parameters(self):
        pass

    def _prepare(self):
        if self.padding_mode != "zeros":
            raise NotImplementedError("%s padding" % self.padding_mode)
        if isinstance(self.padding, str):
            if self.padding == "valid":
                self.padding = (0, 0)
            elif self.stride == (1, 1) and all(
                    (d * (k - 1)) % 2 == 0 for d, k in
                    zip(self.dilation, self.kernel_size)):
                self.padding = tuple(d * (k - 1) // 2 for d, k in
                                     zip(self.dilation, self.kernel_size))
            else:
                raise NotImplementedError("padding=%r" % self.padding)
        # [O, C/g, kh, kw] -> [C*kh*kw, O/g]: group g's rows are its C/g
        # input channels x kernel row x kernel col, the matrix the grouped
        # im2col's columns multiply (with g = 1 the ungrouped layout)
        self.filter = F._group_filter(self.weight, self.groups)

    def forward(self, x):
        n, c, h, w = x.shape
        b = self.bias.raw if isinstance(self.bias, Tensor) else None
        (kh, kw), (sh, sw) = self.kernel_size, self.stride
        (ph, pw), (dh, dw) = self.padding, self.dilation
        y = x.dense().raw.conv2d(self.filter.raw, c, h, w, b, kh, sh, ph,
                                 self.groups, kw, sw, pw, dh, dw)
        oh = (h + 2 * ph - dh * (kh - 1) - 1) // sh + 1
        ow = (w + 2 * pw - dw * (kw - 1) - 1) // sw + 1
        return Tensor(y, (n, self.out_channels, oh, ow))


class Conv1d(Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True,
                 padding_mode="zeros", device=None, dtype=None):
        _one = lambda v: v[0] if isinstance(v, (tuple, list)) else v
        self.conv = Conv2d(in_channels, out_channels,
                           (1, _one(kernel_size)), (1, _one(stride)),
                           (0, _one(padding)) if not isinstance(padding, str)
                           else padding, (1, _one(dilation)), groups, bias,
                           padding_mode)
        self.weight = Parameter()
        self.bias = Parameter() if bias else None

    def _prepare(self):
        o = self.weight.shape[0]
        self.conv.weight = self.weight.view(o, self.weight.shape[1], 1,
                                            self.weight.shape[2])
        self.conv.bias = self.bias
        self.conv._prepare()

    def forward(self, x):
        n, c, l = x.shape
        y = self.conv(x.view(n, c, 1, l))
        return y.view(n, y.shape[1], y.shape[3])


class ConvTranspose2d(Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, output_padding=0, groups=1, bias=True,
                 dilation=1, padding_mode="zeros", device=None, dtype=None):
        if groups != 1:
            raise NotImplementedError("grouped transposed convolution")
        self.out_channels = out_channels
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)
        self.padding = _pair(padding)
        self.output_padding = _pair(output_padding)
        self.dilation = _pair(dilation)
        self.weight = Parameter(in_channels, out_channels, *self.kernel_size)
        self.bias = Parameter(out_channels) if bias else None

    def _prepare(self):
        # [C, O, kh, kw] -> [C*kh*kw, O]
        w = self.weight
        c, o, kh, kw = w.shape
        self.filter = F._matrix(F._host_permute(w, (0, 2, 3, 1)),
                                c * kh * kw, o)

    def forward(self, x):
        n, c, h, w = x.shape
        b = self.bias.raw if isinstance(self.bias, Tensor) else None
        (kh, kw), (sh, sw) = self.kernel_size, self.stride
        (ph, pw), (oph, opw) = self.padding, self.output_padding
        dh, dw = self.dilation
        y = x.dense().raw.conv_transpose2d(self.filter.raw, c, h, w, b, kh,
                                           sh, ph, oph, dh, kw, sw, pw, opw,
                                           dw)
        oh = (h - 1) * sh - 2 * ph + dh * (kh - 1) + oph + 1
        ow = (w - 1) * sw - 2 * pw + dw * (kw - 1) + opw + 1
        return Tensor(y, (n, self.out_channels, oh, ow))


class BatchNorm2d(Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True,
                 track_running_stats=True, device=None, dtype=None):
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats
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
        y = x.raw.reshape([n * c, h * w]).mul(scale.raw).add(shift.raw)
        return Tensor(y, x.shape)


class MaxPool2d(Module):
    def __init__(self, kernel_size, stride=None, padding=0, dilation=1,
                 return_indices=False, ceil_mode=False):
        self.kernel_size = kernel_size
        self.stride = stride if stride not in (None, []) else kernel_size
        self.padding = padding
        self.ceil_mode = ceil_mode
        if dilation != 1 or return_indices:
            raise NotImplementedError("max pool dilation/indices")

    def forward(self, x):
        n, c, h, w = x.shape
        k = _square("kernel", self.kernel_size)
        s = _square("stride", self.stride)
        p = _square("padding", self.padding)
        x = x.dense()
        oh = (h + 2 * p - k) // s + 1
        ow = (w + 2 * p - k) // s + 1
        if self.ceil_mode:
            # ceil mode adds a last window where one starts inside the input
            # (or its left padding); padding the bottom/right with -inf and
            # pooling in floor mode takes exactly those windows
            ch = -(-(h + 2 * p - k) // s) + 1
            cw = -(-(w + 2 * p - k) // s) + 1
            if (ch - 1) * s >= h + p:
                ch -= 1
            if (cw - 1) * s >= w + p:
                cw -= 1
            eh, ew = (ch - 1) * s + k - (h + 2 * p), \
                (cw - 1) * s + k - (w + 2 * p)
            if eh > 0 or ew > 0:
                eh, ew = max(eh, 0), max(ew, 0)
                x = Tensor(x.raw.pad([n, c, h, w],
                                     [0, 0, 0, 0, 0, eh, 0, ew],
                                     float("-inf")), (n, c, h + eh, w + ew))
                h, w = h + eh, w + ew
            oh, ow = ch, cw
        y = x.raw.maxpool2(c, h, w, k, s, p)
        return Tensor(y, (n, c, oh, ow))


class AvgPool2d(Module):
    def __init__(self, kernel_size, stride=None, padding=0, ceil_mode=False,
                 count_include_pad=True, divisor_override=None):
        if divisor_override is not None:
            raise NotImplementedError("divisor_override")
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.ceil_mode = ceil_mode
        self.count_include_pad = count_include_pad

    def forward(self, x):
        return F.avg_pool2d(x, self.kernel_size, self.stride, self.padding,
                            self.ceil_mode, self.count_include_pad)


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


class BatchNorm1d(Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True,
                 track_running_stats=True, device=None, dtype=None):
        self.weight = Parameter(num_features)
        self.bias = Parameter(num_features)

    def forward(self, x):
        raise NotImplementedError("BatchNorm1d")
