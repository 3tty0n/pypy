import torch
from torch import Tensor


def relu(x, inplace=False):
    return x.relu()


def gelu(x, approximate="none"):
    if approximate == "tanh":
        return Tensor(x.raw.gelu(), x.shape, x.perm, col=x.col)
    return Tensor(x.raw.gelu_erf(), x.shape, x.perm, col=x.col)


def silu(x, inplace=False):
    return Tensor(x.raw.silu(), x.shape, x.perm, col=x.col)


def hardtanh(x, min_val=-1.0, max_val=1.0, inplace=False):
    return x.clamp(min_val, max_val)


def relu6(x, inplace=False):
    return x.clamp(0.0, 6.0)


def hardsigmoid(x, inplace=False):
    return (x + 3.0).clamp(0.0, 6.0) / 6.0


def hardswish(x, inplace=False):
    return x * (x + 3.0).clamp(0.0, 6.0) / 6.0


def sigmoid(x):
    return x.sigmoid()


def tanh(x):
    return x.tanh()


def leaky_relu(x, negative_slope=0.01, inplace=False):
    return torch.where(x > 0.0, x, x * negative_slope)


def elu(x, alpha=1.0, inplace=False):
    return torch.where(x > 0.0, x, (x.exp() - 1.0) * alpha)


def max_pool2d(x, kernel_size, stride=None, padding=0, dilation=1,
               ceil_mode=False, return_indices=False):
    from torch.nn import MaxPool2d
    return MaxPool2d(kernel_size, stride, padding, dilation, return_indices,
                     ceil_mode)(x)


def log_softmax(x, dim=-1, dtype=None):
    x = x.dense()
    nd = len(x.shape)
    if (dim + nd if dim < 0 else dim) != nd - 1:
        raise NotImplementedError("log_softmax over dim %d" % dim)
    m = x.amax(-1, keepdim=True)
    z = x - m
    return z - z.exp().sum(-1, keepdim=True).log()


def cross_entropy(input, target, weight=None, size_average=None,
                  ignore_index=-100, reduce=None, reduction="mean",
                  label_smoothing=0.0):
    """log_softmax then nll_loss, as torch composes them.  The targets are
    index data on the host, so picking each row's target log-probability is
    a row gather from the flattened [N*V, 1] log-probabilities at indices
    i*V + t_i; rows whose target is ignore_index are left out of the sum and
    of the count, as nll_loss leaves them out."""
    if weight is not None or label_smoothing or reduction != "mean":
        raise NotImplementedError("cross_entropy options")
    if not target.is_host:
        raise NotImplementedError("cross_entropy with device targets")
    n, v = input.shape
    logp = log_softmax(input, -1)
    # an ignored row gathers index -1, which the kernel reads as 0
    idx = [t if t != ignore_index else -1 for t in target.host]
    keep = len([t for t in target.host if t != ignore_index])
    if keep == 0:
        return torch.from_flat([float("nan")], (), input.dtype)
    ix = torch._host(idx, (n, 1))
    picked = logp.dense().raw.gather(ix.index_tensor(input.dtype), n, 1)
    return Tensor(picked.sum(), ()) * (-1.0 / keep)


def pad(x, pad, mode="constant", value=0.0):
    """Constant padding of the last len(pad) // 2 dims (torch's order: last
    dim first); index tensors (the loss's shifted labels) on the host."""
    if mode != "constant":
        raise NotImplementedError("%s padding" % mode)
    if not x.is_host:
        x = x.dense()
        nd = len(x.shape)
        pads = [0] * (2 * nd)
        for k in range(len(pad) // 2):
            d = nd - 1 - k
            pads[2 * d], pads[2 * d + 1] = pad[2 * k], pad[2 * k + 1]
        shape = tuple(x.shape[d] + pads[2 * d] + pads[2 * d + 1]
                      for d in range(nd))
        return Tensor(x.raw.pad(list(x.shape), pads, float(value)), shape)
    shape = list(x.shape)
    vals = x.host
    for k in range(len(pad) // 2):
        before, after = pad[2 * k], pad[2 * k + 1]
        d = len(shape) - 1 - k
        inner = torch._numel(shape[d + 1:])
        outer = torch._numel(shape[:d])
        n = shape[d]
        out = []
        for o in range(outer):
            base = o * n * inner
            out.extend([value] * (before * inner))
            out.extend(vals[base:base + n * inner])
            out.extend([value] * (after * inner))
        vals = out
        shape[d] = n + before + after
    return torch._host(vals, shape)


def linear(x, weight, bias=None):
    """x @ weight.T + bias, weight in torch's [out, in]."""
    n = x.shape[-1]
    out = weight.shape[0]
    rows = x.numel() // n
    y = x.raw.reshape([rows, n]).matmul(weight.raw, True,
                                      torch.backends.cuda.matmul.allow_tf32)
    if bias is not None:
        y = y.add(bias.raw)
    return Tensor(y, x.shape[:-1] + (out,))


def embedding(idx, weight, padding_idx=None, max_norm=None, norm_type=2.0,
              scale_grad_by_freq=False, sparse=False):
    if not idx.is_host:
        raise NotImplementedError("embedding indexed by a device tensor")
    rows = weight.raw.take(idx.index_tensor(weight.dtype))
    return Tensor(rows, tuple(idx.shape) + (weight.shape[1],))


def layer_norm(x, normalized_shape, weight=None, bias=None, eps=1e-5):
    if weight is None or bias is None:
        raise NotImplementedError("layer_norm without affine parameters")
    x = x.dense()
    d = torch._numel(normalized_shape)
    y = x.raw.reshape([x.numel() // d, d]).layer_norm(weight.raw, bias.raw,
                                                      eps)
    return Tensor(y, x.shape)


_BSHD = (0, 2, 1, 3)


def scaled_dot_product_attention(query, key, value, attn_mask=None,
                                 dropout_p=0.0, is_causal=False, scale=None,
                                 enable_gqa=False):
    """softmax(q k^T * scale + mask) v for q, k, v viewed [B, H, S, D] over
    [B, S, H, D] storage - how every q/k/v projection followed by
    view(B, S, H, D).transpose(1, 2) arrives.  The scores kernel reads the
    heads straight out of the [B*S, H*D] projection rows, and the context
    kernel writes them back the same way, so the result is again a
    [B, H, S, D] view of [B, S, H, D] storage: the transpose(1, 2) callers
    apply next makes it dense without a copy."""
    import math
    for t in (query, key, value):
        if not torch._same(t.perm, _BSHD):
            raise NotImplementedError(
                "attention over a %s layout" % (t.perm,))
    b, h, s, d = query.shape
    if not (torch._same(key.shape, query.shape) and
            torch._same(value.shape, query.shape)):
        raise NotImplementedError("attention with k/v shaped %s, q %s"
                                  % (list(key.shape), list(query.shape)))
    q, qo = _rows(query, b * s, h * d)
    k, ko = _rows(key, b * s, h * d)
    v, vo = _rows(value, b * s, h * d)
    scores = Tensor(q.attn_scores(k, h, h * d, qo, ko, b), (b, h, s, s))
    scores = scores * (scale if scale is not None else 1.0 / math.sqrt(d))
    if is_causal:
        scores = scores + _causal(s, query.dtype)
    if attn_mask is not None:
        if attn_mask.dtype == torch.bool or attn_mask.is_host:
            raise NotImplementedError("boolean attention mask")
        scores = scores + attn_mask
    p = scores.softmax(-1)
    ctx = p.dense().raw.reshape([b * h * s, s]).attn_context(v, h, h * d, vo,
                                                             b)
    return Tensor(ctx, (b, h, s, d), _BSHD)


def _rows(t, rows, width):
    """t's storage as the [rows, ld] matrix the attention kernels read, and
    the column its heads start at: a dense projection, or a window of a
    packed one."""
    if t.col is None:
        return t.raw.reshape([rows, width]), 0
    off, ld, ndw = t.col
    if ndw != 2:
        raise NotImplementedError("attention over a window of %d dims" % ndw)
    return t.raw.reshape([rows, ld]), off


_causal_masks = {}


def _causal(s, dtype):
    m = _causal_masks.get((s, dtype))
    if m is None:
        neg = torch.finfo(dtype).min
        m = torch.from_flat([0.0 if j <= i else neg for i in range(s)
                             for j in range(s)], (1, 1, s, s), dtype)
        _causal_masks[(s, dtype)] = m
    return m


def adaptive_avg_pool2d(x, output_size):
    n, c, h, w = x.shape
    if not isinstance(output_size, (tuple, list)):
        output_size = (output_size, output_size)
    oh = h if output_size[0] is None else output_size[0]
    ow = w if output_size[1] is None else output_size[1]
    if (oh, ow) == (h, w):
        return x
    y = x.dense().raw.adaptive_avg_pool2d(c, h, w, oh, ow)
    return Tensor(y, (n, c, oh, ow))


def _pair(v):
    return tuple(v) if isinstance(v, (tuple, list)) else (v, v)


def avg_pool2d(x, kernel_size, stride=None, padding=0, ceil_mode=False,
               count_include_pad=True, divisor_override=None):
    n, c, h, w = x.shape
    (kh, kw) = _pair(kernel_size)
    (sh, sw) = _pair(stride if stride not in (None, []) else kernel_size)
    (ph, pw) = _pair(padding)
    y = x.dense().raw.avg_pool2d(c, h, w, kh, sh, ph, count_include_pad,
                                 ceil_mode, kw, sw, pw)
    rnd = (lambda a, b: -(-a // b)) if ceil_mode else (lambda a, b: a // b)
    oh = rnd(h + 2 * ph - kh, sh) + 1
    ow = rnd(w + 2 * pw - kw, sw) + 1
    if ceil_mode:
        if (oh - 1) * sh >= h + ph:
            oh -= 1
        if (ow - 1) * sw >= w + pw:
            ow -= 1
    return Tensor(y, (n, c, oh, ow))


def _host_permute(x, order):
    """A permutation of a parameter, done once on the host at load time."""
    shape = x.shape
    st = torch._strides(shape)
    v = x.tolist()
    nshape = [shape[d] for d in order]
    nst = [st[d] for d in order]
    out = []

    def walk(level, off):
        if level == len(nshape):
            out.append(v[off])
            return
        for i in range(nshape[level]):
            walk(level + 1, off + i * nst[level])
    walk(0, 0)
    return torch.from_flat(out, nshape, x.dtype)


def _group_filter(w, groups):
    o = w.shape[0]
    per = w.numel() // o
    g = groups
    # [g, O/g, per] -> [g, per, O/g] -> [g*per, O/g]
    return _matrix(_host_permute(w.view(g, o // g, per), (0, 2, 1)),
                   g * per, o // g)


def _matrix(x, rows, cols):
    """x with its storage itself [rows, cols], as the engine's convolution
    and matmul kernels check it."""
    return Tensor(x.raw.reshape([rows, cols]), (rows, cols))


def softmax(x, dim=-1, dtype=None):
    return x.softmax(dim)


def dropout(x, p=0.5, training=False, inplace=False):
    return x


def _transpose2(x):
    return _host_permute(x, (1, 0))
