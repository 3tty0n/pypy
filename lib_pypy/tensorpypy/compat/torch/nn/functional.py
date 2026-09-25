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


def hardswish(x, inplace=False):
    raise NotImplementedError("hardswish")


def hardtanh(x, min_val=-1.0, max_val=1.0, inplace=False):
    raise NotImplementedError("hardtanh")


def leaky_relu(x, negative_slope=0.01, inplace=False):
    raise NotImplementedError("leaky_relu")


def cross_entropy(input, target, weight=None, size_average=None,
                  ignore_index=-100, reduce=None, reduction="mean",
                  label_smoothing=0.0):
    raise NotImplementedError("cross_entropy")


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
    oh, ow = output_size
    if (oh, ow) == (h, w):
        return x
    if (oh, ow) == (1, 1):
        y = x.raw.reshape([n * c, h * w]).sum(1)
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
