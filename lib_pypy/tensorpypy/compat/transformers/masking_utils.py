"""transformers.masking_utils (5.13.0), for inputs without padding.

With the sdpa implementation upstream builds no mask at all when it can
prove none is needed - no padding, no cache - and lets sdpa's is_causal flag
(or nothing) do the work.  With the eager implementation (a model whose
attention adds the mask itself) it builds the additive float mask
[B, 1, Q, KV]: 0 where a position may attend, the dtype's minimum where it
may not; for a bidirectional mask without padding that is all zeros, which
adds nothing, so None stands for it.  A 4-D mask is passed through as
upstream passes it.  A padded 2-D mask raises."""
import torch


def _no_padding(attention_mask):
    if attention_mask is None:
        return True
    if attention_mask.is_host:
        return all(v == 1 for v in attention_mask.host)
    return all(v == 1.0 for v in attention_mask.tolist())


def create_bidirectional_mask(config, inputs_embeds, attention_mask,
                              encoder_hidden_states=None, past_key_values=None,
                              or_mask_function=None, and_mask_function=None,
                              **kwargs):
    if attention_mask is not None and attention_mask.dim() == 4:
        return attention_mask
    if or_mask_function or and_mask_function:
        raise NotImplementedError("mask functions")
    if _no_padding(attention_mask):
        return None
    raise NotImplementedError("padded attention mask")


def create_causal_mask(config, inputs_embeds, attention_mask,
                       cache_position=None, past_key_values=None,
                       position_ids=None, or_mask_function=None,
                       and_mask_function=None, **kwargs):
    if attention_mask is not None and attention_mask.dim() == 4:
        return attention_mask
    if past_key_values is not None:
        raise NotImplementedError("causal mask with a cache")
    if or_mask_function or and_mask_function:
        raise NotImplementedError("mask functions")
    if not _no_padding(attention_mask):
        raise NotImplementedError("padded attention mask")
    impl = getattr(config, "_attn_implementation", "sdpa")
    if impl == "sdpa":
        return None
    if impl != "eager":
        raise NotImplementedError("%s attention masks" % impl)
    b, q = inputs_embeds.shape[0], inputs_embeds.shape[1]
    return _eager_causal(b, q, inputs_embeds.dtype)


_eager_masks = {}


def _eager_causal(b, q, dtype):
    m = _eager_masks.get((b, q, dtype))
    if m is None:
        neg = torch.finfo(dtype).min
        m = torch.from_flat([0.0 if j <= i else neg for i in range(q)
                             for j in range(q)], (1, 1, q, q), dtype)
        m = m.expand((b, 1, q, q))
        _eager_masks[(b, q, dtype)] = m
    return m


def create_sliding_window_causal_mask(config, inputs_embeds, attention_mask,
                                      cache_position=None,
                                      past_key_values=None,
                                      position_ids=None, **kwargs):
    window = getattr(config, "sliding_window", None)
    if window is not None and window < inputs_embeds.shape[1]:
        raise NotImplementedError("sliding window shorter than the input")
    return create_causal_mask(config, inputs_embeds, attention_mask,
                              cache_position, past_key_values, position_ids,
                              **kwargs)
