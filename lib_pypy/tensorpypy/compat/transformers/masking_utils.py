"""transformers.masking_utils (5.13.0) for the sdpa implementation.

Upstream builds no mask at all when it can prove none is needed - no
padding in attention_mask, no cache - and lets sdpa's is_causal flag (or
nothing) do the work; that is the only case the dashboard's inputs reach.
A 4-D mask is passed through as upstream passes it.  A padded 2-D mask
raises."""


def _no_padding(attention_mask):
    if attention_mask is None:
        return True
    if attention_mask.is_host:
        return all(v == 1 for v in attention_mask.host)
    raise NotImplementedError("attention mask on the device")


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
    if _no_padding(attention_mask):
        return None
    raise NotImplementedError("padded attention mask")
