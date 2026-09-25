"""Flash attention: only reached when a config asks for it, which the
dashboard's eval runs (sdpa) do not."""


class FlashAttentionKwargs(dict):
    pass


def _flash_attention_forward(*args, **kwargs):
    raise NotImplementedError("flash attention")


def flash_attn_supports_top_left_mask():
    return False
