import torch.nn as nn

from transformers.integrations.sdpa_attention import sdpa_attention_forward


class AttentionInterface(object):
    """upstream's registry: the implementation named in the config, or the
    model's own eager function when the config says eager."""
    _global_mapping = {"sdpa": sdpa_attention_forward}

    def get_interface(self, attn_implementation, default):
        if attn_implementation == "eager" or attn_implementation is None:
            return default
        f = self._global_mapping.get(attn_implementation)
        if f is None:
            raise NotImplementedError("attention implementation %r"
                                      % attn_implementation)
        return f


ALL_ATTENTION_FUNCTIONS = AttentionInterface()


class PreTrainedModel(nn.Module):
    """The model base, for a model whose tensors arrive by
    load_state_dict: post_init's weight initialisation and tying have
    nothing to do, the tensors being the checkpoint's."""
    config_class = None
    base_model_prefix = ""

    def __init__(self, config, *inputs, **kwargs):
        self.config = config

    def post_init(self):
        pass

    def warn_if_padding_and_no_attention_mask(self, input_ids,
                                              attention_mask):
        """A warning upstream; no effect on what is computed."""

    @property
    def dtype(self):
        return "float32"
