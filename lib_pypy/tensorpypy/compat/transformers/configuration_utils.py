class PreTrainedConfig(object):
    """A model config: the exported config dict as attributes, with the
    defaults PreTrainedConfig gives the output switches."""

    def __init__(self, **kwargs):
        self.return_dict = True
        self.output_attentions = False
        self.output_hidden_states = False
        self._attn_implementation = "sdpa"
        self.__dict__.update(kwargs)
        if self._attn_implementation is None:
            self._attn_implementation = "sdpa"

    @property
    def use_return_dict(self):
        return self.return_dict

    def get_text_config(self, decoder=False):
        return self


PretrainedConfig = PreTrainedConfig
