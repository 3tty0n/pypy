def _native(v):
    # json gives unicode on Python 2; upstream tests isinstance(v, str)
    if isinstance(v, unicode):
        return v.encode("utf-8")
    if isinstance(v, list):
        return [_native(x) for x in v]
    if isinstance(v, dict):
        return dict((_native(k), _native(x)) for k, x in v.items())
    return v


class PreTrainedConfig(object):
    """A model config: the exported config dict as attributes, with the
    defaults PreTrainedConfig gives the output switches."""

    def __init__(self, **kwargs):
        self.return_dict = True
        self.output_attentions = False
        self.output_hidden_states = False
        self._attn_implementation = "sdpa"
        self.__dict__.update(_native(kwargs))
        if self._attn_implementation is None:
            self._attn_implementation = "sdpa"

    def __getattr__(self, name):
        # upstream's attribute_map aliases (GPT-2's hidden_size -> n_embd)
        amap = self.__dict__.get("attribute_map") or {}
        if name in amap:
            return getattr(self, amap[name])
        raise AttributeError(name)

    def __setattr__(self, name, value):
        amap = self.__dict__.get("attribute_map") or {}
        object.__setattr__(self, amap.get(name, name), value)

    @property
    def use_return_dict(self):
        return self.return_dict

    def get_text_config(self, decoder=False):
        return self


PretrainedConfig = PreTrainedConfig
