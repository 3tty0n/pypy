"""transformers.utils.output_capturing.capture_outputs (5.13.0) when no
hidden states or attentions are asked for - the dashboard's eval forward.
Asking for them raises."""
from functools import wraps


def capture_outputs(func=None, tie_last_hidden_states=True):
    def wrapped_fn(func):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            return_dict = kwargs.pop("return_dict",
                                     getattr(self.config, "return_dict", True))
            for k in ("output_attentions", "output_hidden_states"):
                if kwargs.get(k, getattr(self.config, k, False)):
                    raise NotImplementedError(k)
            outputs = func(self, *args, **kwargs)
            if return_dict is False:
                outputs = outputs.to_tuple()
            return outputs
        return wrapper
    if func is not None:
        return wrapped_fn(func)
    return wrapped_fn


class OutputRecorder(object):
    """Declares which submodule outputs capture_outputs may record; only
    read when outputs are requested."""

    def __init__(self, target_class, index=0, layer_name=None,
                 class_name=None):
        self.target_class = target_class
        self.index = index
        self.layer_name = layer_name
        self.class_name = class_name
