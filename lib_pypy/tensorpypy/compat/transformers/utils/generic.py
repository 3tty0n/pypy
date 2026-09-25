"""transformers.utils.generic's forward decorators (5.13.0), for an eval
forward: can_return_tuple and merge_with_config_defaults as upstream writes
them, less the debug_io and vision-feature branches."""
from functools import wraps


def can_return_tuple(func):
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        return_dict = self.config.return_dict if hasattr(self, "config") \
            else True
        return_dict_passed = kwargs.pop("return_dict", return_dict)
        if return_dict_passed is not None:
            return_dict = return_dict_passed
        output = func(self, *args, **kwargs)
        if not return_dict and not isinstance(output, tuple):
            output = output.to_tuple()
        return output
    return wrapper


def merge_with_config_defaults(func):
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        names = func.__code__.co_varnames
        if "use_cache" in names:
            i = names.index("use_cache") - 1
            if not (len(args) > i and args[i] is not None) and \
                    kwargs.get("use_cache") is None:
                v = getattr(self.config, "use_cache", None)
                if v is not None:
                    kwargs["use_cache"] = v
        is_causal = kwargs.get("is_causal",
                               getattr(self.config, "is_causal", None))
        if is_causal is not None:
            had = hasattr(self.config, "is_causal")
            old = getattr(self.config, "is_causal", None)
            self.config.is_causal = is_causal
            kwargs["is_causal"] = is_causal
            try:
                return func(self, *args, **kwargs)
            finally:
                if had:
                    self.config.is_causal = old
                else:
                    del self.config.is_causal
        return func(self, *args, **kwargs)
    return wrapper


class maybe_autocast(object):
    """torch.autocast when enabled; the dashboard's float32 eval runs with
    it disabled, and so this does nothing."""

    def __init__(self, *args, **kwargs):
        if kwargs.get("enabled", False):
            raise NotImplementedError("autocast")

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False
