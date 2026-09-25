import logging as _logging

from transformers.utils import import_utils
from transformers.utils.import_utils import (  # noqa
    is_torch_greater_or_equal, is_torchdynamo_compiling)
from transformers.utils.generic import can_return_tuple  # noqa


class logging(object):
    """transformers.utils.logging.get_logger, with warning_once."""

    @staticmethod
    def get_logger(name=None):
        log = _logging.getLogger(name)
        seen = set()

        def warning_once(msg, *args):
            if msg not in seen:
                seen.add(msg)
                log.warning(msg, *args)
        log.warning_once = warning_once
        log.info_once = warning_once
        return log


def auto_docstring(obj=None, **kwargs):
    """Documentation only: returns what it decorates."""
    if obj is None:
        return lambda o: o
    return obj


def is_torch_npu_available():
    return False


def is_torch_xpu_available():
    return False


class TransformersKwargs(dict):
    pass
