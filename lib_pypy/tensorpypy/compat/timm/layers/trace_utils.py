"""timm.layers.trace_utils.  Upstream imports torch._assert in a top-level
try; torch's _assert is `assert condition, message` for a Python bool."""


def _assert(condition, message):
    assert condition, message


def _float_to_int(x):
    return int(x)
