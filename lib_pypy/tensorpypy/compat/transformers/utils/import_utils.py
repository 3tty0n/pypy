# The torch the reference run used; transformers gates code paths on it.
REFERENCE_TORCH = (2, 14, 0)


def is_torch_greater_or_equal(version, accept_dev=False):
    want = tuple(int(p) for p in version.split(".")[:3])
    return REFERENCE_TORCH >= want


def is_torchdynamo_compiling():
    return False
