_scalars = {}


def _scalar(v, dtype="float64"):
    import _metatensor
    key = (v, dtype)
    t = _scalars.get(key)
    if t is None:
        t = _metatensor.tensor([v], None, False, dtype)
        _scalars[key] = t
    return t


def relu(x):
    return x.relu()


def softmax(x):
    return x.softmax()


def layer_norm(x, gamma, beta, eps=1e-5):
    return x.layer_norm(gamma, beta, eps)


def rms_norm(x, gamma, eps=1e-5):
    return x.rms_norm(gamma, eps)


def gelu(x):
    return x.gelu()


def silu(x):
    return x.silu()
