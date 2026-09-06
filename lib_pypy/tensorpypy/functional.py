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


_ERF_A = [0.254829592, -0.284496736, 1.421413741, -1.453152027, 1.061405429]


def gelu_erf(x):
    import math
    dt = x.dtype
    one = _scalar(1.0, dt)
    u = x.relu().add(x.mul(_scalar(-1.0, dt)).relu())
    z = u.mul(_scalar(1.0 / math.sqrt(2.0), dt))
    t = one.div(z.mul(_scalar(0.3275911, dt)).add(one))
    poly = _scalar(_ERF_A[4], dt)
    for i in [3, 2, 1, 0]:
        poly = poly.mul(t).add(_scalar(_ERF_A[i], dt))
    e = one.sub(poly.mul(t).mul(z.mul(z).mul(_scalar(-1.0, dt)).exp()))
    return x.add(u.mul(e)).mul(_scalar(0.5, dt))
