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
    rows = x.shape[0]
    m = x.max(1).reshape([rows, 1])
    e = x.sub(m).exp()
    s = e.sum(1).reshape([rows, 1])
    return e.div(s)


def layer_norm(x, gamma, beta, eps=1e-5):
    rows, cols = x.shape
    dt = x.dtype
    inv = _scalar(1.0 / cols, dt)
    s1 = x.sum(1).reshape([rows, 1])
    s2 = x.mul(x).sum(1).reshape([rows, 1])
    mean = s1.mul(inv)
    var = s2.mul(inv).sub(mean.mul(mean))
    denom = var.add(_scalar(eps, dt)).sqrt()
    return x.sub(mean).div(denom).mul(gamma).add(beta)


def rms_norm(x, gamma, eps=1e-5):
    rows, cols = x.shape
    dt = x.dtype
    inv = _scalar(1.0 / cols, dt)
    denom = x.mul(x).sum(1).mul(inv).add(_scalar(eps, dt)).sqrt()
    return x.div(denom.reshape([rows, 1])).mul(gamma)


def gelu(x):
    import math
    dt = x.dtype
    c = _scalar(math.sqrt(2.0 / math.pi), dt)
    x3 = x.mul(x).mul(x).mul(_scalar(0.044715, dt))
    z = x.add(x3).mul(c).mul(_scalar(2.0, dt))
    one = _scalar(1.0, dt)
    tanh = one.sub(_scalar(2.0, dt).div(z.exp().add(one)))
    return x.mul(_scalar(0.5, dt)).mul(tanh.add(one))


def silu(x):
    dt = x.dtype
    one = _scalar(1.0, dt)
    return x.div(one.add(x.mul(_scalar(-1.0, dt)).exp()))
