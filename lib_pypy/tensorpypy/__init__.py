import _metatensor

Tensor = _metatensor.Tensor

float32 = "float32"
float64 = "float64"
float16 = "float16"


def _shape_of(data):
    shape = []
    x = data
    while isinstance(x, list):
        shape.append(len(x))
        if len(x) == 0:
            break
        x = x[0]
    return shape


def _flatten(data, out):
    if isinstance(data, list):
        for item in data:
            _flatten(item, out)
    else:
        out.append(float(data))


def asarray(obj, dtype=None):
    if isinstance(obj, Tensor):
        if dtype is not None and obj.dtype != dtype:
            return obj.astype(dtype)
        return obj
    shape = _shape_of(obj)
    flat = []
    _flatten(obj, flat)
    return _metatensor._tensor_flat(flat, shape, False, dtype or float64)


def zeros(shape, dtype=None):
    return _metatensor.zeros(shape, False, dtype or float64)


def ones_like(x):
    n = x.size
    return asarray([1.0] * n, dtype=x.dtype).reshape(list(x.shape))


def matmul(a, b):
    return a.matmul(b)


def sum(x, axis=None, keepdims=False):
    a = -1 if axis is None else axis
    r = x.sum(a)
    if keepdims and axis is not None:
        shape = list(x.shape)
        shape[axis] = 1
        r = r.reshape(shape)
    return r


def max(x, axis=None, keepdims=False):
    a = -1 if axis is None else axis
    r = x.max(a)
    if keepdims and axis is not None:
        shape = list(x.shape)
        shape[axis] = 1
        r = r.reshape(shape)
    return r


def exp(x):
    return x.exp()


def sqrt(x):
    return x.sqrt()


def reshape(x, shape):
    return x.reshape(shape)


def take(x, indices, axis=0):
    assert axis == 0
    return x.take(indices)


def astype(x, dtype):
    return x.astype(dtype)
