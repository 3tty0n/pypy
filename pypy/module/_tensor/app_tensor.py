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


def tensor(data, shape=None, requires_grad=False):
    import _tensor
    if shape is None:
        shape = _shape_of(data)
    flat = []
    _flatten(data, flat)
    return _tensor._tensor_flat(flat, shape, requires_grad)
