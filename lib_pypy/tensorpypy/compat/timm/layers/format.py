"""timm.layers.format.  Upstream's Format is a str Enum, which Python 2 has
no module for; here each member is a str subclass instance and Format(v)
looks a member up by value, as Enum does, so the `is` comparisons upstream
makes hold."""


class _Member(str):
    pass


class Format(object):
    NCHW = _Member('NCHW')
    NHWC = _Member('NHWC')
    NCL = _Member('NCL')
    NLC = _Member('NLC')

    def __new__(cls, value):
        for m in (cls.NCHW, cls.NHWC, cls.NCL, cls.NLC):
            if m == value:
                return m
        raise ValueError("%r is not a valid Format" % (value,))


def get_spatial_dim(fmt):
    fmt = Format(fmt)
    if fmt is Format.NLC:
        dim = (1,)
    elif fmt is Format.NCL:
        dim = (2,)
    elif fmt is Format.NHWC:
        dim = (1, 2)
    else:
        dim = (2, 3)
    return dim


def get_channel_dim(fmt):
    fmt = Format(fmt)
    if fmt is Format.NHWC:
        dim = 3
    elif fmt is Format.NLC:
        dim = 2
    else:
        dim = 1
    return dim


def nchw_to(x, fmt):
    if fmt == Format.NHWC:
        x = x.permute(0, 2, 3, 1)
    elif fmt == Format.NLC:
        x = x.flatten(2).transpose(1, 2)
    elif fmt == Format.NCL:
        x = x.flatten(2)
    return x


def nhwc_to(x, fmt):
    if fmt == Format.NCHW:
        x = x.permute(0, 3, 1, 2)
    elif fmt == Format.NLC:
        x = x.flatten(1, 2)
    elif fmt == Format.NCL:
        x = x.flatten(1, 2).transpose(1, 2)
    return x
