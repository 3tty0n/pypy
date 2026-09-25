"""timm.layers.create_act.  Upstream builds its tables from star imports of
activations and activations_me and completes them in top-level loops, which
the porter does not carry.  The tables are upstream's as the reference run
evaluates them: its torch has F.silu, F.hardswish, F.hardsigmoid and
F.mish, so those names map to torch's own, and the memory-efficient (_me)
variants are never chosen.  A name whose torch module the compat torch does
not have raises NotImplementedError when built."""
from torch import nn
from torch.nn import functional as F

from timm.layers.activations import (gelu, gelu_tanh, quick_gelu, sigmoid,
                                     tanh, hard_mish, PReLU, GELU, GELUTanh,
                                     QuickGELU, Sigmoid, Tanh, HardMish)
from timm.layers.config import is_exportable, is_scriptable


def _missing(name):
    def build(*args, **kwargs):
        raise NotImplementedError("activation %s" % name)
    return build


_ACT_FN_DEFAULT = dict(
    silu=F.silu,
    swish=F.silu,
    mish=_missing("mish"),
    relu=F.relu,
    relu6=F.relu6,
    leaky_relu=F.leaky_relu,
    elu=F.elu,
    celu=_missing("celu"),
    selu=_missing("selu"),
    gelu=gelu,
    gelu_tanh=gelu_tanh,
    quick_gelu=quick_gelu,
    sigmoid=sigmoid,
    tanh=tanh,
    hard_sigmoid=F.hardsigmoid,
    hard_swish=F.hardswish,
    hard_mish=hard_mish,
)

_ACT_FN_ME = dict(
    silu=F.silu,
    swish=F.silu,
    mish=_missing("mish"),
    hard_sigmoid=F.hardsigmoid,
    hard_swish=F.hardswish,
    hard_mish=_missing("hard_mish_me"),
)

_ACT_FNS = (_ACT_FN_ME, _ACT_FN_DEFAULT)
for a in _ACT_FNS:
    a.setdefault('hardsigmoid', a.get('hard_sigmoid'))
    a.setdefault('hardswish', a.get('hard_swish'))

_ACT_LAYER_DEFAULT = dict(
    silu=nn.SiLU,
    swish=nn.SiLU,
    mish=_missing("mish"),
    relu=nn.ReLU,
    relu6=nn.ReLU6,
    leaky_relu=nn.LeakyReLU,
    elu=_missing("elu"),
    prelu=PReLU,
    celu=_missing("celu"),
    selu=_missing("selu"),
    gelu=GELU,
    gelu_tanh=GELUTanh,
    quick_gelu=QuickGELU,
    sigmoid=Sigmoid,
    tanh=Tanh,
    hard_sigmoid=nn.Hardsigmoid,
    hard_swish=nn.Hardswish,
    hard_mish=HardMish,
    identity=nn.Identity,
)

_ACT_LAYER_ME = dict(
    silu=nn.SiLU,
    swish=nn.SiLU,
    mish=_missing("mish"),
    hard_sigmoid=nn.Hardsigmoid,
    hard_swish=nn.Hardswish,
    hard_mish=_missing("hard_mish_me"),
)

_ACT_LAYERS = (_ACT_LAYER_ME, _ACT_LAYER_DEFAULT)
for a in _ACT_LAYERS:
    a.setdefault('hardsigmoid', a.get('hard_sigmoid'))
    a.setdefault('hardswish', a.get('hard_swish'))


def get_act_fn(name='relu'):
    if not name:
        return None
    if callable(name):
        return name
    name = name.lower()
    if not (is_exportable() or is_scriptable()):
        if name in _ACT_FN_ME:
            return _ACT_FN_ME[name]
    return _ACT_FN_DEFAULT[name]


def get_act_layer(name='relu'):
    if name is None:
        return None
    if not isinstance(name, str):
        return name
    if not name:
        return None
    name = name.lower()
    if not (is_exportable() or is_scriptable()):
        if name in _ACT_LAYER_ME:
            return _ACT_LAYER_ME[name]
    return _ACT_LAYER_DEFAULT[name]


def create_act_layer(name, inplace=None, **kwargs):
    act_layer = get_act_layer(name)
    if act_layer is None:
        return None
    if inplace is None:
        return act_layer(**kwargs)
    try:
        return act_layer(inplace=inplace, **kwargs)
    except TypeError:
        return act_layer(**kwargs)
