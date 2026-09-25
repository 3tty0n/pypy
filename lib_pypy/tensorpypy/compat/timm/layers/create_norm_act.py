"""timm.layers.create_norm_act.  Upstream star-imports the EvoNorm layers,
which the porter cannot carry; the tables are upstream's, with the layers
that are not ported (EvoNorm, FilterResponseNorm, InplaceAbn) raising
NotImplementedError when built."""
import functools
import types

from timm.layers.norm_act import (BatchNormAct2d, GroupNormAct,
                                  GroupNorm1Act, LayerNormAct,
                                  LayerNormActFp32, LayerNormAct2d,
                                  LayerNormAct2dFp32, RmsNormAct,
                                  RmsNormActFp32, RmsNormAct2d,
                                  RmsNormAct2dFp32)


def _missing(name):
    def build(*args, **kwargs):
        raise NotImplementedError("norm-act layer %s" % name)
    return build


EvoNorm2dB0 = _missing("EvoNorm2dB0")
EvoNorm2dB1 = _missing("EvoNorm2dB1")
EvoNorm2dB2 = _missing("EvoNorm2dB2")
EvoNorm2dS0 = _missing("EvoNorm2dS0")
EvoNorm2dS0a = _missing("EvoNorm2dS0a")
EvoNorm2dS1 = _missing("EvoNorm2dS1")
EvoNorm2dS1a = _missing("EvoNorm2dS1a")
EvoNorm2dS2 = _missing("EvoNorm2dS2")
EvoNorm2dS2a = _missing("EvoNorm2dS2a")
FilterResponseNormAct2d = _missing("FilterResponseNormAct2d")
FilterResponseNormTlu2d = _missing("FilterResponseNormTlu2d")
InplaceAbn = _missing("InplaceAbn")

_NORM_ACT_MAP = dict(
    batchnorm=BatchNormAct2d,
    batchnorm2d=BatchNormAct2d,
    groupnorm=GroupNormAct,
    groupnorm1=GroupNorm1Act,
    layernorm=LayerNormAct,
    layernorm2d=LayerNormAct2d,
    layernormfp32=LayerNormActFp32,
    layernorm2dfp32=LayerNormAct2dFp32,
    evonormb0=EvoNorm2dB0,
    evonormb1=EvoNorm2dB1,
    evonormb2=EvoNorm2dB2,
    evonorms0=EvoNorm2dS0,
    evonorms0a=EvoNorm2dS0a,
    evonorms1=EvoNorm2dS1,
    evonorms1a=EvoNorm2dS1a,
    evonorms2=EvoNorm2dS2,
    evonorms2a=EvoNorm2dS2a,
    frn=FilterResponseNormAct2d,
    frntlu=FilterResponseNormTlu2d,
    inplaceabn=InplaceAbn,
    iabn=InplaceAbn,
    rmsnorm=RmsNormAct,
    rmsnorm2d=RmsNormAct2d,
    rmsnormfp32=RmsNormActFp32,
    rmsnorm2dfp32=RmsNormAct2dFp32,
)
_NORM_ACT_TYPES = set(m for n, m in _NORM_ACT_MAP.items())
_NORM_TO_NORM_ACT_MAP = dict(
    batchnorm=BatchNormAct2d,
    batchnorm2d=BatchNormAct2d,
    groupnorm=GroupNormAct,
    groupnorm1=GroupNorm1Act,
    layernorm=LayerNormAct,
    layernorm2d=LayerNormAct2d,
    layernormfp32=LayerNormActFp32,
    layernorm2dfp32=LayerNormAct2dFp32,
    rmsnorm=RmsNormAct,
    rmsnorm2d=RmsNormAct2d,
    rmsnormfp32=RmsNormActFp32,
    rmsnorm2dfp32=RmsNormAct2dFp32,
)
_NORM_ACT_REQUIRES_ARG = set([
    BatchNormAct2d, GroupNormAct, GroupNorm1Act, LayerNormAct, LayerNormAct2d,
    LayerNormActFp32, LayerNormAct2dFp32, FilterResponseNormAct2d, InplaceAbn,
    RmsNormAct, RmsNormAct2d, RmsNormActFp32, RmsNormAct2dFp32])


def create_norm_act_layer(layer_name, num_features, act_layer=None,
                          apply_act=True, jit=False, **kwargs):
    layer = get_norm_act_layer(layer_name, act_layer=act_layer)
    layer_instance = layer(num_features, apply_act=apply_act, **kwargs)
    if jit:
        raise NotImplementedError("torch.jit.script")
    return layer_instance


def get_norm_act_layer(norm_layer, act_layer=None):
    if norm_layer is None:
        return None
    assert isinstance(norm_layer, (type, str, types.FunctionType,
                                   functools.partial))
    assert act_layer is None or isinstance(act_layer, (
        type, str, types.FunctionType, functools.partial))
    norm_act_kwargs = {}
    if isinstance(norm_layer, functools.partial):
        norm_act_kwargs.update(norm_layer.keywords)
        norm_layer = norm_layer.func
    if isinstance(norm_layer, str):
        if not norm_layer:
            return None
        layer_name = norm_layer.replace('_', '').lower().split('-')[0]
        norm_act_layer = _NORM_ACT_MAP[layer_name]
    elif norm_layer in _NORM_ACT_TYPES:
        norm_act_layer = norm_layer
    elif isinstance(norm_layer, types.FunctionType):
        norm_act_layer = norm_layer
    else:
        type_name = norm_layer.__name__.lower()
        norm_act_layer = _NORM_TO_NORM_ACT_MAP.get(type_name, None)
        assert norm_act_layer is not None, \
            "No equivalent norm_act layer for %s" % type_name
    if norm_act_layer in _NORM_ACT_REQUIRES_ARG:
        norm_act_kwargs.setdefault('act_layer', act_layer)
    if norm_act_kwargs:
        norm_act_layer = functools.partial(norm_act_layer, **norm_act_kwargs)
    return norm_act_layer
