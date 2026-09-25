"""timm.layers.create_attn.  Upstream imports every attention module this
factory can name; get_attn is upstream's, over the modules that are ported
(squeeze-excite, split attention).  Naming any other raises
NotImplementedError when it is built."""
from functools import partial

import torch

from timm.layers.squeeze_excite import SEModule, EffectiveSEModule
from timm.layers.split_attn import SplitAttn


def _missing(name):
    def build(*args, **kwargs):
        raise NotImplementedError("attention module %s" % name)
    return build


EcaModule = _missing("EcaModule")
CecaModule = _missing("CecaModule")
GatherExcite = _missing("GatherExcite")
GlobalContext = _missing("GlobalContext")
CbamModule = _missing("CbamModule")
LightCbamModule = _missing("LightCbamModule")
CoordAttn = _missing("CoordAttn")
SimpleCoordAttn = _missing("SimpleCoordAttn")
EfficientLocalAttn = _missing("EfficientLocalAttn")
StripAttn = _missing("StripAttn")
SelectiveKernel = _missing("SelectiveKernel")
LambdaLayer = _missing("LambdaLayer")
BottleneckAttn = _missing("BottleneckAttn")
HaloAttn = _missing("HaloAttn")
NonLocalAttn = _missing("NonLocalAttn")
BatNonLocalAttn = _missing("BatNonLocalAttn")


def get_attn(attn_type):
    if isinstance(attn_type, torch.nn.Module):
        return attn_type
    module_cls = None
    if attn_type:
        if isinstance(attn_type, str):
            attn_type = attn_type.lower()
            if attn_type == 'se':
                module_cls = SEModule
            elif attn_type == 'ese':
                module_cls = EffectiveSEModule
            elif attn_type == 'eca':
                module_cls = EcaModule
            elif attn_type == 'ecam':
                module_cls = partial(EcaModule, use_mlp=True)
            elif attn_type == 'ceca':
                module_cls = CecaModule
            elif attn_type == 'ge':
                module_cls = GatherExcite
            elif attn_type == 'gc':
                module_cls = GlobalContext
            elif attn_type == 'gca':
                module_cls = partial(GlobalContext, fuse_add=True,
                                     fuse_scale=False)
            elif attn_type == 'cbam':
                module_cls = CbamModule
            elif attn_type == 'lcbam':
                module_cls = LightCbamModule
            elif attn_type == 'coord':
                module_cls = CoordAttn
            elif attn_type == 'scoord':
                module_cls = SimpleCoordAttn
            elif attn_type == 'ela':
                module_cls = EfficientLocalAttn
            elif attn_type == 'strip':
                module_cls = StripAttn
            elif attn_type == 'sk':
                module_cls = SelectiveKernel
            elif attn_type == 'splat':
                module_cls = SplitAttn
            elif attn_type == 'lambda':
                return LambdaLayer
            elif attn_type == 'bottleneck':
                return BottleneckAttn
            elif attn_type == 'halo':
                return HaloAttn
            elif attn_type == 'nl':
                module_cls = NonLocalAttn
            elif attn_type == 'bat':
                module_cls = BatNonLocalAttn
            else:
                assert False, "Invalid attn module (%s)" % attn_type
        elif isinstance(attn_type, bool):
            if attn_type:
                module_cls = SEModule
        else:
            module_cls = attn_type
    return module_cls


def create_attn(attn_type, channels, **kwargs):
    module_cls = get_attn(attn_type)
    if module_cls is not None:
        return module_cls(channels, **kwargs)
    return None
