"""timm.models._registry.  register_model records each entrypoint as
upstream does.  Upstream's pretrained configs are each model file's
default_cfgs, built by _cfg(**kwargs) dict displays the porter has no
rewrite for, so they are not ported; _PRETRAINED_CFGS is upstream's
get_pretrained_cfg(name).to_dict() (timm 1.0.22) for the tags the ported
models are built with, restricted to the fields build_model_with_cfg reads,
and _DEFAULT_TAGS the tag an untagged name resolves to.  A name outside the
table raises NotImplementedError rather than fall back to upstream's
default config.  _DEPRECATED is the part of the maps upstream's model files
pass to register_model_deprecations that the suites name."""
import sys

_model_entrypoints = {}
_model_to_module = {}

_PRETRAINED_CFGS = {
    'deit_tiny_patch16_224.fb_in1k': {'num_classes': 1000, 'input_size': (3, 224, 224), 'fixed_input_size': True},
    'deit_base_distilled_patch16_224.fb_in1k': {'num_classes': 1000, 'input_size': (3, 224, 224), 'fixed_input_size': True},
    'vit_base_patch16_siglip_256.v2_webli': {'num_classes': 0, 'input_size': (3, 256, 256), 'fixed_input_size': True},
    'vit_base_patch14_dinov2.lvd142m': {'num_classes': 0, 'input_size': (3, 518, 518), 'fixed_input_size': True},
    'beit_base_patch16_224.in22k_ft_in22k_in1k': {'num_classes': 1000, 'input_size': (3, 224, 224), 'fixed_input_size': True},
    'repvgg_a2.rvgg_in1k': {'num_classes': 1000, 'input_size': (3, 224, 224), 'fixed_input_size': False},
    'mobilenetv2_100.ra_in1k': {'num_classes': 1000, 'input_size': (3, 224, 224), 'fixed_input_size': False},
    'mobilenetv3_large_100.ra_in1k': {'num_classes': 1000, 'input_size': (3, 224, 224), 'fixed_input_size': False},
    'tf_efficientnet_b0.ns_jft_in1k': {'num_classes': 1000, 'input_size': (3, 224, 224), 'fixed_input_size': False},
    'ghostnet_100.in1k': {'num_classes': 1000, 'input_size': (3, 224, 224), 'fixed_input_size': False},
    'inception_v3.tv_in1k': {'num_classes': 1000, 'input_size': (3, 299, 299), 'fixed_input_size': False},
    'inception_v3.tf_adv_in1k': {'num_classes': 1000, 'input_size': (3, 299, 299), 'fixed_input_size': False},
    'convnextv2_nano.fcmae_ft_in22k_in1k': {'num_classes': 1000, 'input_size': (3, 224, 224), 'fixed_input_size': False},
    'visformer_small.in1k': {'num_classes': 1000, 'input_size': (3, 224, 224), 'fixed_input_size': True},
    'dm_nfnet_f0.dm_in1k': {'num_classes': 1000, 'input_size': (3, 192, 192), 'fixed_input_size': False},
    'nfnet_l0.ra2_in1k': {'num_classes': 1000, 'input_size': (3, 224, 224), 'fixed_input_size': False},
    'mobilevit_s.cvnets_in1k': {'num_classes': 1000, 'input_size': (3, 256, 256), 'fixed_input_size': False},
    'swin_base_patch4_window7_224.ms_in22k_ft_in1k': {'num_classes': 1000, 'input_size': (3, 224, 224), 'fixed_input_size': True},
    'vit_small_patch16_224.augreg_in21k_ft_in1k': {'num_classes': 1000, 'input_size': (3, 224, 224), 'fixed_input_size': True},
    'vit_giant_patch14_224.untrained': {'num_classes': 1000, 'input_size': (3, 224, 224), 'fixed_input_size': True},
    'regnety_120.sw_in12k_ft_in1k': {'num_classes': 1000, 'input_size': (3, 224, 224), 'fixed_input_size': False},
    'resnest14d.gluon_in1k': {'num_classes': 1000, 'input_size': (3, 224, 224), 'fixed_input_size': False},
    'vovnet39a.untrained': {'num_classes': 1000, 'input_size': (3, 224, 224), 'fixed_input_size': False},
    'efficientnet_b0.ra_in1k': {'num_classes': 1000, 'input_size': (3, 224, 224), 'fixed_input_size': False},
}

_DEFAULT_TAGS = {
    'deit_tiny_patch16_224': 'fb_in1k',
    'deit_base_distilled_patch16_224': 'fb_in1k',
    'vit_base_patch16_siglip_256': 'v2_webli',
    'vit_base_patch14_dinov2': 'lvd142m',
    'beit_base_patch16_224': 'in22k_ft_in22k_in1k',
    'repvgg_a2': 'rvgg_in1k',
    'mobilenetv2_100': 'ra_in1k',
    'mobilenetv3_large_100': 'ra_in1k',
    'tf_efficientnet_b0': 'ns_jft_in1k',
    'ghostnet_100': 'in1k',
    'inception_v3': 'tv_in1k',
    'convnextv2_nano': 'fcmae_ft_in22k_in1k',
    'visformer_small': 'in1k',
    'dm_nfnet_f0': 'dm_in1k',
    'nfnet_l0': 'ra2_in1k',
    'mobilevit_s': 'cvnets_in1k',
    'swin_base_patch4_window7_224': 'ms_in22k_ft_in1k',
    'vit_small_patch16_224': 'augreg_in21k_ft_in1k',
    'vit_giant_patch14_224': 'untrained',
    'regnety_120': 'sw_in12k_ft_in1k',
    'resnest14d': 'gluon_in1k',
    'vovnet39a': 'untrained',
    'efficientnet_b0': 'ra_in1k',
}

_DEPRECATED = {'adv_inception_v3': 'inception_v3.tf_adv_in1k'}


def split_model_name_tag(model_name, no_tag=''):
    parts = model_name.split('.', 1)
    return parts[0], parts[1] if len(parts) > 1 else no_tag


def get_arch_name(model_name):
    return split_model_name_tag(model_name)[0]


def generate_default_cfgs(cfgs):
    raise NotImplementedError("default_cfgs (see _PRETRAINED_CFGS)")


def register_model(fn):
    mod = sys.modules[fn.__module__]
    model_name = fn.__name__
    if hasattr(mod, '__all__'):
        mod.__all__.append(model_name)
    else:
        mod.__all__ = [model_name]
    _model_entrypoints[model_name] = fn
    _model_to_module[model_name] = fn.__module__.split('.')[-1]
    return fn


def _deprecated_model_shim(deprecated_name, current_fn, current_tag=''):
    def _fn(pretrained=False, **kwargs):
        pretrained_cfg = kwargs.pop('pretrained_cfg', None)
        return current_fn(pretrained=pretrained,
                          pretrained_cfg=pretrained_cfg or current_tag,
                          **kwargs)
    return _fn


def register_model_deprecations(module_name, deprecation_map):
    raise NotImplementedError("see _DEPRECATED")


def is_model(model_name):
    arch = get_arch_name(model_name)
    return arch in _model_entrypoints or arch in _DEPRECATED


def model_entrypoint(model_name, module_filter=None):
    arch = get_arch_name(model_name)
    if arch not in _model_entrypoints and arch in _DEPRECATED:
        current, tag = split_model_name_tag(_DEPRECATED[arch])
        return _deprecated_model_shim(arch, _model_entrypoints[current], tag)
    return _model_entrypoints[arch]


def get_pretrained_cfg(model_name, allow_unregistered=True):
    arch, tag = split_model_name_tag(model_name)
    if arch not in _DEFAULT_TAGS:
        raise NotImplementedError("no pretrained cfg transcribed for %s"
                                  % model_name)
    tag = tag or _DEFAULT_TAGS[arch]
    key = arch + '.' + tag
    if key not in _PRETRAINED_CFGS:
        raise RuntimeError('Invalid pretrained tag (%s) for %s.'
                           % (tag, arch))
    cfg = PretrainedCfg(**_PRETRAINED_CFGS[key])
    cfg.architecture = arch
    cfg.tag = tag
    return cfg


class PretrainedCfg(object):
    """Upstream's PretrainedCfg dataclass, holding the transcribed fields:
    read as attributes, to_dict() for build_model_with_cfg."""

    def __init__(self, **kw):
        self.__dict__.update(kw)

    def to_dict(self, remove_source=False, remove_null=True):
        return dict((k, v) for k, v in self.__dict__.items()
                    if v is not None or not remove_null)
