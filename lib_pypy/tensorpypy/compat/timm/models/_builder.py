"""timm.models._builder.build_model_with_cfg on the path create_model takes:
resolve the pretrained config, default the model kwargs from it, build the
model.  Pretrained weights are not loaded here: the harness loads the
exported state_dict (the weights upstream's load_pretrained put in the
reference model) through load_state_dict.  Feature extraction wrappers and
pruned variants raise NotImplementedError."""
from timm.models._registry import get_pretrained_cfg, PretrainedCfg


def pretrained_cfg_for_features(pretrained_cfg):
    from copy import deepcopy
    pretrained_cfg = deepcopy(pretrained_cfg)
    for tr in ('num_classes', 'classifier', 'global_pool'):
        pretrained_cfg.pop(tr, None)
    return pretrained_cfg


def _filter_kwargs(kwargs, names):
    if not kwargs or not names:
        return
    for n in names:
        kwargs.pop(n, None)


def _update_default_model_kwargs(pretrained_cfg, kwargs, kwargs_filter):
    default_kwarg_names = ('num_classes', 'global_pool', 'in_chans')
    if pretrained_cfg.get('fixed_input_size', False):
        default_kwarg_names += ('img_size',)
    for n in default_kwarg_names:
        if n == 'img_size':
            input_size = pretrained_cfg.get('input_size', None)
            if input_size is not None:
                assert len(input_size) == 3
                kwargs.setdefault(n, input_size[-2:])
        elif n == 'in_chans':
            input_size = pretrained_cfg.get('input_size', None)
            if input_size is not None:
                assert len(input_size) == 3
                kwargs.setdefault(n, input_size[0])
        elif n == 'num_classes':
            default_val = pretrained_cfg.get(n, None)
            if default_val is not None and default_val >= 0:
                kwargs.setdefault(n, pretrained_cfg[n])
        else:
            default_val = pretrained_cfg.get(n, None)
            if default_val is not None:
                kwargs.setdefault(n, pretrained_cfg[n])
    _filter_kwargs(kwargs, names=kwargs_filter)


def resolve_pretrained_cfg(variant, pretrained_cfg=None,
                           pretrained_cfg_overlay=None):
    if isinstance(pretrained_cfg, dict):
        raise NotImplementedError("an explicit pretrained_cfg dict")
    if isinstance(pretrained_cfg, PretrainedCfg):
        cfg = pretrained_cfg
    else:
        name = variant
        if pretrained_cfg:
            name = '.'.join([variant, pretrained_cfg])
        cfg = get_pretrained_cfg(name)
    if pretrained_cfg_overlay:
        cfg = PretrainedCfg(**dict(cfg.to_dict(remove_null=False),
                                   **pretrained_cfg_overlay))
    return cfg


def build_model_with_cfg(model_cls, variant, pretrained, pretrained_cfg=None,
                         pretrained_cfg_overlay=None, model_cfg=None,
                         feature_cfg=None, pretrained_strict=True,
                         pretrained_filter_fn=None, cache_dir=None,
                         kwargs_filter=None, **kwargs):
    if kwargs.pop('pruned', False):
        raise NotImplementedError("pruned models")
    pretrained_cfg = resolve_pretrained_cfg(
        variant, pretrained_cfg=pretrained_cfg,
        pretrained_cfg_overlay=pretrained_cfg_overlay).to_dict()
    _update_default_model_kwargs(pretrained_cfg, kwargs, kwargs_filter)
    if kwargs.pop('features_only', False):
        raise NotImplementedError("features_only models")
    if model_cfg is None:
        model = model_cls(**kwargs)
    else:
        model = model_cls(cfg=model_cfg, **kwargs)
    model.pretrained_cfg = pretrained_cfg
    model.default_cfg = model.pretrained_cfg
    return model
