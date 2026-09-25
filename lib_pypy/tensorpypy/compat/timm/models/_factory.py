"""timm.models._factory.create_model for a registry name: prune the None
kwargs, split off the pretrained tag, call the entrypoint under the layer
config, as upstream does.  Hugging Face Hub and local-dir sources and
checkpoint_path raise NotImplementedError."""
from timm.layers import set_layer_config
from timm.models._registry import (is_model, model_entrypoint,
                                   split_model_name_tag)


def create_model(model_name, pretrained=False, pretrained_cfg=None,
                 pretrained_cfg_overlay=None, checkpoint_path=None,
                 cache_dir=None, scriptable=None, exportable=None,
                 no_jit=None, **kwargs):
    kwargs = dict((k, v) for k, v in kwargs.items() if v is not None)
    if ':' in model_name:
        raise NotImplementedError("model source %s" % model_name)
    model_name, pretrained_tag = split_model_name_tag(model_name)
    if pretrained_tag and not pretrained_cfg:
        pretrained_cfg = pretrained_tag
    if not is_model(model_name):
        raise RuntimeError('Unknown model (%s)' % model_name)
    create_fn = model_entrypoint(model_name)
    with set_layer_config(scriptable=scriptable, exportable=exportable,
                          no_jit=no_jit):
        model = create_fn(pretrained=pretrained,
                          pretrained_cfg=pretrained_cfg,
                          pretrained_cfg_overlay=pretrained_cfg_overlay,
                          cache_dir=cache_dir, **kwargs)
    if checkpoint_path:
        raise NotImplementedError("checkpoint_path")
    return model
