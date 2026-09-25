"""timm.layers.weight_init: initialisers are no-ops, as torch.nn.init's are
here: every tensor a model reads comes from the checkpoint through
load_state_dict."""


def _noop(tensor, *args, **kwargs):
    return tensor


trunc_normal_ = trunc_normal_tf_ = variance_scaling_ = lecun_normal_ = _noop


def init_weight_vit(module, name, init_bias=0.02, head_bias=0.,
                    classifier_name='head'):
    pass


def init_weight_jax(module, name, head_bias=0., classifier_name='head'):
    pass
