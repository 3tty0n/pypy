"""Initialisers are no-ops: every tensor a model reads comes from the
checkpoint through load_state_dict."""


def _noop(tensor, *args, **kwargs):
    return tensor


kaiming_normal_ = kaiming_uniform_ = constant_ = normal_ = uniform_ = _noop
zeros_ = ones_ = xavier_uniform_ = xavier_normal_ = trunc_normal_ = _noop
orthogonal_ = _noop
