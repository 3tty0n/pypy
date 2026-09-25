"""Weight initialisers: every tensor comes from the checkpoint, so the
_init_weights methods that call these never run in an eval forward."""


def _noop(tensor, *args, **kwargs):
    return tensor


normal_ = zeros_ = ones_ = constant_ = copy_ = uniform_ = _noop
trunc_normal_ = xavier_uniform_ = kaiming_normal_ = _noop
