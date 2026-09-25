"""timm.layers.config: the layer-config flags.  Upstream decides the fused
attention and reentrant-checkpoint defaults in top-level if/else blocks
reading the environment, which the porter does not carry; they are the same
statements here."""
import os

import torch

_NO_JIT = False
_NO_ACTIVATION_JIT = False
_EXPORTABLE = False
_SCRIPTABLE = False

_HAS_FUSED_ATTN = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
if 'TIMM_FUSED_ATTN' in os.environ:
    _USE_FUSED_ATTN = int(os.environ['TIMM_FUSED_ATTN'])
else:
    _USE_FUSED_ATTN = 1

if 'TIMM_REENTRANT_CKPT' in os.environ:
    _USE_REENTRANT_CKPT = bool(os.environ['TIMM_REENTRANT_CKPT'])
else:
    _USE_REENTRANT_CKPT = False


def is_no_jit():
    return _NO_JIT


def is_exportable():
    return _EXPORTABLE


def is_scriptable():
    return _SCRIPTABLE


class _Setter(object):
    def __enter__(self):
        pass

    def __exit__(self, *args):
        self.restore()
        return False


class set_no_jit(_Setter):
    def __init__(self, mode):
        global _NO_JIT
        self.prev = _NO_JIT
        _NO_JIT = mode

    def restore(self):
        global _NO_JIT
        _NO_JIT = self.prev


class set_exportable(_Setter):
    def __init__(self, mode):
        global _EXPORTABLE
        self.prev = _EXPORTABLE
        _EXPORTABLE = mode

    def restore(self):
        global _EXPORTABLE
        _EXPORTABLE = self.prev


class set_scriptable(_Setter):
    def __init__(self, mode):
        global _SCRIPTABLE
        self.prev = _SCRIPTABLE
        _SCRIPTABLE = mode

    def restore(self):
        global _SCRIPTABLE
        _SCRIPTABLE = self.prev


class set_layer_config(_Setter):
    def __init__(self, scriptable=None, exportable=None, no_jit=None,
                 no_activation_jit=None):
        global _SCRIPTABLE, _EXPORTABLE, _NO_JIT, _NO_ACTIVATION_JIT
        self.prev = _SCRIPTABLE, _EXPORTABLE, _NO_JIT, _NO_ACTIVATION_JIT
        if scriptable is not None:
            _SCRIPTABLE = scriptable
        if exportable is not None:
            _EXPORTABLE = exportable
        if no_jit is not None:
            _NO_JIT = no_jit
        if no_activation_jit is not None:
            _NO_ACTIVATION_JIT = no_activation_jit

    def restore(self):
        global _SCRIPTABLE, _EXPORTABLE, _NO_JIT, _NO_ACTIVATION_JIT
        _SCRIPTABLE, _EXPORTABLE, _NO_JIT, _NO_ACTIVATION_JIT = self.prev


def use_fused_attn(experimental=False):
    if not _HAS_FUSED_ATTN or _EXPORTABLE:
        return False
    if experimental:
        return _USE_FUSED_ATTN > 1
    return _USE_FUSED_ATTN > 0


def set_fused_attn(enable=True, experimental=False):
    global _USE_FUSED_ATTN
    if experimental and enable:
        _USE_FUSED_ATTN = 2
    elif enable:
        _USE_FUSED_ATTN = 1
    else:
        _USE_FUSED_ATTN = 0


def use_reentrant_ckpt():
    return _USE_REENTRANT_CKPT


def set_reentrant_ckpt(enable=True):
    global _USE_REENTRANT_CKPT
    _USE_REENTRANT_CKPT = enable
