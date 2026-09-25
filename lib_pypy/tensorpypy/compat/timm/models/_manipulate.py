"""timm.models._manipulate.  named_apply is upstream's; checkpoint has a
keyword-only parameter after *args, which Python 2 cannot express, and it
and checkpoint_seq are activation checkpointing for training, which an eval
forward never reaches; adapt_input_conv adapts checkpoint weights to another
input channel count at load time, which the export makes unnecessary."""


def named_apply(fn, module, name='', depth_first=True, include_root=False):
    if not depth_first and include_root:
        fn(module=module, name=name)
    for child_name, child_module in module.named_children():
        child_name = '.'.join((name, child_name)) if name else child_name
        named_apply(fn=fn, module=child_module, name=child_name,
                    depth_first=depth_first, include_root=True)
    if depth_first and include_root:
        fn(module=module, name=name)
    return module


def named_modules(module, name='', depth_first=True, include_root=False):
    if not depth_first and include_root:
        yield name, module
    for child_name, child_module in module.named_children():
        child_name = '.'.join((name, child_name)) if name else child_name
        for x in named_modules(module=child_module, name=child_name,
                               depth_first=depth_first, include_root=True):
            yield x
    if depth_first and include_root:
        yield name, module


def flatten_modules(named_modules, depth=1, prefix='',
                    module_types='sequential'):
    import torch.nn as nn
    prefix_is_tuple = isinstance(prefix, tuple)
    if isinstance(module_types, str):
        if module_types == 'container':
            module_types = (nn.Sequential, nn.ModuleList, nn.ModuleDict)
        else:
            module_types = (nn.Sequential,)
    for name, module in named_modules:
        if depth and isinstance(module, module_types):
            for x in flatten_modules(
                    module.named_children(), depth - 1,
                    prefix=(name,) if prefix_is_tuple else name,
                    module_types=module_types):
                yield x
        else:
            if prefix_is_tuple:
                name = prefix + (name,)
                yield name, module
            else:
                if prefix:
                    name = '.'.join([prefix, name])
                yield name, module


def checkpoint(function, *args, **kwargs):
    raise NotImplementedError("activation checkpointing")


def checkpoint_seq(functions, x, every=1, flatten=False, skip_last=False,
                   use_reentrant=None):
    raise NotImplementedError("activation checkpointing")


def adapt_input_conv(in_chans, conv_weight):
    raise NotImplementedError("adapt_input_conv")
