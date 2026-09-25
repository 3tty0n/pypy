"""The kernel hub swaps in optimised kernels when the `kernels` package is
installed; without it, as in the reference run, upstream's own code runs."""


def use_kernel_forward_from_hub(layer_name):
    return lambda cls: cls


def use_kernel_func_from_hub(func_name):
    return lambda fn: fn


def use_kernelized_func(*args, **kwargs):
    return lambda obj: obj
