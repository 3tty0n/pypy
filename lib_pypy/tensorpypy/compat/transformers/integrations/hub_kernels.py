def use_kernel_forward_from_hub(layer_name):
    """No kernel hub: the module's own forward runs, as it does upstream when
    the `kernels` package is absent."""
    return lambda cls: cls
