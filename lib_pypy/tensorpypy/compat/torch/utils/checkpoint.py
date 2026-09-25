def checkpoint(function, *args, **kwargs):
    """Activation checkpointing trades memory for recomputation in training;
    an eval forward has nothing to recompute."""
    raise NotImplementedError("activation checkpointing")
