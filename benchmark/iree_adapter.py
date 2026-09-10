"""Run a jitted JAX function through IREE: jax.export -> StableHLO -> IREE CUDA.

    exe = compile(fn, *example_args)      # fn is any jax-traceable function
    out = exe(*args)                      # args: jax/numpy arrays or exe outputs
    exe.sync(out)                         # block until out is computed

The same Python function that jax.jit runs under XLA is what IREE compiles,
so the comparison is XLA versus IREE on one StableHLO module, nothing else.
Pytrees are flattened on the way in and out.  compile_ms is the IREE compile
alone; the StableHLO export (a JAX trace) is charged to export_ms.
"""
import os, time

import jax
import numpy as np

_ARCH = os.environ.get("IREE_CUDA_TARGET", "sm_%s" % os.environ.get("RTENSOR_CC", "86"))
_CONFIG = []


def config():
    """One CUDA device for the process, so outputs of one executable can be
    fed to another without a host round trip."""
    if not _CONFIG:
        import iree.runtime as ir
        _CONFIG.append(ir.Config("cuda"))
    return _CONFIG[0]


def available():
    try:
        import iree.compiler, iree.runtime  # noqa: F401
    except ImportError:
        return False
    return True


class Executable(object):
    def __init__(self, fn, example):
        import iree.compiler as ic
        import iree.runtime as ir
        from jax import export
        leaves, self.in_tree = jax.tree_util.tree_flatten(example)

        def flat_fn(*flat):
            return jax.tree_util.tree_leaves(
                fn(*jax.tree_util.tree_unflatten(self.in_tree, flat)))

        t0 = time.perf_counter()
        exported = export.export(jax.jit(flat_fn))(*leaves)
        mlir = exported.mlir_module()
        self.export_ms = (time.perf_counter() - t0) * 1e3
        _, self.out_tree = jax.tree_util.tree_flatten(
            jax.eval_shape(fn, *example))
        t0 = time.perf_counter()
        vmfb = ic.compile_str(mlir, target_backends=["cuda"],
                              input_type="stablehlo",
                              extra_args=["--iree-cuda-target=%s" % _ARCH,
                                          "--iree-input-demote-f64-to-f32=false"])
        self.compile_ms = (time.perf_counter() - t0) * 1e3
        self.config = config()
        ctx = ir.SystemContext(config=self.config)
        ctx.add_vm_module(ir.VmModule.copy_buffer(ctx.instance, vmfb))
        self.main = ctx.modules.jit_flat_fn["main"]
        self.ir = ir

    def to_device(self, a):
        if isinstance(a, self.ir.DeviceArray):
            return a
        return self.ir.asdevicearray(self.config.device, np.asarray(a))

    def __call__(self, *args):
        return self.bind(*args)()

    def bind(self, *args):
        """Move args to the device once; the returned thunk only launches."""
        flat = [self.to_device(a) for a in jax.tree_util.tree_leaves(args)]

        def run():
            out = self.main(*flat)
            if not isinstance(out, (list, tuple)):
                out = [out]
            return jax.tree_util.tree_unflatten(self.out_tree, list(out))
        return run

    def sync(self, out):
        return jax.tree_util.tree_map(
            lambda a: np.asarray(a.to_host()) if hasattr(a, "to_host") else a,
            out)


def compile(fn, *example):
    return Executable(fn, example)
