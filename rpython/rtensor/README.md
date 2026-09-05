# rtensor

Rank-N float tensors with GPU kernels fused by the JIT (see
`rpython/jit/metainterp/optimizeopt/vtensor.py`).

## Modules

Import order is strictly bottom-up; there are no cycles.

    core  <  device  <  kernels  <  runtime  <  ops  <  nn

| module | responsibility |
| --- | --- |
| `core.py` | dtype constants, ll types (`TENSOR`, `KERNEL`, `NODE`, `SHAPEARRAY`, `HOSTARRAY`, `TENSORARRAY`), the opcode table (`ADD`..`EQMASK`, `ARITY`, `HAS_PARAM`, param slots), broadcast and gather codes, host-side constructors (`new_tensor`, `zeros`, `from_list`), the size/dtype promotion policy and `config` |
| `device.py` | `ExternalCompilationInfo` for `cuda.c`, every `rffi.llexternal` binding, `DeviceBuffer`/`attach_buffer`/`device_tensor`, `host()`/`dev()` transfers, `gpu_enabled`, GC budget (`collect_if_needed`), the per-kernel profiler, `sync_device`, `launch_count` |
| `kernels.py` | `KERNEL` construction (`new_kernel`, `set_node`, `finish_kernel`, `add_output`, `build_kernel`), the kernel cache (`kernel_key`, `compile_or_reuse`), the single-op kernels and `init_device`/`init_dtype`, TTIR emission (flat, row-tiled, gather) and `compile_gpu` (writes `.ttir`, runs `triton_compile.py`, loads the PTX) |
| `runtime.py` | execution: `eval_op` (single-op GPU kernel, else `eval_op_cpu`), `launch`/`launch_gpu` for fused kernels, `gather_gpu`, the cuBLAS/gather device ops (`tensor_matmul`, `tensor_bmm`, `im2col`, `col2chw`, `maxpool2`, `head_split`, `head_merge`, `tensor_assign`) with their CPU fallbacks, the constant caches (`scalar`, `ones`) and `reset_device` |
| `ops.py` | the `@jit.oopspec` primitives (`tensor_add` .. `tensor_dtype`) and the traced library wrappers (`add`, `mul`, `sub`, `div`, `relu`, `exp`, `sqrt`, `sum`, `max`, `matmul`, `view`, `reshape`, `ones_like`, `astype`, `assign`, `item`, `size`, `cols_of`, `bcast`) |
| `nn.py` | autograd `Tensor`, layers and `sgd_step` on top of `ops` |

`cuda.c` is the CUDA driver shim, `triton_compile.py` the python3 helper that
turns `.ttir` into `.ptx` (its interpreter comes from `RTENSOR_PYTHON`).

Tests: `python2 pytest.py rpython/rtensor/test/test_tensor.py`.
