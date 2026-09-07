# metatensor

Tensors whose ops are JIT virtuals: a traced chain of ops becomes one fused
Triton kernel (`rpython/jit/metainterp/optimizeopt/metatensor.py`).

Modules, bottom-up, no cycles:

| module | role |
|---|---|
| `core.py` | dtypes, ll structs, opcode table, shape helpers, size promotion policy |
| `device.py` | bindings to `cuda.c`, `DeviceBuffer` ownership, host/device copies, GC trigger, profiler |
| `ttir.py` | mode analysis and all TTIR emission: flat, row, gather |
| `kernels.py` | kernel DAG build and cache, `compile_ttir` via `triton_compile.py` |
| `devops.py` | cuBLAS/gather library ops with their CPU fallbacks, constant caches |
| `runtime.py` | `eval_op`, fused launches |
| `ops.py` | `@jit.oopspec` primitives and the traced wrappers (`add`, `matmul`, `sum`, ...) |
| `nn.py` | autograd `Tensor`, layers, `sgd_step` |

`cuda.c` is the CUDA driver shim; `triton_compile.py` runs under `RTENSOR_PYTHON`.

Tests: `RTENSOR_PYTHON=... python2 pytest.py rpython/metatensor/test/` (`RTENSOR_CPU=1` for no GPU).
