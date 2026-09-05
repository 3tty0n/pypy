# rtensor

Tensors whose ops are JIT virtuals: a traced chain of ops becomes one fused
Triton kernel (`rpython/jit/metainterp/optimizeopt/vtensor.py`).

Modules, bottom-up, no cycles:

| module | role |
|---|---|
| `core.py` | dtypes, ll structs, opcode table, shape helpers, size promotion policy |
| `device.py` | bindings to `cuda.c`, `DeviceBuffer` ownership, host/device copies, GC trigger, profiler |
| `kernels.py` | kernel DAG build and cache, TTIR emission, compile via `triton_compile.py` |
| `runtime.py` | `eval_op`, fused launches, cuBLAS/gather ops with CPU fallbacks, constant caches |
| `ops.py` | `@jit.oopspec` primitives and the traced wrappers (`add`, `matmul`, `sum`, ...) |
| `nn.py` | autograd `Tensor`, layers, `sgd_step` |

`cuda.c` is the CUDA driver shim; `triton_compile.py` runs under `RTENSOR_PYTHON`.

Tests: `RTENSOR_PYTHON=... python2 pytest.py rpython/rtensor/test/` (`RTENSOR_CPU=1` for no GPU).
