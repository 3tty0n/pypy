# Virtual-tensor JIT benchmark

Measures one loop iteration of `h <- relu(h * b + b)` repeated `K` times over
rank-1 float64 tensors of `N` elements, under the PyPy meta-tracing JIT with
tensor ops kept virtual and fused into one Triton kernel (`rpython/rlib/rtensor.py`,
`rpython/jit/metainterp/optimizeopt/vtensor.py`).  `h` is loop-carried so the
chain cannot be hoisted out of the loop.  The same computation in PyTorch
(eager and `torch.compile`) is the baseline.

## Prerequisites

- NVIDIA GPU with driver >= 13.x and `libcuda.so`; CUDA headers at
  `$CUDA_HOME/include` (default `/usr/local/cuda`).
- Python 3 with Triton for kernel compilation, and PyTorch for the baseline:

      python3 -m venv ~/.venvs/triton
      ~/.venvs/triton/bin/pip install triton torch --index-url https://download.pytorch.org/whl/cu130
      export RTENSOR_PYTHON=~/.venvs/triton/bin/python

- Python 2 (PyPy 2 recommended) to run the RPython toolchain.

Environment variables read at run time:

| variable | default | meaning |
|---|---|---|
| `RTENSOR_PYTHON` | `python3` | interpreter with Triton, used to compile `.ttir` to PTX |
| `RTENSOR_CC` | `86` | compute capability passed to Triton |
| `RTENSOR_BLOCK` | `4096` | elements per Triton program |
| `RTENSOR_WARPS` | `8` | warps per program |
| `RTENSOR_CPU` | unset | set to force the CPU evaluator (no GPU) |
| `RTENSOR_BUDGET_MB` | `64` | live device memory above which a launch first runs a GC to recycle buffers |
| `CUDA_HOME` | `/usr/local/cuda` | CUDA include directory |
| `TMPDIR` | `/tmp` | where `.ttir`, `.ptx`, `.meta` files are written |

## Build

    benchmark/build.sh            # writes benchmark/rtensor-bench (about 10 minutes)

The translation compiles the whole JIT; use `MAKE_JOBS=N` to limit the C
build's parallelism on memory-constrained machines.  The generated C can be
rebuilt alone with `make` in the `usession-*/testing_1` directory printed by
the translator.

## Run one configuration

    benchmark/rtensor-bench MODE VARIANT K N ITERS

| argument | values |
|---|---|
| `MODE` | `fused` (tensor ops fused into one kernel), `eager` (the `tensor` optimization disabled, one GPU kernel per op, JIT otherwise on), `nojit` (interpreter, one GPU kernel per op) |
| `VARIANT` | `0` plain chain; `1` a branch `if i % 7 == 0: h = h + b` inside the fused region (a guard, no force); `2` the same branch after an explicit force of `h`, modelling a graph break; `3` a data-dependent branch `if sum(h).item() > 0` (a true force in every system); `4` the chain continued inside `try/except` with an exception raised every 5th iteration; `5` a host-side write to `/dev/null` every 50th iteration between tensor ops |
| `K` | chain length |
| `N` | tensor size |
| `ITERS` | timed iterations |

Output is one line:

    mode variant k n iters warm_s steady_us kernels acc compiled_in_timed launches_per_iter

`warm_s` is the first warmup run (tracing plus Triton compilation), `steady_us`
the per-iteration time of the timed run after two more warmup runs, `kernels`
the number of kernels compiled in the process, `acc` the final checksum
(`sum(h)`), and `compiled_in_timed` the number of kernels compiled inside the
timed run, which must be `0` for the number to be meaningful, and
`launches_per_iter` the number of GPU kernel launches per timed iteration.

The PyTorch baseline takes the same arguments with `MODE` in `eager`, `compile`
and appends two columns, the graph count and graph-break count reported by
`torch._dynamo.explain` for the step function:

    $RTENSOR_PYTHON benchmark/torch_bench.py compile 0 4 1000000 200

## Run the full grid

    RTENSOR_PYTHON=~/.venvs/triton/bin/python benchmark/run_experiments.sh

Writes `rtensor_chain.tsv`, `rtensor_branch.tsv`, `torch.tsv` and `summary.txt`
under `benchmark/results/<date>-<host>/`.  `REPS` (default 3) and `ITERS`
(default 200) can be overridden.  `summarize.py` prints the median per
configuration:

    python3 benchmark/summarize.py benchmark/results/<dir>/*.tsv

## Reading the results

`results/2026-09-05-rtx3090-v2/` holds a run on an RTX 3090 (compute
capability 8.6, CUDA 13.1, Triton 3.8.0, PyTorch 2.14, cu130), medians of 3
repetitions, 200 iterations, per-iteration microseconds.  Checksums agree
across all modes and with PyTorch in every configuration.

| configuration | fused | eager (ours) | torch.compile | torch eager |
|---|---|---|---|---|
| K=1, N=1e6 | 40.5 | 113.3 | 42.7 | 78.6 |
| K=4, N=1e6 | 51.6 | 452.5 | 69.0 | 312.4 |
| K=8, N=1e6 | 82.9 | 903.2 | 133.4 | 622.9 |
| K=4, N=1e4 | 15.0 | 138.3 | 41.9 | 55.2 |
| K=4, N=1e5 | 15.0 | 70.3 | 43.3 | 56.1 |
| K=4, N=1e7 | 422.7 | 4125.5 | 646.0 | 3045.4 |

Disabling fusion in the same runtime costs 3x to 11x, growing with `K`.  For
small tensors the fused loop runs in about 15 us against 42 us for
`torch.compile`, because the loop control and the launch are JIT-compiled
machine code.  The fused kernel itself is about 1.5x faster than Inductor's at
large `N`, which is a property of the Triton output, not of the frontend.

Graph-break set (K=4, N=1e6).  `launches` is the number of GPU kernel launches
per iteration in our fused mode; `graphs`/`breaks` are what `torch._dynamo.explain`
reports for the same step function:

| variant | fused (us) | launches | torch.compile (us) | graphs / breaks | torch eager (us) |
|---|---|---|---|---|---|
| 0 plain chain | 51.6 | 1.00 | 69.0 | 1 / 0 | 312.4 |
| 1 branch on the loop counter | 53.3 | 1.00 | 308.2 | 1 / 0 (recompiles, then eager) | 317.3 |
| 2 same after an explicit force | 57.5 | 1.15 | 308.2 | 1 / 0 | 317.3 |
| 3 branch on `sum(h).item()` | 111.3 | 2.00 | 169.9 | 2 / 1 | 376.7 |
| 4 `try/except` around the chain | 64.3 | 1.00 | 333.0 | 1 / 0 (recompiles, then eager) | 343.2 |
| 5 host write to `/dev/null` | 68.7 | 1.12 | 333.6 | 2 / 1 | 343.0 |

Variant 3 is a real force in every system (the value is needed on the host),
so both produce two kernels.  Variants 1 and 4 depend on the loop counter;
Dynamo specialises on the integer, hits its recompilation limit and falls back
to eager, while the meta-tracer keeps the branch as a guard inside one fused
kernel.  Variant 5 breaks the Dynamo graph at the `print`; here the write is
an ordinary residual call that does not touch the virtual tensors.  The 1.12
launches per iteration in variant 5 and the 1.15 in variant 2 are the eager
re-evaluation that the blackhole performs when a guard fails before its
bridge is compiled; they converge to 1.0 over longer runs.

## Pitfalls that produce wrong numbers

- The JIT's default loop threshold is 1039 iterations; the harness sets
  `threshold=3` so 200-iteration runs are compiled.
- Tensor ops are elidable, so a chain over loop-invariant inputs is hoisted
  out of the loop.  The benchmark carries `h` across iterations for that reason.
- Device buffers are owned by finalizable objects and recycled through a
  free list; a launch first runs a GC when live device memory exceeds
  `RTENSOR_BUDGET_MB` and no buffer of the requested size is free.  A budget
  larger than the working set means fresh `cuMemAlloc` calls every iteration,
  which is several times slower than the kernel for medium sizes.

To verify what a run actually launches, an `LD_PRELOAD` shim that counts
`cuLaunchKernel` calls is the quickest check; launches must grow by exactly
one per iteration in `fused` mode.
