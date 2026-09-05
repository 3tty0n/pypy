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
| `VARIANT` | `0` plain chain; `1` a branch `if i % 7 == 0: h = h + b` inside the fused region (a guard, no force); `2` the same branch after an explicit force of `h`, modelling a graph break |
| `K` | chain length |
| `N` | tensor size |
| `ITERS` | timed iterations |

Output is one line:

    mode variant k n iters warm_s steady_us kernels acc compiled_in_timed

`warm_s` is the first warmup run (tracing plus Triton compilation), `steady_us`
the per-iteration time of the timed run after two more warmup runs, `kernels`
the number of kernels compiled in the process, `acc` the final checksum
(`sum(h)`), and `compiled_in_timed` the number of kernels compiled inside the
timed run, which must be `0` for the number to be meaningful.

The PyTorch baseline takes the same arguments with `MODE` in `eager`, `compile`:

    $RTENSOR_PYTHON benchmark/torch_bench.py compile 0 4 1000000 200

## Run the full grid

    RTENSOR_PYTHON=~/.venvs/triton/bin/python benchmark/run_experiments.sh

Writes `rtensor_chain.tsv`, `rtensor_branch.tsv`, `torch.tsv` and `summary.txt`
under `benchmark/results/<date>-<host>/`.  `REPS` (default 3) and `ITERS`
(default 200) can be overridden.  `summarize.py` prints the median per
configuration:

    python3 benchmark/summarize.py benchmark/results/<dir>/*.tsv

## Reading the results

`results/2026-09-05-rtx3090/` holds a run on an RTX 3090 (compute capability
8.6, CUDA 13.1, Triton 3.8.0, PyTorch 2.14, cu130), medians of 3 repetitions,
200 iterations, per-iteration microseconds:

| configuration | fused | eager (ours) | torch.compile | torch eager |
|---|---|---|---|---|
| K=4, N=1e6 | 51.5 | 456.8 | 69.0 | 313.2 |
| K=8, N=1e6 | 82.9 | 912.1 | 133.4 | 626.0 |
| K=4, N=1e4 | 14.5 | 62.9 | 41.3 | 54.2 |
| K=4, N=1e7 | 423.3 | 4125.5 | 640.2 | 3051.2 |
| branch `i % 7`, N=1e6 | 53.5 | 462.8 | 308.3 | 313 |

Checksums agree across all modes and with PyTorch.  Disabling fusion in the
same runtime costs 3x to 11x, growing with `K`.  For small tensors the fused
loop runs in about 14.5 us against 42 us for `torch.compile`, because the loop
control and the launch are JIT-compiled machine code.  With a branch on the
loop counter inside the region, `torch.compile` hits its recompilation limit
and falls back to eager speed while the fused loop keeps a single kernel.

## Pitfalls that produce wrong numbers

- The JIT's default loop threshold is 1039 iterations; the harness sets
  `threshold=3` so 200-iteration runs are compiled.
- Tensor ops are elidable, so a chain over loop-invariant inputs is hoisted
  out of the loop.  The benchmark carries `h` across iterations for that reason.
- The lazy chain is materialised where it escapes, which is the loop end, after
  `release_range` in program order.  The harness forces `h` before releasing
  the previous iteration's device buffers; otherwise the input buffer is
  recycled and zeroed before the launch reads it.
- Device buffers are released in bulk (`rtensor.release_range`,
  `rtensor.reset_device`); there are no per-tensor finalizers yet.

To verify what a run actually launches, an `LD_PRELOAD` shim that counts
`cuLaunchKernel` calls is the quickest check; launches must grow by exactly
one per iteration in `fused` mode.
