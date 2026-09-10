# MLSys paper evaluation harness

## Requirements

- NVIDIA GPU + driver. Driver >= 580 for the default CUDA 13 torch/triton
  wheels; on an older driver set `TORCH_INDEX=https://download.pytorch.org/whl/cu128`
  (or `cu121`) to match what your driver supports.
- `gcc`, `make`, `$CUDA_HOME/include/cuda.h` (default `CUDA_HOME=/usr/local/cuda`).
- A `python2`/`pypy2` on PATH for the RPython toolchain, or none at all —
  `setup.sh` downloads PyPy2.7 if it's missing.
- `python3` to create the venv.

## Reproduce

    benchmark/paper/bench.sh setup
    benchmark/paper/bench.sh all

The Python 3 dependencies are declared in `benchmark/paper/pyproject.toml` and
pinned in `requirements.lock`; `setup.sh` installs from the lock with `uv`,
fetching uv into `.toolchain/` if it is not on PATH and falling back to plain
pip (`NO_UV=1` forces the fallback). The pin matters: which torch and Triton
you get decides whether the kernels run on the GPU at all, and a mismatch
against the driver is silent. Move a pin with

    uv pip compile benchmark/paper/pyproject.toml -o benchmark/paper/requirements.lock

`bench.sh setup` is idempotent: it creates the venv at `${VENV:-$HOME/.venvs/metatensor}`,
detects the GPU compute capability with torch and writes it (plus every
derived path) to `benchmark/paper/env.sh`, translates `metatensor-bench` and
`pypy-c` into `benchmark/paper/build/` (skipped if already built), and
exports the eight checkpoints into `${WEIGHTS:-benchmark/paper/weights}`
(skip with `SKIP_WEIGHTS=1`). `bench.sh` sources `config.sh`, which sources
`env.sh` if present, so no manual exports are needed afterwards.

Expected durations: pypy-c translation 15-60 min depending on cores, the
bench binary about 3 min, the full `bench.sh all` grid about 2 h.

Useful overrides: `VENV`, `BUILD`, `WEIGHTS`, `TORCH_INDEX`, `MAKE_JOBS`,
and for `bench.sh`: `ITERS`, `WARMUP`, `ROUNDS`, `OUT`,
`SKIP_EXPORT`/`SKIP_MICRO`/`SKIP_MODELS`/`SKIP_ABLATION`/`SKIP_DYNAMIC`/`SKIP_SUMMARIZE`.

### Run one piece

`bench.sh` also runs any single piece of the evaluation; each piece appends
to `$OUT/*.tsv` (writing a header only if the file doesn't exist yet), so
several piecewise runs plus a final `bench.sh summarize` compose one result
set. `bench.sh all` clears `$OUT/*.tsv` first for a clean run.

    benchmark/paper/bench.sh micro 8 1 256000        # one microbenchmark point
    benchmark/paper/bench.sh models distilgpt2 resnet18-b1
    benchmark/paper/bench.sh ablation fusion
    benchmark/paper/bench.sh summarize
    benchmark/paper/bench.sh plot                    # figures into $OUT/figures

## Two binaries, two micro columns

The microbenchmarks default to `build/metatensor-bench`, a translated RPython
program with no bytecode dispatch, while `torch_bench.py` pays for CPython.
That gap flatters us, so `--applevel` runs the same models as Python through
the translated `build/pypy-c`, which is the apples-to-apples comparison and the
same execution path the end-to-end models use:

    benchmark/paper/bench.sh micro --applevel        # adds "app" rows

It covers every micro variant, float64 only (micro.py rejects other dtypes,
so the precision sweep is skipped under `--applevel`), and only adds `app`
rows, so run it alongside a normal micro run rather than instead of one.
`summarize` then
fills the `app-level (ours)` column and `interp tax` = app / fused, which is
what the PyPy interpreter costs on top of the mechanism itself.

## Figures

`bench.sh plot` - also the last step of `bench.sh all` - renders the paper
figures from the tsv files into `$OUT/figures`: vector PDFs with TrueType
fonts embedded, sized to one MLSys column or the full text width. Beside each
figure it writes a `.tex` tabular of the same numbers, for the values a plot
cannot carry.

Every figure carries the spread over `$ROUNDS` rounds: marks sit at the median
with whiskers to the observed minimum and maximum, and the tables give the same
range. With three rounds a standard deviation would be noise, so the range is
what is reported. Speedup figures show the envelope of the ratio, taken from
opposite ends of the two ranges, because the rounds are not paired across the
two systems.

    micro_speedup   per-benchmark speedup over torch.compile
    fusion          fused against interpreted, same benchmark
    models          end-to-end inference, three systems
    dynamic         cost against sequence length
    precision       speedup at float64/32/16
    ablation        one runtime knob at a time

Flags: `--only NAME` (repeatable), `--format pdf,png`, `--column
single|double` to override the per-figure default, `--texture` to hatch the
fills for grayscale print, `--titles` for slide versions (a paper puts the
title in the caption).

### Two machines side by side

    benchmark/paper/bench.sh compare ../results/luchkylilac-rtx3090/paper-2026-09-07

compares `$OUT` against one or two other result directories (three machines is
the limit: the categorical palette is used in fixed order and never cycled, so
a fourth would need a hue that does not exist). Each series is named by its
accelerator alone - a figure that ends up in a paper has no business naming
somebody's server - read from the run's `machine.txt` and falling back to the
`<gpu>` half of its directory name for runs recorded before that file existed.
Two runs on the same model of accelerator are separated by their run date.

    compare_speedup   speedup per benchmark, one series per machine
    compare_models    end-to-end ratio per model, one series per machine
    compare_dynamic   ratio against torch.compile over sequence length

These compare *ratios*, not microseconds: an absolute time is a property of the
accelerator, while the ratio is a property of the system under test, and the
claim to support is that the ordering survives a change of GPU. Points the two
runs did not measure at the same size - the memory fit shrinks variant 10
differently per GPU - are left out and named on stderr.

`bench.sh list` prints the available model and ablation names and the micro grid.

## Other backends: JAX/XLA, IREE, handwritten Triton

Besides PyTorch eager and `torch.compile`, the harness measures the same
workloads on three more systems. Each is one more script that prints the
same line the torch twin prints, so the run scripts, `summarize.py` and
`plot.py` treat it as another `mode` (micro) or `system` (models) and no
benchmark had to change.

| backend | script | mode / system | what it covers |
|---|---|---|---|
| JAX/XLA (`jax.jit`) | `benchmark/jax_bench.py`, `benchmark/applevel/jax_models.py` | `jax` | every micro variant 0-13; gpt2 (distilgpt2, tiny-gpt2), bert (tiny, mini), vit-tiny, mixer_b16 |
| IREE (StableHLO from the same JAX code, CUDA HAL) | same two scripts, mode `iree` | `iree` | micro variants 0, 6, 11, 12, 13; bert-mini, vit-tiny |
| handwritten Triton | `benchmark/triton_bench.py` | `triton` | micro variants 0-5 (fused elementwise chain), 6 (MLP), 11 (reduction), 12 (matmul + bias + relu), 13 (attention: Triton GEMMs + a flash-style online-softmax kernel) |
| Torch-TensorRT | - | - | not built: no wheel for the Python 3.14 / CUDA 13 pairing the lock pins (`uv pip install torch-tensorrt` fails to build from source) |

### Installing

JAX and IREE go into a second venv, `${JAX_VENV:-$VENV-jax}`, because their
CUDA wheels would otherwise move the torch/triton pins that `requirements.lock`
exists to hold still. `bench.sh setup` creates it from `requirements-jax.lock`
(`WITH_JAX=0` skips it) and writes `JAX_PYTHON` into `env.sh`; without it
the jax/iree rows are simply absent and everything else runs. By hand:

    uv venv --python 3.14 ~/.venvs/metatensor-jax
    VIRTUAL_ENV=~/.venvs/metatensor-jax uv pip sync benchmark/paper/requirements-jax.lock
    export JAX_PYTHON=~/.venvs/metatensor-jax/bin/python

The Triton baseline needs nothing beyond the torch venv (`triton` is already
pinned there). `machine.txt` records jax, jaxlib and IREE versions next to
torch and triton.

### Running

Nothing changes: `bench.sh micro` adds `jax`, `iree` (where supported) and
`triton` rows to every point, `bench.sh models` adds `jax` (and `iree` for
bert-mini and vit-tiny) rows, and `bench.sh check baselines` runs one point
on each and checks its accumulator against torch eager. `summarize.md` gains
the JAX/IREE/Triton columns, an `ours/Triton` ratio (MetaTensor latency over
handwritten-Triton latency) and a compilation-overhead table; `plot` writes
`micro_baselines.*`, `compile_overhead.*` and the extra model bars.

### Methodology per backend

All backends share the shapes, dtypes, seeds, iteration counts, rounds and
GPU of the existing runs, and all synchronize before and after the timed
loop (`torch.cuda.synchronize`, `block_until_ready`, IREE `to_host`).
Compilation is excluded from `steady_us` everywhere and reported separately
in `results.jsonl` (`compile_ms`, `first_run_ms`; both also appear as the
last two fields of the micro line and as `compile_ms=`/`first_run_ms=` on
the model line):

- JAX: `compile_ms` is `jit(f).lower(...).compile()` alone;
  `first_run_ms` the first executed call. The Python control-flow variants
  1-5 keep their branches in Python and jit only the tensor expressions, the
  same split `torch.compile` ends up with after its graph breaks.
  `jax_default_matmul_precision` is `highest`, so fp32 GEMMs are real fp32
  like torch eager and cuBLAS, not TF32. Weights are jit arguments, not
  baked constants.
- IREE: the identical Python function is exported to StableHLO with
  `jax.export` and compiled with `--iree-cuda-target=sm_<cc>` and f64 kept
  (`--iree-input-demote-f64-to-f32=false`). `compile_ms` is the IREE
  compile only (the trace is `export_ms`). It runs through the Python
  runtime API with inputs moved to the device once; each call still pays
  that API's dispatch, and IREE's CUDA backend has no cuBLAS, so its
  steady-state numbers are well behind XLA. It is reported as-is, as an
  optional column, not tuned.
- Triton: `compile_ms` is the first launch of each kernel (Triton compiles
  then), `first_run_ms` the first iteration after every kernel exists.
  Kernels use fixed tiles (64x64x32 GEMM, 1024-element elementwise blocks,
  32x32 attention tiles at fp64 to fit sm_86 shared memory) and no
  autotuning; they are the kernel a person writes in an afternoon, not the
  best one for this GPU.
- torch.compile: `first_run_ms` is compile plus one iteration; the split is
  not observable from outside, so `compile_ms` is -1 and the break-even
  column uses the first run.
- MetaTensor: `first_run_ms` on the model line is the first forward, which
  carries tracing and kernel compilation.

Break-even (`summarize.md`, `compile_overhead.tex`) is
`(first_run(system) - first_run(eager)) / (steady(eager) - steady(system))`
in iterations, against PyTorch eager.

Correctness: every new row is checked like the torch rows. Micro lines carry
the same accumulator as torch (they agree to the printed digits on every
supported variant; fp16 attention differs in the last digits of a sum that
cancels to ~1e-4). Model lines carry `maxabsdiff` against the stored
MetaTensor logits, the same reference and the same tolerance (`check.sh`:
1e-3) the torch rows use. No tolerance was changed for any backend.

Known comparability limits:

- Triton's GEMM is a Triton kernel; MetaTensor's GEMM is cuBLAS with the
  epilogue fused around it. The `ours/Triton` ratio on the matmul variants
  compares against a plain Triton GEMM, not cuBLAS.
- Triton and torch.compile cache compiled kernels on disk across processes
  (`~/.triton`, inductor cache), so their `compile_ms` reflects that cache;
  JAX's persistent compilation cache is off, so its `compile_ms` is always
  a cold compile.
- IREE covers a subset and its numbers include Python runtime dispatch.
- `kernel_count` is left empty for JAX, IREE and torch; only MetaTensor and
  the Triton baseline (where it is the number of distinct kernels) report it.
- The precision sweep (float32/float16) runs jax and triton at those dtypes
  too; IREE is float64 only there.

## Checking a setup

    benchmark/paper/bench.sh check          # every mode, ~100 s
    benchmark/paper/bench.sh check micro dtypes

runs the smallest thing that still exercises each mode - our three execution
modes, the three dtypes, both torch baselines, the jax/iree/triton baselines,
one model, one ablation, one dynamic sweep, then summarize and plot - into a scratch directory, and exits
non-zero on the first thing that is wrong. Run it before a grid, and after any
change to the toolchain.

The check that earns its keep is `launches_per_iter`: a kernel that fails to
load falls back to the CPU silently, and the run still produces plausible
numbers, just two orders of magnitude slower. It also compares the accumulator
across fused/eager/nojit and against torch, so a kernel that runs but computes
the wrong thing fails too.

## Adding a microbenchmark

The grid and the names the figures use live in `benchmark/benchmarks.toml`, so
a new point is one entry there rather than a loop in `run_micro.sh` plus a
label table in `plot.py`. The two implementations still have to be written
separately - `benchmark/bench/` for the RPython side, `benchmark/torch_bench.py`
for the baseline - which is the comparison the harness exists to make.

## Result files

Each stage appends to its tsv as before, and every measurement also lands in
`$OUT/results.jsonl` as a record that names its own fields. The tsv files are
positional and a row's meaning depends on which system wrote it - micro.tsv
carries the dtype in the `breaks` column for our rows and in `graphs` for
torch's, because the two emitters have different field counts - so the jsonl is
the format to read from new code. `summarize.py` and `plot.py` still read the
tsv files.

Runtime knobs (see also `benchmark/README.md`): `RTENSOR_FLAT_BLOCK`
(elements per block for elementwise/gather kernels, default 4096; smaller
values trade off differently per workload, see the root README) and
`RTENSOR_BUDGET_MB` (device GC byte threshold before a GC pass, default 8).

## Cloud / Docker

    docker build -t metatensor-paper -f benchmark/paper/Dockerfile .
    docker run --gpus all metatensor-paper

The image builds on `nvidia/cuda:13.0.1-devel-ubuntu24.04`, installs the
PyPy2.7 toolchain and runs `setup.sh` with `SKIP_WEIGHTS=1` at build time
(weights are downloaded on first `run_all.sh` inside the container, since
they need a writable volume to persist). Not built to completion in this
environment — verify on a machine with GPU-backed docker before relying on
it.

## Compare against the reference run

Results are filed as `benchmark/results/<host>-<gpu>/paper-<date>/`, because a
number means nothing without knowing which machine and which accelerator
produced it; `$OUT/machine.txt` carries the rest of the provenance (driver,
toolkit, wheel versions, CPU, and the knobs the run used). `RUN_HOST` and
`RUN_GPU` override the two path components; `OUT` overrides the whole path.

The evaluation in the paper is
`benchmark/results/luchkylilac-rtx3090/paper-2026-09-07/summary.md` (RTX 3090,
sm_86). Diff your `$OUT/summary.md` against it; per-op numbers will differ by
GPU, but ordering between `fused`/`torch.compile`/`eager` should hold.

## Troubleshooting

- `metatensor: cuMemAlloc ... failed` / a kernel silently falling back to
  running on the CPU / no cuBLAS found: all three are silent-degrade
  warnings, not hard failures — the run keeps going on the CPU path, which
  is much slower and will skew numbers. Check `nvidia-smi`, `$CUDA_HOME`,
  and that `RTENSOR_CUBLAS` in `env.sh` points at a real `libcublas.so.*`
  inside the venv (`nvidia/cu*/lib/`).
- `Failed to initialize NVML: Driver/library version mismatch`: a
  `nvidia-smi` vs. kernel-driver version skew, usually harmless — CUDA
  programs still run fine through the kernel driver; only `nvidia-smi`
  itself is affected. A reboot or a matching NVML package clears it.
- Blackwell (sm_120, e.g. RTX 50-series): needs Triton >= 3.3. The compute
  capability probe in `setup.sh` (`torch.cuda.get_device_capability()`)
  reports `(12, 0)`, which `setup.sh` turns into `RTENSOR_CC=120` — this is
  the format Triton's CUDA backend expects (`triton/backends/nvidia/compiler.py`
  builds `sm_{capability}` directly from the integer, so `120` -> `sm_120`).
  If `RTENSOR_CC` is unset, `triton_compile.py` asks torch for
  `get_device_capability()` on every kernel compile, so an unset value is
  always correct for the GPU torch sees; set it only to pin a target.
