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

It covers variants 0, 8, 9 and 13 in float64, and only adds `app` rows, so run
it alongside a normal micro run rather than instead of one. `summarize` then
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

## Checking a setup

    benchmark/paper/bench.sh check          # every mode, ~100 s
    benchmark/paper/bench.sh check micro dtypes

runs the smallest thing that still exercises each mode - our three execution
modes, the three dtypes, both torch baselines, one model, one ablation, one
dynamic sweep, then summarize and plot - into a scratch directory, and exits
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
