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

`bench.sh list` prints the available model and ablation names and the micro grid.

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

The evaluation in the paper is `benchmark/results/paper-2026-09-07-luchkylilac/summary.md`
(RTX 3090, sm_86). Diff your `$OUT/summary.md` against it; per-op numbers
will differ by GPU, but ordering between `fused`/`torch.compile`/`eager`
should hold.

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
