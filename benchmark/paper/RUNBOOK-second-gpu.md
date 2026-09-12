# Running the paper grid on a second GPU (RTX 5070 Ti on stillinlove)

The paper's ordering claims are checked on two accelerators. Everything
below runs on the second host; only the last step comes back to the main
checkout. Commit `82d3d119cd` (branch `mlsys2027`) is the reference state.

## Requirements on the host

- NVIDIA driver >= 580 (CUDA 13 wheels; `nvidia-smi` must list the GPU and
  report no driver/library mismatch). RTX 5070 Ti is sm_120 (Blackwell);
  the pinned Triton 3.8 / torch 2.14+cu130 support it, older Triton (< 3.3)
  does not.
- `gcc`, `make`, CUDA toolkit with `$CUDA_HOME/include/cuda.h`
  (`CUDA_HOME=/usr/local/cuda` by default), `python3` >= 3.11 with venv,
  `uv` on PATH (setup downloads one otherwise), a `python2`/`pypy2` for the
  RPython toolchain (setup downloads PyPy2 if missing).
- nsys is optional (only `bench.sh gap` needs it).
- About 40 GB of disk for the three venvs and the translation; 1-2 h of
  translation time; the full grid takes about 4 h.

## Commands

    cd ~/src/github.com/3tty0n/pypy-tile-ir      # or wherever the clone is
    git fetch origin && git checkout mlsys2027 && git pull --ff-only
    ./benchmark/paper/bench.sh setup             # venvs (torch, jax+iree, tensorrt),
                                                 # translation, checkpoint export;
                                                 # writes benchmark/paper/env.sh with
                                                 # RTENSOR_CC detected from the GPU (120)
    ./benchmark/paper/bench.sh check             # every mode in minutes; must print 0 failed
    SKIP_SETUP=1 ./benchmark/paper/bench.sh all  # micro, models, ablation, dynamic, summarize, plot
    ./benchmark/paper/bench.sh micro --applevel  # app-level rows (interp tax)
    ./benchmark/paper/bench.sh warmup -- batch -- deopt -- fusion   # the extra experiments
    ./benchmark/paper/bench.sh gap               # only if nsys exists on the host

If setup's optional venvs fail (no jax/tensorrt wheel for the host), rerun
with `WITH_JAX=0` or `WITH_TRT=0`; the corresponding columns are simply
absent. If `check` reports a CPU fallback (launches_per_iter 0), stop and
look at `RTENSOR_CC` in `env.sh` before running the grid.

Results land in `benchmark/results/<host>-<gpu>/paper-<date>/`, which for
this host is `benchmark/results/stillinlove-rtx5070ti/paper-<date>/`
(`machine.txt` there records driver, wheels, compute capability, warm-up
counts and binary hashes).

## Bringing the results back

On the second host, commit the result directory on a branch and push:

    git add benchmark/results/stillinlove-rtx5070ti
    git commit -m "record the <date> RTX 5070 Ti run"
    git push origin mlsys2027

On the main host (luchkylilac), pull, then render the cross-GPU figures
from the RTX 3090 result set against the new one:

    git pull --ff-only
    OUT=$PWD/benchmark/results/luchkylilac-rtx3090/paper-2026-09-11 \
      ./benchmark/paper/bench.sh compare benchmark/results/stillinlove-rtx5070ti/paper-<date>

This writes `figures/compare_speedup.*`, `compare_models.*` and
`compare_dynamic.*` into the RTX 3090 result directory; copy those
`*.pdf` and `*.tex` files into `metatensor-paper/figures/` (the paper repo
keeps figures as a snapshot; see its FIGURES.md).
