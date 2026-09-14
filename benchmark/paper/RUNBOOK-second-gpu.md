# Running the paper grid on a second GPU (RTX 5070 Ti on stillinlove)

The paper's ordering claims are checked on two accelerators. Everything
below runs on the second host; only the last step comes back to the main
checkout. The working branch is `pypytensor`, pushed to both the public
fork (github.com/3tty0n/pypy) and the private `tensorpypy` remote
(git@github.com:3tty0n/tensorpypy.git); `main` on the public fork carries
none of this work.

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

    git clone -b pypytensor git@github.com:3tty0n/pypy.git pypy-tile-ir   # first time (or tensorpypy.git)
    cd pypy-tile-ir
    git fetch origin && git checkout pypytensor && git pull --ff-only
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

## IREE on sm_120

IREE 3.11.0 cannot target sm_120. `iree-compile` is invoked with
`--iree-cuda-target=sm_120` (from RTENSOR_CC) and answers `missing GPU target
in #hal.executable.target`, then `failed to configure executables`, for every
model and micro variant; `iree-stderr.txt` is the full output for distilgpt2.

It is the target name, not the host: `IREE_CUDA_TARGET=sm_86` compiles and runs
on this GPU (distilgpt2 at 17.6 ms). Those numbers would come from a kernel
tuned for Ampere and reaching Blackwell through PTX JIT, so they are not an
IREE-on-5070-Ti result and the column is left empty instead.

## cuDNN TF32 on Blackwell: mixer_b16 and vit-tiny

At the default settings on the RTX 5070 Ti, torch's `mixer_b16` and `vit-tiny`
rows missed the 1e-3 tolerance (maxabsdiff about 2.5e-3) while JAX on the same
GPU agreed to 4e-5: cuDNN picks a TF32 kernel for the patch-embedding
convolution there, and the RTX 3090 records the same `torch_tf32_cudnn 1`
without doing it. Comparing a TF32 torch against our float32 is not like for
like, so those two models' torch rows are measured with the flag off:

    OUT=... TORCH_CUDNN_TF32=0 BASELINES="eager compile compile-ro compile-mat" \
      ./benchmark/paper/bench.sh models mixer_b16 vit-tiny

which brings them to 3.9e-5 (mixer_b16) and 1.2e-5 (vit-tiny). Re-run this
after any fresh `bench.sh all` on a Blackwell host, deleting the default-TF32
rows for those two models first, or their accuracy column will fail.

## Re-measuring the TensorRT column

TensorRT has no f64 kernels (its precisions are i8/f16/f32/bf16/f8/f4), and
`torch.compile(backend="tensorrt")` falls back to eager without raising when it
refuses a dtype - so every float64 "torch-tensorrt" row measured before
2026-09-15 is PyTorch eager under a TensorRT label. Those rows have been
removed from the committed result sets; `torch_bench.py` and
`applevel/torch_common.py` now refuse the point instead of reporting a
fallback. The TensorRT column is therefore float32/float16 plus the models,
which run in float32; the float64 grid has no TensorRT number and cannot have
one.

The RTX 3090 set still has to be re-measured, because its rows were taken with
torch-tensorrt 2.9 (the 2.9.0+cu130 wheels have since been withdrawn from the
index; the venv now pins 2.14.0+cu130, the same torch as the main venv). On
luchkylilac, against the result directory being corrected:

    ./benchmark/paper/bench.sh setup            # rebuilds the trt venv on the new pin
    OUT=$PWD/benchmark/results/luchkylilac-rtx3090/paper-<date> \
      BASELINES=tensorrt ./benchmark/paper/bench.sh micro --baselines-only -- models --baselines-only
    OUT=... ./benchmark/paper/bench.sh summarize -- plot

Check the rows before believing them: `launches_per_iter` must differ from the
same model's torch-eager row, and `steady_us` within about 1% of torch-eager is
the signature of a silent fallback, not a result.

## Bringing the results back

On the second host, commit the result directory on a branch and push:

    git add benchmark/results/stillinlove-rtx5070ti
    git commit -m "record the <date> RTX 5070 Ti run"
    git push origin pypytensor && git push tensorpypy pypytensor   # never main on the public fork

On the main host (luchkylilac), pull, then render the cross-GPU figures
from the RTX 3090 result set against the new one:

    git pull --ff-only
    OUT=$PWD/benchmark/results/luchkylilac-rtx3090/paper-2026-09-11 \
      ./benchmark/paper/bench.sh compare benchmark/results/stillinlove-rtx5070ti/paper-<date>

This writes `figures/compare_speedup.*`, `compare_models.*` and
`compare_dynamic.*` into the RTX 3090 result directory; copy those
`*.pdf` and `*.tex` files into `metatensor-paper/figures/` (the paper repo
keeps figures as a snapshot; see its FIGURES.md).
