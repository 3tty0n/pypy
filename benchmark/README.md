# Virtual-tensor JIT benchmark

RPython benchmark (`rtensor_bench.py`) for the tensor-as-virtual JIT, with
PyTorch baselines (`torch_bench.py`).  All numbers: RTX 3090, float64
unless noted, steady-state microseconds per iteration.

## Setup

    python3 -m venv ~/.venvs/triton
    ~/.venvs/triton/bin/pip install triton torch --index-url https://download.pytorch.org/whl/cu130
    export RTENSOR_PYTHON=~/.venvs/triton/bin/python   # compiles .ttir to PTX
    benchmark/build.sh                                  # -> benchmark/rtensor-bench, ~10 min, needs python2

Other env vars: `RTENSOR_DTYPE` (`float64|float32|float16`), `RTENSOR_CPU=1`
(no GPU), `RTENSOR_BUDGET_MB` (device GC byte threshold, default 8),
`RTENSOR_PROFILE=1`, `CUDA_HOME`, `RTENSOR_CUBLAS` (path to `libcublas.so`; read at
run time, and if set at translation time it becomes the compiled-in default,
otherwise `libcublas.so` is looked up through the dynamic loader).

## Run

    benchmark/rtensor-bench MODE VARIANT K N ITERS
    $RTENSOR_PYTHON benchmark/torch_bench.py {eager|compile} VARIANT K N ITERS
    benchmark/run_experiments.sh        # full grid -> benchmark/results/<date>-<host>/summary.txt

`MODE`: `fused` (ours), `eager` (tensor opt off, one kernel per op), `nojit`.
`K` is the chain length, `N` the tensor size or row count.

| variant | what |
|---|---|
| 0 | elementwise chain of K ops |
| 1–5 | the chain with a guard, a forced graph break, an `.item()` branch, `try/except`, a host write |
| 6 / 7 | 3-layer MLP forward / training step |
| 8 / 10 | Transformer block forward / 2-block training step |
| 9 | small CNN forward |
| 11 | row reduction, carried: `h = sum(x*x, axis=1); x = x + h` |
| 12 | one matmul layer, carried: `x = relu(matmul(x, W) + b)` |
| 13 | multi-head self-attention only, carried, D=256, 8 heads |

Output columns: `mode variant k n iters warm_s steady_us kernels acc compiled_in_timed launches_per_iter dtype`.
`launches_per_iter` must be 1.0 for the chain in `fused` mode.

## Results

| workload | fused (ours) | torch.compile | torch eager |
|---|---|---|---|
| chain K=4, N=1e6 | 67.9 | 68.7 | 312.4 |
| chain + loop-counter branch (variant 1) | 53.3 | 308.2 (recompiles, falls back) | 317.3 |
| chain + host write (variant 5) | 68.7 | 333.6 (graph break) | 343.0 |
| reduction, 1000 rows (variant 11) | 63.6 | 284.0 | 221.2 |
| matmul + fused epilogue, 1000 rows (variant 12) | 382.9 | 637.4 | 622.3 |
| attention, 1000 rows (variant 13) | 4844.3 | 6497.0 | 5097.5 |
| MLP forward, 1000 rows | 1064.9 | 1269.5 | 1256.1 |
| MLP training, 1000 rows | 2862.5 | 3054.1 | 3083.2 |
| Transformer forward, 1024 rows | 1690.5 | 1883.6 | 1874.2 |
| Transformer forward, 1024 rows, float16 | 108.9 | 273.5 | 415.9 |
| Transformer training, 64 rows | 1933.1 | 1816.0 | 3210.0 |
| Transformer training, 1024 rows | 13625.5 | 9531.5 | 10513.4 |
| CNN forward, 21 images | 228.1 | 354.5 | 265.1 |

Launches per iteration in fused mode: chain 1, Transformer forward 9, CNN 6,
Transformer training 112 (eager mode: 38, 10, 251).  Where we win it is
fewer launches and no graph breaks at Python control flow; where we lose
(large training steps) the cuBLAS matmul share dominates and the residual
is not yet attributed.  Raw runs are in `results/`.

Real HuggingFace checkpoints (`applevel/gpt2_export.py` writes the weights,
`applevel/gpt2.py` runs them on our PyPy, `applevel/gpt2_torch.py` runs
`transformers.GPT2LMHeadModel` on the same token ids; the `llama_*.py`
triple does the same for Llama-architecture models; seq 64, float32,
argmax agrees at every position, max logits difference 1.7e-4 for GPT-2
and 1.9e-4 for SmolLM2):

| model | ours (PyPy) | torch.compile | torch eager |
|---|---|---|---|
| distilgpt2 (6 layers, 768, 12 heads) | 2984 | 1356 | 2521 |
| sshleifer/tiny-gpt2 (2 layers, width 2) | 558 | 373 | 1418 |
| SmolLM2-135M (Llama, 30 layers, 576, 9/3 heads) | 11593 | 6304 | 14855 |

distilgpt2 takes 157 launches per iteration (cuBLAS included), SmolLM2-135M
576; the gap to torch.compile is the open item.

App-level scripts in `applevel/` (Transformer, CNN, chains) run on a PyPy
translated with `--withmod-_tensor` and track the RPython numbers within
about 1.3x:

    ./pypy-c -S --jit threshold=3,function_threshold=3,trace_eagerness=2 benchmark/applevel/transformer.py

## Pitfalls

- The JIT loop threshold is 1039 by default; the bench sets `threshold=3`.
- Tensor ops are elidable, so a chain over loop-invariant inputs is hoisted;
  carry the result across iterations.
- Device buffers are recycled by GC finalizers.  A GC runs when no buffer of
  the requested size is free and either `max(RTENSOR_BUDGET_MB, live after
  last GC)` bytes or an adaptive count of fresh allocations (1..65536,
  doubled when a GC recycles nothing) has accumulated.  Touching this needs
  chain, CNN and both training sizes re-measured together.
- The host is noisy; interleave ours and torch runs when comparing.
