# Virtual-tensor JIT benchmark

RPython benchmark (`metatensor_bench.py`) for the tensor-as-virtual JIT, with
PyTorch baselines (`torch_bench.py`).  All numbers: RTX 3090, float64
unless noted, steady-state microseconds per iteration.

## Setup

    python3 -m venv ~/.venvs/triton
    ~/.venvs/triton/bin/pip install triton torch --index-url https://download.pytorch.org/whl/cu130
    export RTENSOR_PYTHON=~/.venvs/triton/bin/python   # compiles .ttir to PTX
    benchmark/build.sh                                  # -> benchmark/metatensor-bench, ~10 min, needs python2

Other env vars: `RTENSOR_DTYPE` (`float64|float32|float16`), `RTENSOR_CPU=1`
(no GPU), `RTENSOR_BUDGET_MB` (device GC byte threshold, default 8),
`RTENSOR_FLAT_BLOCK` (elements per block of the elementwise and gather
kernels, default 4096; `256` is 3% faster on distilgpt2 and 2.5% on
SmolLM2, 19% slower on a 1e6-element chain),
`RTENSOR_PROFILE=1`, `CUDA_HOME`, `RTENSOR_CUBLAS` (path to `libcublas.so`; read at
run time, and if set at translation time it becomes the compiled-in default,
otherwise `libcublas.so` is looked up through the dynamic loader).

## Run

    benchmark/metatensor-bench MODE VARIANT K N ITERS
    $RTENSOR_PYTHON benchmark/torch_bench.py {eager|compile} VARIANT K N ITERS
    benchmark/paper/run_all.sh          # full grid -> benchmark/results/paper-<date>-<host>/summary.md

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
The variants live in the `benchmark/bench/` package (`chain.py` for 0-5,
`mlp.py` for 6-7, `transformer.py` for 8/10/13, `cnn.py` for 9, `reduce.py`
for 11-12, shared setup in `common.py`); `metatensor_bench.py` is the thin
translation entry point.

## Results

| workload | fused (ours) | torch.compile | torch eager |
|---|---|---|---|
| chain K=4, N=1e6 | 39.9 | 68.7 | 312.4 |
| chain + loop-counter branch (variant 1) | 53.3 | 308.2 (recompiles, falls back) | 317.3 |
| chain + host write (variant 5) | 68.7 | 333.6 (graph break) | 343.0 |
| reduction, 1000 rows (variant 11) | 63.6 | 284.0 | 221.2 |
| matmul + fused epilogue, 1000 rows (variant 12) | 382.9 | 637.4 | 622.3 |
| attention, 1024 rows (variant 13) | 4789.3 | 6497.0 | 5097.5 |
| MLP forward, 1000 rows | 1064.9 | 1269.5 | 1256.1 |
| MLP training, 1000 rows | 2862.5 | 3054.1 | 3083.2 |
| Transformer forward, 1024 rows | 1684.0 | 1883.6 | 1874.2 |
| Transformer forward, 1024 rows, float16 | 108.9 | 273.5 | 415.9 |
| Transformer training, 64 rows | 1905.0 | 1816.0 | 3210.0 |
| Transformer training, 1024 rows | 14089.3 | 9531.5 | 10513.4 |
| CNN forward, 21 images | 228.1 | 354.5 | 265.1 |

Launches per iteration in fused mode: chain 1, Transformer forward 5,
attention-only 1, CNN 6, Transformer training 96 (eager mode: 38, 10, 251).  Where we win it is
fewer launches and no graph breaks at Python control flow; where we lose
(large training steps) the cuBLAS matmul share dominates and the residual
is not yet attributed.  A fused elementwise chain only runs as one row
kernel when its row fits a tile, so a column broadcast over a row wider
than `RTENSOR_BLOCK` is compiled flat instead of falling back to the
per-node CPU path.  Raw runs are in `results/`.

Real HuggingFace checkpoints (`applevel/gpt2_export.py` writes the weights,
`applevel/gpt2.py` runs them on our PyPy, `applevel/gpt2_torch.py` runs
`transformers.GPT2LMHeadModel` on the same token ids; the `llama_*.py`
triple does the same for Llama-architecture models; seq 64, float32,
argmax agrees at every position, max logits difference 1.6e-4 for GPT-2
and 1.5e-4 for SmolLM2):

| model | ours (PyPy) | torch.compile | torch eager |
|---|---|---|---|
| distilgpt2 (6 layers, 768, 12 heads) | 1372 | 1297 | 2461 |
| sshleifer/tiny-gpt2 (2 layers, width 2) | 189 | 371 | 1431 |
| SmolLM2-135M (Llama, 30 layers, 576, 9/3 heads) | 4097 | 5342 | 14755 |
| prajjwal1/bert-tiny (2 layers, 128, 2 heads) | 393 | 483 | 968 |
| bert_uncased_L-4_H-256_A-4 (4 layers, 256, 4 heads) | 664 | 781 | 1590 |
| vit-tiny-patch16-224 (12 layers, 192, 3 heads, 197 tokens) | 1649 | 2086 | 3383 |
| mixer_b16_224 (12 layers, 768, 196 tokens) | 3793 | 3126 | 2872 |
| resnet18, batch 1 | 787 | 970 | 1393 |
| resnet18, batch 8 | 3191 | 2173 | 2242 |

The BERT, ViT, Mixer and ResNet triples are `applevel/{bert,vit,mixer,resnet}
{_export,,_torch}.py` and follow the GPT-2 pattern: the export script writes
the real checkpoint to `weights.bin` plus `index.json`, our runner runs it on
our PyPy and dumps the logits, the torch runner runs the reference model
(`BertForMaskedLM`, `ViTForImageClassification`, `timm`) on the same ids or
the same pixel tensor and reports `maxabsdiff` against them.  Medians of 3
interleaved rounds, 200 iterations after 30 warmup, float32, seq 64 for BERT
and one real 224x224 photo for the vision models.  Agreement: BERT argmax at
all 64 positions, max logits difference 3.1e-5 (tiny) and 2.1e-5 (mini); ViT
and Mixer identical top-5 with 1.2e-5 and 3.7e-5; ResNet-18 identical top-5
with 5.2e-3 against torch on the GPU but 8.6e-6 against a plain fp32 numpy
reference of the same arithmetic, so the gap is torch's TF32 convolutions,
not ours.  Launches per iteration: BERT-mini 68 (17 per layer), ViT 172,
Mixer 243, ResNet-18 60 (was 101).

New primitives per model: BERT needed none -- the exact (erf) GELU is built
app-level out of `relu`/`mul`/`add`/`div`/`exp` with the Abramowitz-Stegun
7.1.26 approximation (max error 1.4e-7) and fuses into the surrounding
elementwise chain, and bidirectional attention is the causal one with the
mask argument left out.  ViT needed none: the 16x16 stride-16 patch
embedding is a row gather (`take`) over the image viewed as a (rows, 16)
table, and the CLS token is one extra all-zero gathered row plus a constant,
so patch embedding, CLS and position embeddings are one gather, one GEMM and
one add.  Mixer needed none beyond the strided `im2col` below: token mixing
is `W @ X` with the bias broadcast down a column, so the (patches, channels)
activation is never transposed.  ResNet-18 needed the only interp-level
change in this batch: `im2col` grew a `stride` parameter and `maxpool2` grew
`k`, `stride` and `pad` (max pooling clamps out-of-range window positions
instead of masking them, which is exact because a clamped position is always
another position of the same window), so `conv2d`, `Conv2d` and `MaxPool2d`
now take `k`, `stride` and `pad` and cover the 7x7 stride-2 stem, the
stride-2 3x3 and 1x1 downsample convolutions and the 3x3 stride-2 pad-1 pool.
Two more gathers followed, `im2col_nhwc` and `maxpool2_nhwc`, the
channels-last forms of those two; the CHW `im2col`, `col2chw`, `maxpool2` and
`conv2d` stay as they are for the RPython CNN benchmark (variant 9) and the
app-level `CNN` model.

What fuses in ResNet: the activation stays channels-last, shaped
`(N*H*W, C)`, from the exported image to the last block, so a convolution is
one `im2col_nhwc` gather into `(N*Ho*Wo, k*k*C)` and one GEMM against the
weight the export script already permuted to `(k, k, C, O)` -- the result is
`(N*Ho*Wo, O)`, the next activation, with no scatter back to planes.  That
kills the `col2chw` launch per convolution outright, and it also makes every
BatchNorm a plain row broadcast of `gamma`/`beta` over the `C` columns, which
fuses with its ReLU and with the residual addition into one elementwise
kernel per convolution instead of two.  1x1 stride-1 convolutions skip the
gather entirely (the activation already is the GEMM operand) and the 1x1
stride-2 downsamples reuse `im2col_nhwc` with `k=1` as a row subsample, so
they need no separate primitive; max pooling gained an NHWC variant with the
same clamped padding, and global average pooling is a GEMM against a constant
`(N, N*H*W)` averaging matrix, so it needs no axis-0 reduction.  Per
iteration the counted (Triton) launches went from 81.3 to 39.6 -- 41 gathers
to 21, 40.3 elementwise kernels to 18.6 -- with the 20 cuBLAS GEMMs on top,
so 101 to 60; device time per iteration, batch 1: im2col 221 us -> 208,
col2chw 138 -> 0, elementwise 223 -> 120, total counted 590 -> 336.  At
batch 8: im2col 987 -> 968, col2chw 337 -> 0, elementwise 519 -> 280, total
1885 -> 1294.  What still does not fuse is the convolution itself: im2col
materializes a `(rows, k*k*C)` buffer, so the gather and the GEMM stay two
launches and the 9x memory blow-up of the 3x3 stem-resolution convolutions is
why batch 8 is still 1.5x torch -- at that size we are bandwidth-bound on the
im2col buffer (75% of our device time), which an implicit-GEMM convolution
that gathers inside the GEMM's k-loop would remove.

q, k and v are one GEMM against a weight concatenated at model build time,
so attention reads q, k and v as column slices of one (rows, 3d) buffer
through the `lda`/`ldb` and offset arguments of the strided-batched cuBLAS
call: per layer GPT-2 went from 3 GEMMs plus 3 bias launches to 1 GEMM plus
1 bias launch, and Llama from 5 GEMMs (q, k, v and the two 576x576 RoPE
permutation matmuls) plus 2 RoPE chains to 1 GEMM, 1 `rot_half` gather and
1 fused chain -- RoPE now rotates through a gather kernel with the sign
folded into the sine table, over the whole (rows, 3d) buffer at once
(`cos` is 1 and `sin` is 0 over the v columns).  Before that the app-level
`softmax`, `layer_norm`, `rms_norm`, `gelu` and `silu` were routed to the
fused RPython row kernels and attention was moved onto strided-batched
cuBLAS without head gathers, which took distilgpt2 from 23 non-GEMM
launches per layer to 9; it is 7 now.  What is left against torch.compile
is the per-kernel launch floor: `RTENSOR_FLAT_BLOCK=256` still buys about
3% on distilgpt2 and 2.5% on SmolLM2, so a size-dependent elementwise block
is worth roughly that much.

App-level scripts in `applevel/` (Transformer, CNN, chains) run on a PyPy
translated with `--withmod-_metatensor` and track the RPython numbers within
about 1.3x:

    ./pypy-c -S --jit threshold=3,function_threshold=3,trace_eagerness=2 benchmark/applevel/transformer.py

## Pitfalls

- The JIT loop threshold is 1039 by default; the bench sets `threshold=3`.
- `RTENSOR_BLOCK` bounds the row-kernel tile as well as being the default
  elementwise block, so lowering it past the widest row silently routes row
  kernels to the CPU (10x slower).  `RTENSOR_FLAT_BLOCK` moves the
  elementwise and gather block on its own.  A 64x768 tensor at the default
  4096 fills 12 of the GPU's 82 SMs.
- Tensor ops are elidable, so a chain over loop-invariant inputs is hoisted;
  carry the result across iterations.
- Device buffers under 8 MB are carved from 32 MB slabs; a fresh `cuMemAlloc`
  costs about 15 us and was half of distilgpt2's time.
- Device buffers are recycled by GC finalizers.  A GC runs when no buffer of
  the requested size is free and either `max(RTENSOR_BUDGET_MB, live after
  last GC)` bytes or an adaptive count of fresh allocations (1..65536,
  doubled when a GC recycles nothing) has accumulated.  Touching this needs
  chain, CNN and both training sizes re-measured together.
- Three conditions silently route work to the CPU evaluator: no `libcublas`
  found (matmul only), a missing `triton_compile.py` next to `kernels.py`
  in the tree the binary was built from, and a failed `cuMemAlloc` (the
  shim prints one warning).  A run that is 50x slower than the tables is
  one of these, not a JIT problem.
- The host is noisy; interleave ours and torch runs when comparing.
