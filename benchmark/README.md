# Virtual-tensor JIT benchmark

Measures one loop iteration of `h <- relu(h * b + b)` repeated `K` times over
rank-1 float64 tensors of `N` elements, under the PyPy meta-tracing JIT with
tensor ops kept virtual and fused into one Triton kernel (`rpython/rtensor/`,
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
| `RTENSOR_BUDGET_MB` | `8` | live device memory above which a launch first runs a GC to recycle buffers |
| `RTENSOR_CUBLAS` | PyTorch wheel's bundled `libcublas.so.13` | path to `libcublas.so`, dlopen'd lazily for `matmul` (variant 6) |
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
| `VARIANT` | `0` plain chain; `1` a branch `if i % 7 == 0: h = h + b` inside the fused region (a guard, no force); `2` the same branch after an explicit force of `h`, modelling a graph break; `3` a data-dependent branch `if sum(h).item() > 0` (a true force in every system); `4` the chain continued inside `try/except` with an exception raised every 5th iteration; `5` a host-side write to `/dev/null` every 50th iteration between tensor ops; `6` an MLP forward (see below), `K` unused; `7` the same MLP with a reverse-mode backward pass and an SGD step per iteration, `K` unused; `8` a Transformer block forward per iteration (see below), `K` unused; `9` a small CNN forward per iteration (see below), `K` unused; `10` a 2-block Transformer training step (forward, backward and SGD) per iteration (see below), `K` unused |

`VARIANT 6` runs a 3-layer MLP forward (`rpython/rtensor/nn.py`) each
iteration: `x` is `(B, 256)` with `B = N / 256`, each layer is `256x256`.
Matrix multiply is delegated to cuBLAS (`rt_cuda_matmul` in
`rpython/rtensor/cuda.c`, `nvidia-cublas`'s `cublasDgemm_v2`) as a
non-fusible library call: it is `@jit.dont_look_inside`, so it is never part
of a fused Triton kernel, and its tensor arguments are always forced before
the call. The bias add and relu after each matmul stay virtual and fuse into
one Triton launch each, so one MLP forward pass launches 3 matmuls (cuBLAS,
not counted as Triton kernels) plus 3 fused elementwise launches (the last one
also folds in the final `sum`). `RTENSOR_CUBLAS` overrides the path to
`libcublas.so`; it defaults to the path where the PyTorch wheel installs it
under `nvidia/cu13/lib`.

`VARIANT 7` adds the backward pass and a plain SGD step to variant 6.  The
autograd tape is ordinary RPython host code in `rpython/rtensor/nn.py`:
each forward method records a small `Node` object, `Tensor.backward()` walks
the tape in reverse and issues the gradient formulas through the same
`rtensor` primitives, so the backward pass is traced and fused by the same
JIT as the forward one.  The elementwise gradients use a fusible `relugrad`
opcode (`g if y > 0 else 0`) and `sum(axis=0)` for bias gradients; the two
matmul gradients (`dx = dy @ W^T` and `dW = x^T @ dy`) reuse `rt_cuda_matmul`
with cuBLAS transpose flags, so no transpose kernel is needed.  The loss is
`sum(y)` and the SGD step is `p = p + (-lr) * p.grad` with `-lr` held in a
`(1,)` tensor, which fuses into one kernel per parameter.
`VARIANT 9` runs a small CNN forward each iteration: `conv2d` 3x3 with
padding 1, batchnorm in inference form, `relu`, `maxpool` 2x2, flatten and a
linear layer to 10 classes.  The input is `(B, 3*32*32)` with `B = N / 3072`,
the convolution has 8 output channels.  Images are stored channel-major, one
image per row of a `(B, C*H*W)` matrix.  The convolution is
`col2chw(im2col(x) @ Wcol)`: `im2col` builds the `(B*H*W, C*3*3)` patch matrix,
`Wcol` is `(C*3*3, O)`, the matmul is the same cuBLAS call as variant 6, and
`col2chw` transposes the `(B*H*W, O)` result back to channel-major
`(B, O*H*W)`.  `im2col`, `col2chw` and `maxpool2` are hand-written Triton IR
gather kernels (`to_ttir_gather` in `rpython/rtensor/kernels.py`) compiled through
the same `triton_compile.py` path and cached per (op, geometry); the geometry
is baked into the kernel text as constants, so they launch through the
existing `rt_cuda_launch` shim with no extra arguments.  Like `matmul` they are
`@jit.dont_look_inside`, so they force their arguments and are never fused.
Batchnorm needs a per-channel broadcast, which is expressed without a new
broadcast code: the `(B, C*H*W)` matrix is viewed as `(B*C, H*W)` and the
folded per-channel scale and shift (`gamma/sqrt(var+eps)` and
`beta - mean*gamma/sqrt(var+eps)`) are held in `(B*C, 1)` tensors, so the
existing column broadcast `BC_R_COL` indexes them.  Batchnorm and the following
`relu` therefore stay virtual and fuse into a single Triton launch, as do the
final bias add, `relu` and `sum`.  One forward launches 3 gather kernels plus
3 fused elementwise kernels and 2 cuBLAS matmuls.  The checksum is
loop-carried: the same `x` is fed every iteration and `sum(y).item()` is
accumulated, which forces every iteration.

| `K` | chain length |
| `N` | tensor size |
| `ITERS` | timed iterations |

Output is one line:

    mode variant k n iters warm_s steady_us kernels acc compiled_in_timed launches_per_iter dtype

`RTENSOR_DTYPE` selects the element type of every tensor the benchmark builds
(`float64`, the default, `float32` or `float16`) and is echoed in the trailing
`dtype` column; `torch_bench.py` takes the same values in `TORCH_DTYPE`.

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

Small model (variant 6): a 3-layer MLP `relu(x @ W + b)` with D=256 and
rows = N/256, float64, matmul delegated to cuBLAS (never fused) and the bias
add plus relu fused into one Triton kernel per layer (3 launches per
iteration in `fused`, 6 in `eager`).  Per-iteration microseconds:

| rows | fused | eager (ours) | torch.compile | torch eager |
|---|---|---|---|---|
| 100 | 300.9 | 314.8 | 390.9 | 386.6 |
| 1000 | 1064.9 | 1042.4 | 1269.5 | 1256.1 |

Variant 7 (`forward + backward + SGD`) runs the same shapes with the tape
enabled; per iteration it issues 8 cuBLAS matmuls (3 forward, 3 for `dW`,
and 2 for `dx`; the first layer's input needs no gradient) plus the fused
elementwise kernels for the relu gradients, the column sums for the bias
gradients and the 6 parameter updates.  Checksums (the final loss) agree with PyTorch:

    PYTHONPATH=. python2 benchmark/rtensor_bench.py nojit 7 1 2560 3
    $RTENSOR_PYTHON benchmark/torch_bench.py eager 7 1 2560 3

both print `63.912787`.

The float64 matmul dominates on this GPU (1/64-rate fp64), so fusion of the
epilogue is worth only a few percent here and `torch.compile` gains nothing
over eager either; the point of the row is that the wrapper layer, cuBLAS
delegation and fusion compose, and that checksums agree with PyTorch.

Training step (variant 7): the same MLP with `loss = sum(y)`, reverse-mode
autograd written as ordinary RPython code on top of the tensor library, and
an SGD update.  Per iteration: 8 cuBLAS matmuls (3 forward, 3 weight
gradients, 2 input gradients) plus the fused elementwise gradient and update
kernels.  Per-iteration microseconds and kernel launches:

| rows | fused (launches) | eager, ours (launches) | torch.compile | torch eager |
|---|---|---|---|---|
| 100 | 712.2 (18) | 740.4 (26) | 839.5 | 793.7 |
| 1000 | 2862.5 (18) | 2902.0 (26) | 3054.1 | 3083.2 |

The backward pass fuses into the same kind of kernels as the forward pass
because the tape is host code the tracer sees; the remaining launches are the
matmuls and the per-parameter updates.

Model demos (variants 8 and 9, forward only, float64, 100 iterations).  The
Transformer block has D=64, H=4 heads: three (D,D) projections, a head-split
gather, one strided-batched cuBLAS call for the scores, scale and softmax over
the (H*rows, rows) matrix in a single row-tiled kernel (one Triton program per
row, tile width specialised to the promoted column count), a second batched
call with V, a head-merge gather, the output projection, and layernorms and
the MLP as further fused kernels; the torch baseline uses the same batched
formulation.  The CNN is conv3x3 (im2col + cuBLAS) -> batchnorm -> relu ->
maxpool2 -> linear.  Per-iteration microseconds and launches (matmuls not
counted):

| model | size | fused (launches) | eager, ours (launches) | torch.compile | torch eager |
|---|---|---|---|---|---|
| Transformer | 64 rows | 219.6 (9) | 540.3 (38) | 250.4 | 388.9 |
| Transformer | 1024 rows | 1695.5 (9) | 3825.2 (38) | 1883.6 | 1874.2 |
| CNN | 1 image | 81.4 (6) | 89.2 (10) | 202.9 | 135.6 |
| CNN | 21 images | 230.3 (6) | 238.6 (10) | 354.5 | 265.1 |

Three runtime changes took the Transformer from 978 / 4077 us to these
numbers: device buffers are zero-filled only when the kernel accumulates
atomically (a memset per allocation had cost about 100 GPU operations per
iteration), scalar constants such as `eps` and the attention scale are cached
device tensors instead of per-call uploads, and attention runs batched over
heads (8 cuBLAS calls instead of 14 small ones).  Row kernels compiled with
the tile width of the promoted column count (64 for layernorm, 1024 for the
attention softmax) then halved the 1024-row time again.

Precision (`RTENSOR_DTYPE` / `TORCH_DTYPE`, 100 iterations, per-iteration
microseconds, fused mode; torch uses the same dtype end to end):

| model | size | dtype | fused | torch.compile | torch eager |
|---|---|---|---|---|---|
| Transformer | 64 rows | float32 | 79.8 | 228.5 | 365.4 |
| Transformer | 64 rows | float16 | 60.1 | 214.5 | 376.0 |
| Transformer | 1024 rows | float32 | 251.2 | 336.0 | 423.5 |
| Transformer | 1024 rows | float16 | 108.9 | 273.5 | 415.9 |
| CNN | 1 image | float32 | 60.6 | 194.0 | 136.5 |
| CNN | 21 images | float32 | 67.3 | 240.3 | 189.8 |

Below fp64 the GPU is no longer compute bound on this card and the launch
count decides: nine fused kernels plus eight cuBLAS calls per Transformer
iteration against several dozen kernels in PyTorch.  Checksums agree with
torch to the printed digits in float32 and within 1e-2 relative in float16.

Variant 3 is a real force in every system (the value is needed on the host),
so both produce two kernels.  Variants 1 and 4 depend on the loop counter;
Dynamo specialises on the integer, hits its recompilation limit and falls back
to eager, while the meta-tracer keeps the branch as a guard inside one fused
kernel.  Variant 5 breaks the Dynamo graph at the `print`; here the write is
an ordinary residual call that does not touch the virtual tensors.  The 1.12
launches per iteration in variant 5 and the 1.15 in variant 2 are the eager
re-evaluation that the blackhole performs when a guard fails before its
bridge is compiled; they converge to 1.0 over longer runs.

## App-level measurements

`benchmark/applevel/` holds two plain-Python scripts run on a PyPy translated
with `--withmod-_tensor` (translate with
`rpython -Ojit --no-shared pypy/goal/targetpypystandalone.py --withmod-_tensor`,
about an hour; module options go after the target).  Run them with a low JIT
threshold, since they iterate a few hundred times:

    ./pypy-c -S --jit threshold=3,function_threshold=3,trace_eagerness=2 benchmark/applevel/chain_unrolled.py 1000000 300

`chain_unrolled.py` writes the four `relu(h*b+b)` steps out; `chain_loop.py`
uses an inner `for` loop.  Per-iteration microseconds on the RTX 3090, 300
iterations, with the RPython benchmark and PyTorch for the same chain:

| N | app-level, unrolled | RPython bench (fused) | torch.compile | app-level, inner `for` |
|---|---|---|---|---|
| 1e4 | 19.9 | 15.0 | 41.9 | |
| 1e5 | 20.3 | 15.0 | 43.3 | |
| 1e6 | 49.4 | 40.2 | 69.0 | 933.3 |

The model demos from unmodified Python on the same binary (100 iterations,
checksums equal to PyTorch):

| script | size | app-level | RPython bench (fused) | torch.compile | torch eager |
|---|---|---|---|---|---|
| `transformer.py` | 64 rows | 287.9 | 219.6 | 250.4 | 388.9 |
| `transformer.py` | 1024 rows | 2560.7 | 1695.5 | 1883.6 | 1874.2 |
| `cnn.py` | 1 image | 137.3 | 81.4 | 202.9 | 135.6 |
| `cnn.py` | 21 images | 278.8 | 230.3 | 354.5 | 265.1 |

The app-level versions pay for the Python-level object model around every
op (a `W_Tensor` and an inner `Tensor` per result, virtual inside a trace but
real at every residual call boundary such as the cuBLAS calls), which is the
gap to the RPython benchmark.

The unrolled chain becomes one fused kernel per iteration from unmodified
Python: the trace shows a single `tensor_launch` call with the tensor objects
kept virtual.  With an inner `for` loop, PyPy gives the inner loop its own
trace, so every inner iteration is a kernel boundary (4 launches, and the
loop exit re-enters through the interpreter); unrolling short constant loops
inside the tracer is the missing piece there.  With the default JIT threshold
(1039 iterations) short scripts spend most of their time in the interpreter,
where every op is a separate one-node GPU kernel.

`VARIANT 8` runs one pre-norm Transformer block forward per iteration with
`D = 64`, `H = 4` heads and `rows = N / 64`, loop-carried (`x = block(x)`):

    x + sum_h Wo_h(attention(LN1(x) Wq_h, LN1(x) Wk_h, LN1(x) Wv_h))

followed by `x + MLP(LN2(x))` with two `relu` layers.  Heads are kept as
separate `(D, D/H)` projections and recombined through per-head `(D/H, D)`
output projections, so no column slicing of a `(rows, D)` matrix is needed.
`layernorm`, `softmax` and `attention` are ordinary host code in
`rpython/rtensor/nn.py` built from the primitives `sub`, `div`, `exp`,
`sqrt`, `sum(axis)` and `maxr(axis)`; the six matmuls per head go to cuBLAS.
Every weight uses the same deterministic init as the MLP variants (element
`i` is `((i * 7) % 13 - 6) / D`), MLP biases are `0.01`, `gamma = 1`,
`beta = 0` and `eps = 1e-5`, so the checksum can be compared with PyTorch:

    PYTHONPATH=. python2 benchmark/rtensor_bench.py nojit 8 1 256 2
    $RTENSOR_PYTHON benchmark/torch_bench.py eager 8 1 256 2

both print `51.262221`.

`VARIANT 10` trains a `B = 2` block Transformer end to end, `D = 64`, `H = 4`,
`rows = N / 64`, forward, backward and one SGD step per iteration.  The model
is variant 8's block repeated twice followed by a final `Linear` to `D`
outputs, the loss is `sum(y)` and the update is `p = p + (-lr) * p.grad` with
`lr = 1e-6`, the same as variant 7.  Every parameter (`Wq`, `Wk`, `Wv`, `Wo`,
both `gamma`/`beta` pairs, the two MLP layers of each block and the head)
requires a gradient, so the autograd tape covers the whole block: `softmax`
and `layernorm` are compositions of primitives and need no dedicated backward
nodes, while `sub`, `div`, `exp`, `sqrt`, `max(axis=1)`, `matmul` with
transpose flags, batched `bmm` and `head_split`/`head_merge` each have a
`Node` whose `apply` is written in the same primitives as the forward, so the
backward pass fuses the same way.  `max(axis=1)` routes its gradient with a
new fusible `eqmask(a, b)` opcode (`1.0` where `a == b`, else `0.0`) and a
column broadcast; ties are not divided by their multiplicity, so a row with
two equal maxima sends the gradient to both, unlike torch's `argmax`
behaviour.  The softmax `max` gradient cancels analytically anyway, so the
loss trajectory still matches torch:

    PYTHONPATH=. python2 benchmark/rtensor_bench.py nojit 10 1 256 2
    $RTENSOR_PYTHON benchmark/torch_bench.py eager 10 1 256 2

both print `64.778154`, and after 40 steps at `N = 1024` both print
`200.537667`.  One unfused training step is 254 Triton launches (cuBLAS
matmuls are not counted).  Traced, one step of the same model shape
(`test_transformer_train_step` in `rpython/jit/metainterp/test/test_tensor.py`)
is 78 fused Triton launches plus 22 intermediates read back out of earlier
kernels as extra outputs, 39 cuBLAS `matmul`s (13 forward, 26 backward),
12 batched `bmm`s (4 forward, 8 backward) and 16 `head_split`/`head_merge`
gather kernels (8 forward, 8 backward).

`benchmark/applevel/transformer.py ROWS ITERS` and `benchmark/applevel/cnn.py
IMAGES ITERS` reproduce variants 8 and 9 through the app-level `_tensor` API
and `lib_pypy/tensorlite.py` instead of `nn.py`, using the same
`TB_D`/`TB_H`/`CNN_*` sizes and deterministic weight init, so their printed
`checksum` matches `rtensor_bench.py`'s for the same `ROWS*TB_D`/`IMAGES*3072`
and `ITERS`; each does 10 warmup iterations on a throwaway model before timing
a freshly built one for `ITERS` iterations. Run them on a translated PyPy the
same way as `chain_unrolled.py`, or check them against the untranslated
interpreter (slow, so keep sizes tiny) with `RTENSOR_PYTHON=... RTENSOR_CPU=1
python2 pypy/bin/pyinteractive.py --withmod-_tensor
benchmark/applevel/transformer.py 4 2`.

Broadcasting now covers column vectors as well as row vectors: the `bcast`
parameter of an elementwise op is `0` none, `1`/`3` right/left row vector
(index `i % c`), `2`/`4` right/left scalar, `5`/`6` right/left column vector
(index `i / c`).  `add`/`sub`/`mul`/`div` pick the code from the shapes: same
shape gives `0`, a 2-D operand against a `(cols,)` vector gives a row
broadcast, against a `(rows, 1)` or (when `rows != cols`) a `(rows,)` vector a
column broadcast, and `(1,)` a scalar.  When `rows == cols` the shapes are
ambiguous; a 1-D vector is then taken as a row vector, so a column broadcast
of a square matrix must be spelled with shape `(rows, 1)`.

Row-wise `max` (the softmax shift) has no GPU kernel: PTX has no
`atom.max.f64`, and Triton 3.8 accepts `tt.atomic_rmw max` on `f64` only by
reinterpreting the bits as `s64`, which orders negative doubles backwards.
Axis-0/1 max therefore falls back to `eval_op_cpu`, and only a whole-tensor
`max` that fits in one Triton program is emitted (`tt.reduce` with
`arith.maximumf`, masked lanes set to `-inf`).

## Pitfalls that produce wrong numbers

- The JIT's default loop threshold is 1039 iterations; the harness sets
  `threshold=3` so 200-iteration runs are compiled.
- Tensor ops are elidable, so a chain over loop-invariant inputs is hoisted
  out of the loop.  The benchmark carries `h` across iterations for that reason.
- Device buffers are owned by finalizable objects and recycled through a
  free list.  When no buffer of the requested size is free, an allocation
  first runs a GC once `max(RTENSOR_BUDGET_MB, live-after-last-GC)` bytes or
  an adaptive number of fresh buffers (64 to 65536, doubled after a GC that
  recycled nothing, halved after one that did) have been allocated since the
  last GC.  Retained activations in training therefore do not trigger a GC
  per allocation, while short-lived chains recycle every iteration.

To verify what a run actually launches, an `LD_PRELOAD` shim that counts
`cuLaunchKernel` calls is the quickest check; launches must grow by exactly
one per iteration in `fused` mode.

## App-level `_tensor` module

`rpython/rtensor/nn.py` (autograd `Tensor`, `Linear`, `MLP`, `sgd_step`)
is also exposed to app-level Python as the built-in module `_tensor`
(`pypy/module/_tensor/`), so a translated PyPy can run

    import _tensor, tensorlite
    x = _tensor.tensor([[1.0, 2.0]])
    layer = tensorlite.Linear(_tensor.tensor([[1.0], [1.0]], requires_grad=True),
                               _tensor.tensor([0.0], requires_grad=True))
    y = layer(x)
    loss = y.sum()
    loss.backward()

`_tensor.tensor(data, shape=None, requires_grad=False)` accepts nested lists
(flattened and shape-inferred at app level) and `_tensor.zeros(shape,
requires_grad=False)`; `Tensor` instances support `add`/`mul`/`sub`/`div`/`__add__`/
`__mul__`/`__sub__`/`__div__`/`__truediv__`/`exp`/`sqrt`/`max(axis=-1)`/
`relu`/`sum(axis=-1)`/`item`/`matmul(other, transpose_b=False)`/`reshape`/
`add_`/`mul_`/`detach`/`backward`/`zero_grad` and the
`shape`/`size`/`grad`/`requires_grad` properties.  Only the ops that have
gradient rules (`add`, `mul`, `relu`, `sum`, `matmul`, `reshape`) can be
backpropagated through; `backward` over the others raises `ValueError`. This PyPy2's grammar has no `@` operator (no `MatMult` AST node),
so matrix multiplication is the `matmul` method only, no `__matmul__`.
`lib_pypy/tensorlite.py` is a pure app-level `Linear`/`MLP`/`sgd_step`,
`softmax`/`layernorm`/`attention`/`Head`/`TransformerBlock` built
on that API (mirroring `nn.py`), for `torch`-style user code.

Build a PyPy with the module:

    python2 rpython/bin/rpython -Ojit --withmod-_tensor pypy/goal/targetpypystandalone.py

`--withmod-_tensor` is generated automatically for every directory under
`pypy/module` that has an `__init__.py`
(see `all_modules` in `pypy/config/pypyoption.py`), so `_tensor` needed no
changes to `essential_modules`/`default_modules` there to gain the flag; it
is simply not enabled by default, so it must be passed explicitly both for
`pyinteractive.py` and for a real translation.

Tests: `RTENSOR_PYTHON=/home/yusuke/.venvs/triton/bin/python RTENSOR_CPU=1
python2 pytest.py pypy/module/_tensor/test/ -q` (`RTENSOR_CPU=1` skips the
GPU device-init path so the untranslated, single-threaded app-level
interpreter — which is slow — only exercises the CPU tensor ops; sizes in
the tests are kept to a handful of elements for the same reason).
