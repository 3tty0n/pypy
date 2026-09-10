"""Handwritten Triton twin of torch_bench.py for the kernel-quality reference.

    triton_bench.py triton VARIANT K N ITERS

Variants 0-5 (elementwise chain), 6 (MLP), 11 (reduction), 12 (matmul with
bias+relu epilogue) and 13 (attention) are written as plain Triton kernels
with fixed block sizes and no autotuning; the point is a kernel a person
would write in an afternoon, not the best kernel this GPU can run.  Data,
shapes and dtype are exactly torch_bench.py's, so the acc field must match
torch's.  The line adds compile_ms and first_run_ms after dtype, like
jax_bench.py.  Every torch call outside the kernels is data setup; the timed
loop launches only Triton kernels (torch.empty for outputs aside).
"""
import math, os, sys, time

import torch
import triton
import triton.language as tl

mode, variant, k, n, iters = (sys.argv[1], int(sys.argv[2]), int(sys.argv[3]),
                              int(sys.argv[4]), int(sys.argv[5]))
dev = "cuda"
DTNAME = os.environ.get("TORCH_DTYPE", "float64")
DT = {"float64": torch.float64, "float32": torch.float32,
      "float16": torch.float16}[DTNAME]
ACC = {"float64": tl.float64, "float32": tl.float32,
       "float16": tl.float32}[DTNAME]

MLP_D = 256
RED_COLS = 64
ATT_D = 256
ATT_H = 8

COMPILE_MS = [0.0]
FIRST_MS = [None]
KERNELS = set()
COMPILED_NOW = [False]


def rows_of(d):
    return max(n // d, 1)


def gen_weight(rows, cols, divisor):
    return torch.tensor([float((i * 7) % 13 - 6) / divisor
                         for i in range(rows * cols)],
                        dtype=DT, device=dev).reshape(rows, cols)


def bias(size, value):
    return torch.full((size,), value, dtype=DT, device=dev)


def make_input(rows, cols):
    return torch.tensor([(i % 7) - 3.0 for i in range(rows * cols)],
                        dtype=DT, device=dev).reshape(rows, cols)


# -- kernels ----------------------------------------------------------------

@triton.jit
def chain_kernel(h_ptr, b_ptr, out_ptr, n, K: tl.constexpr,
                 BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    h = tl.load(h_ptr + offs, mask=mask)
    b = tl.load(b_ptr + offs, mask=mask)
    for _ in tl.static_range(K):
        h = tl.maximum(h * b + b, 0.0)
    tl.store(out_ptr + offs, h, mask=mask)


@triton.jit
def binop_kernel(h_ptr, b_ptr, out_ptr, n, MUL: tl.constexpr,
                 BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    h = tl.load(h_ptr + offs, mask=mask)
    b = tl.load(b_ptr + offs, mask=mask)
    tl.store(out_ptr + offs, h * b if MUL else h + b, mask=mask)


@triton.jit
def rowsq_kernel(x_ptr, out_ptr, rows, COLS: tl.constexpr,
                 BLOCK_R: tl.constexpr):
    r = tl.program_id(0) * BLOCK_R + tl.arange(0, BLOCK_R)
    c = tl.arange(0, COLS)
    ptrs = r[:, None] * COLS + c[None, :]
    mask = (r < rows)[:, None]
    x = tl.load(x_ptr + ptrs, mask=mask, other=0.0)
    h = tl.sum(x * x, axis=1)
    tl.store(out_ptr + ptrs, x + h[:, None], mask=mask)


@triton.jit
def matmul_kernel(a_ptr, b_ptr, bias_ptr, c_ptr, M, N, K,
                  sam, sak, sbk, sbn, scm, scn,
                  ACT: tl.constexpr, ACC_T: tl.constexpr,
                  BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
                  BLOCK_K: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    rk = tl.arange(0, BLOCK_K)
    a_ptrs = a_ptr + rm[:, None] * sam + rk[None, :] * sak
    b_ptrs = b_ptr + rk[:, None] * sbk + rn[None, :] * sbn
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=ACC_T)
    for kk in range(0, tl.cdiv(K, BLOCK_K)):
        kmask = rk < K - kk * BLOCK_K
        a = tl.load(a_ptrs, mask=(rm < M)[:, None] & kmask[None, :], other=0.0)
        b = tl.load(b_ptrs, mask=kmask[:, None] & (rn < N)[None, :], other=0.0)
        acc = tl.dot(a, b, acc)
        a_ptrs += BLOCK_K * sak
        b_ptrs += BLOCK_K * sbk
    if ACT:
        acc = acc + tl.load(bias_ptr + rn, mask=rn < N, other=0.0)[None, :]
        acc = tl.maximum(acc, 0.0)
    c = acc.to(c_ptr.dtype.element_ty)
    mask = (rm < M)[:, None] & (rn < N)[None, :]
    tl.store(c_ptr + rm[:, None] * scm + rn[None, :] * scn, c, mask=mask)


@triton.jit
def attn_kernel(q_ptr, k_ptr, v_ptr, o_ptr, N, D, scale,
                DH: tl.constexpr, ACC_T: tl.constexpr,
                BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    """softmax(q k^T * scale) v for one head and one block of query rows.

    q, k, v and o are [N, D] row-major with head h in columns h*DH:(h+1)*DH,
    the layout torch_bench's attention() produces; online softmax over key
    blocks in the style of the Triton fused-attention tutorial, no causal
    mask because torch_bench's attention has none."""
    pid_m = tl.program_id(0)
    head = tl.program_id(1)
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rd = tl.arange(0, DH)
    col = head * DH + rd
    q = tl.load(q_ptr + rm[:, None] * D + col[None, :],
                mask=(rm < N)[:, None], other=0.0)
    m_i = tl.full((BLOCK_M,), float("-inf"), dtype=ACC_T)
    l_i = tl.zeros((BLOCK_M,), dtype=ACC_T)
    acc = tl.zeros((BLOCK_M, DH), dtype=ACC_T)
    for start in range(0, N, BLOCK_N):
        rn = start + tl.arange(0, BLOCK_N)
        kmask = rn < N
        kb = tl.load(k_ptr + rn[:, None] * D + col[None, :],
                     mask=kmask[:, None], other=0.0)
        vb = tl.load(v_ptr + rn[:, None] * D + col[None, :],
                     mask=kmask[:, None], other=0.0)
        s = tl.dot(q, tl.trans(kb)).to(ACC_T) * scale
        s = tl.where(kmask[None, :], s, float("-inf"))
        m_new = tl.maximum(m_i, tl.max(s, axis=1))
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(s - m_new[:, None])
        l_i = l_i * alpha + tl.sum(p, axis=1)
        acc = acc * alpha[:, None]
        acc = tl.dot(p.to(vb.dtype), vb, acc)
        m_i = m_new
    o = (acc / l_i[:, None]).to(o_ptr.dtype.element_ty)
    tl.store(o_ptr + rm[:, None] * D + col[None, :], o,
             mask=(rm < N)[:, None])


# -- launch wrappers ----------------------------------------------------------

EW_BLOCK = 1024
BM, BN, BK = 64, 64, 32
RED_BLOCK_R = 16
# fp64 tiles of 64x64 overrun the 100KB of shared memory on sm_86.
ATT_BM, ATT_BN = (32, 32) if DT == torch.float64 else (64, 64)


def launch(kernel, grid, *args, **kw):
    """Launch, charging the first launch of each kernel to compile_ms.

    Triton compiles on first launch; time it once per kernel, then the
    second launch's cost is what the steady state pays.  The first timed
    iteration (see first()) is measured on already-compiled kernels."""
    key = (kernel.fn.__name__, tuple(v for v in kw.values()))
    if key not in KERNELS:
        KERNELS.add(key)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        kernel[grid](*args, **kw)
        torch.cuda.synchronize()
        COMPILE_MS[0] += (time.perf_counter() - t0) * 1e3
        COMPILED_NOW[0] = True
        return
    kernel[grid](*args, **kw)


def chain(h, b, K):
    out = torch.empty_like(h)
    launch(chain_kernel, (triton.cdiv(h.numel(), EW_BLOCK),), h, b, out,
           h.numel(), K=K, BLOCK=EW_BLOCK)
    return out


def binop(h, b, mul):
    out = torch.empty_like(h)
    launch(binop_kernel, (triton.cdiv(h.numel(), EW_BLOCK),), h, b, out,
           h.numel(), MUL=mul, BLOCK=EW_BLOCK)
    return out


def rowsq(x):
    out = torch.empty_like(x)
    launch(rowsq_kernel, (triton.cdiv(x.shape[0], RED_BLOCK_R),), x, out,
           x.shape[0], COLS=x.shape[1], BLOCK_R=RED_BLOCK_R)
    return out


def matmul(a, b, bias_=None):
    M, K = a.shape
    K2, N = b.shape
    assert K == K2
    c = torch.empty((M, N), dtype=a.dtype, device=a.device)
    grid = (triton.cdiv(M, BM), triton.cdiv(N, BN))
    launch(matmul_kernel, grid, a, b, bias_ if bias_ is not None else a, c,
           M, N, K, a.stride(0), a.stride(1), b.stride(0), b.stride(1),
           c.stride(0), c.stride(1), ACT=bias_ is not None, ACC_T=ACC,
           BLOCK_M=BM, BLOCK_N=BN, BLOCK_K=BK)
    return c


def attention(q, k_, v, heads, scale):
    N, D = q.shape
    o = torch.empty_like(q)
    launch(attn_kernel, (triton.cdiv(N, ATT_BM), heads), q, k_, v, o, N, D,
           scale, DH=D // heads, ACC_T=ACC, BLOCK_M=ATT_BM, BLOCK_N=ATT_BN)
    return o


# -- benchmarks ---------------------------------------------------------------

def first(fn, *args):
    """Time the first call whose kernels were all compiled beforehand."""
    if FIRST_MS[0] is None:
        COMPILED_NOW[0] = False
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        out = fn(*args)
        torch.cuda.synchronize()
        if not COMPILED_NOW[0]:
            FIRST_MS[0] = (time.perf_counter() - t0) * 1e3
        return out
    return fn(*args)


def runner(step, x0):
    def run(iters):
        x = x0
        for _ in range(iters):
            x = first(step, x)
        torch.cuda.synchronize()
        return x.sum().item()
    return run


def run_mlp():
    x = make_input(rows_of(MLP_D), MLP_D)
    layers = [(gen_weight(MLP_D, MLP_D, MLP_D), bias(MLP_D, 0.01))
              for _ in range(3)]

    def forward(y):
        for W, b in layers:
            y = matmul(y, W, b)
        return y
    return runner(forward, x)


def run_reduction():
    x = make_input(rows_of(RED_COLS), RED_COLS)
    return runner(rowsq, x)


def run_matmul():
    x = make_input(rows_of(MLP_D), MLP_D)
    W, b = gen_weight(MLP_D, MLP_D, MLP_D), bias(MLP_D, 0.01)
    return runner(lambda t: matmul(t, W, b), x)


def run_attn():
    rows = rows_of(ATT_D)
    x = make_input(rows, ATT_D)
    dh = ATT_D // ATT_H
    scale = 1.0 / math.sqrt(dh)
    heads = [(gen_weight(ATT_D, dh, ATT_D), gen_weight(ATT_D, dh, ATT_D),
              gen_weight(ATT_D, dh, ATT_D), gen_weight(dh, ATT_D, ATT_D))
             for _ in range(ATT_H)]
    WQ = torch.cat([hd[0] for hd in heads], dim=1)
    WK = torch.cat([hd[1] for hd in heads], dim=1)
    WV = torch.cat([hd[2] for hd in heads], dim=1)
    WO = torch.cat([hd[3] for hd in heads], dim=0)

    def forward(t):
        q, k_, v = matmul(t, WQ), matmul(t, WK), matmul(t, WV)
        return matmul(attention(q, k_, v, ATT_H, scale), WO)
    return runner(forward, x)


def report(warm, steady, acc):
    print("%s %d %d %d %d %f %f %d %f 0 -1 -1 %s %.3f %.3f" % (
        mode, variant, k, n, iters, warm, steady, len(KERNELS), acc, DTNAME,
        COMPILE_MS[0], FIRST_MS[0] if FIRST_MS[0] is not None else -1.0))


def timed(run):
    t0 = time.time(); run(20); warm = time.time() - t0
    t0 = time.time(); acc = run(iters); steady = (time.time() - t0) / iters * 1e6
    report(warm, steady, acc)


MODELS = {6: run_mlp, 11: run_reduction, 12: run_matmul, 13: run_attn}

if variant in MODELS:
    timed(MODELS[variant]())
    sys.exit(0)
if variant > 5:
    print("triton_bench.py: no handwritten kernel for variant %d" % variant,
          file=sys.stderr)
    sys.exit(2)

w = torch.tensor([(i % 7) - 3.0 for i in range(n)], dtype=DT, device=dev)
b = torch.full((n,), 0.5, dtype=DT, device=dev)
sink = open(os.devnull, "w")


def step(h, b, i):
    h = first(chain, h, b, k)
    if variant in (1, 2):
        if i % 7 == 0:
            h = binop(h, b, False)
    elif variant == 3:
        if h.sum().item() > 0.0:
            h = binop(h, b, False)
    elif variant == 4:
        try:
            if i % 5 == 0:
                raise ValueError
            h = binop(h, b, False)
        except ValueError:
            h = binop(h, b, True)
    elif variant == 5:
        if i % 50 == 0:
            print("step", file=sink)
        h = binop(h, b, False)
    return h


def run(iters):
    h = w
    for i in range(iters):
        h = step(h, b, i)
    torch.cuda.synchronize()
    return h.sum().item()


timed(run)
