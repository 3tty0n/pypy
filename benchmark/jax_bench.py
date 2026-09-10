"""JAX/XLA twin of torch_bench.py: same argv, same variants, same line.

    jax_bench.py jax VARIANT K N ITERS       (mode iree: same function via IREE)

Every step function is jax.jit-compiled ahead of the timed loop; the line adds
two fields after dtype, compile_ms (lower+compile, no execution) and
first_run_ms (the first executed iteration), so the compile cost is reported
separately from the steady state, which is what the other columns measure.
The Python control-flow variants 1-5 keep their branches in Python, as the
torch.compile twin does: only the tensor expressions are jitted.
"""
import math, os, sys, time

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import iree_adapter

mode, variant, k, n, iters = (sys.argv[1], int(sys.argv[2]), int(sys.argv[3]),
                              int(sys.argv[4]), int(sys.argv[5]))
DTNAME = os.environ.get("TORCH_DTYPE", os.environ.get("RTENSOR_DTYPE", "float64"))
jax.config.update("jax_enable_x64", True)
# torch eager and cuBLAS do full fp32 GEMMs by default; XLA would use TF32.
jax.config.update("jax_default_matmul_precision", "highest")
DT = {"float64": jnp.float64, "float32": jnp.float32,
      "float16": jnp.float16}[DTNAME]

MLP_D = 256
LR = 1e-06
TB_D = 64
TB_H = 4
TB_EPS = 1e-05
TF_BLOCKS = 2
RED_COLS = 64
ATT_D = 256
ATT_H = 8
CNN_C, CNN_HW, CNN_O, CNN_CLS = 3, 32, 8, 10

COMPILE_MS = [0.0]
FIRST_MS = [None]


def rows_of(d):
    return max(n // d, 1)


def dev(a):
    return jax.device_put(jnp.asarray(a, dtype=DT))


def gen_weight(rows, cols, divisor):
    i = np.arange(rows * cols)
    return dev(((i * 7) % 13 - 6) / float(divisor)).reshape(rows, cols)


def tb_weight(rows, cols):
    return gen_weight(rows, cols, TB_D)


def bias(size, value):
    return dev(np.full((size,), value))


def make_input(rows, cols):
    return dev((np.arange(rows * cols) % 7) - 3.0).reshape(rows, cols)


def compiled(fn, *example):
    """jit fn for the example arguments, charging lower+compile to COMPILE_MS.

    mode iree compiles the same function through iree_adapter instead."""
    t0 = time.perf_counter()
    if mode == "iree":
        exe = iree_adapter.compile(fn, *example)
        COMPILE_MS[0] += exe.compile_ms
        return exe
    exe = jax.jit(fn).lower(*example).compile()
    COMPILE_MS[0] += (time.perf_counter() - t0) * 1e3
    return exe


def sync(x):
    if mode == "iree":
        return jax.tree_util.tree_map(
            lambda a: np.asarray(a.to_host()) if hasattr(a, "to_host") else a, x)
    jax.tree_util.tree_map(lambda a: a.block_until_ready(), x)
    return x


def first(fn, *args):
    """Run one call, timing it as the first execution if none was timed yet."""
    if FIRST_MS[0] is None:
        t0 = time.perf_counter()
        out = sync(fn(*args))
        FIRST_MS[0] = (time.perf_counter() - t0) * 1e3
        return out
    return fn(*args)


def runner(step, x0):
    def run(iters):
        x = x0
        for _ in range(iters):
            x = first(step, x)
        return float(sync(x).sum())
    return run


def layer_norm(t, g, b):
    mu = t.mean(1, keepdims=True)
    var = ((t - mu) * (t - mu)).mean(1, keepdims=True)
    return (t - mu) / jnp.sqrt(var + TB_EPS) * g + b


def attention(h, rows, heads, dh, WQ, WK, WV, scale):
    def split(t):
        return t.reshape(rows, heads, dh).transpose(1, 0, 2)
    q, k_, v = split(h @ WQ), split(h @ WK), split(h @ WV)
    p = jax.nn.softmax(jnp.einsum("hqd,hkd->hqk", q, k_) * scale, axis=2)
    return jnp.einsum("hqk,hkd->hqd", p, v).transpose(1, 0, 2).reshape(
        rows, heads * dh)


def mlp_layer():
    return gen_weight(MLP_D, MLP_D, MLP_D), bias(MLP_D, 0.01)


def run_mlp():
    x = make_input(rows_of(MLP_D), MLP_D)
    layers = [mlp_layer(), mlp_layer(), mlp_layer()]

    def forward(y):
        for W, b in layers:
            y = jax.nn.relu(y @ W + b)
        return y
    return runner(compiled(forward, x), x)


def sgd_train(params0, loss_fn):
    """SGD on loss_fn(params), one jitted call per step, from params0 each run."""
    def step(p):
        loss, g = jax.value_and_grad(loss_fn)(p)
        return jax.tree_util.tree_map(lambda w, gw: w - LR * gw, p, g), loss
    step = compiled(step, params0)

    def run(iters):
        params, loss = params0, 0.0
        for _ in range(iters):
            params, loss = first(step, params)
        return float(sync(loss))
    return run


def run_mlp_train():
    x = make_input(rows_of(MLP_D), MLP_D)
    layers = [mlp_layer(), mlp_layer(), mlp_layer()]

    def loss_fn(layers):
        y = x
        for W, b in layers:
            y = jax.nn.relu(y @ W + b)
        return y.sum()
    return sgd_train(layers, loss_fn)


def block_weights(rows, dh):
    heads = [(tb_weight(TB_D, dh), tb_weight(TB_D, dh), tb_weight(TB_D, dh),
              tb_weight(dh, TB_D)) for _ in range(TB_H)]
    WQ = jnp.concatenate([hd[0] for hd in heads], axis=1)
    WK = jnp.concatenate([hd[1] for hd in heads], axis=1)
    WV = jnp.concatenate([hd[2] for hd in heads], axis=1)
    WO = jnp.concatenate([hd[3] for hd in heads], axis=0)
    return WQ, WK, WV, WO


def run_block():
    rows = rows_of(TB_D)
    x = make_input(rows, TB_D)
    dh = TB_D // TB_H
    WQ, WK, WV, WO = block_weights(rows, dh)
    mlp = [(tb_weight(TB_D, TB_D), bias(TB_D, 0.01)) for _ in range(2)]
    g = jnp.ones(TB_D, dtype=DT)
    b = jnp.zeros(TB_D, dtype=DT)
    scale = 1.0 / math.sqrt(dh)

    def forward(t):
        h = layer_norm(t, g, b)
        o = attention(h, rows, TB_H, dh, WQ, WK, WV, scale)
        t = t + o @ WO
        y = layer_norm(t, g, b)
        for W, bb in mlp:
            y = jax.nn.relu(y @ W + bb)
        return t + y
    return runner(compiled(forward, x), x)


def run_transformer_train():
    rows = rows_of(TB_D)
    x = make_input(rows, TB_D)
    dh = TB_D // TB_H
    scale = 1.0 / math.sqrt(dh)
    blocks = []
    for _ in range(TF_BLOCKS):
        WQ, WK, WV, WO = block_weights(rows, dh)
        g1 = jnp.ones(TB_D, dtype=DT)
        b1 = jnp.zeros(TB_D, dtype=DT)
        g2 = jnp.ones(TB_D, dtype=DT)
        b2 = jnp.zeros(TB_D, dtype=DT)
        mlp = [(tb_weight(TB_D, TB_D), bias(TB_D, 0.01)) for _ in range(2)]
        blocks.append((WQ, WK, WV, WO, g1, b1, g2, b2, mlp))
    params = (blocks, tb_weight(TB_D, TB_D), bias(TB_D, 0.01))

    def block(t, bl):
        WQ, WK, WV, WO, g1, b1, g2, b2, mlp = bl
        h = layer_norm(t, g1, b1)
        o = attention(h, rows, TB_H, dh, WQ, WK, WV, scale)
        t = t + o @ WO
        y = layer_norm(t, g2, b2)
        for W, bb in mlp:
            y = jax.nn.relu(y @ W + bb)
        return t + y

    def loss_fn(params):
        blocks, HW, HB = params
        y = x
        for bl in blocks:
            y = block(y, bl)
        return jax.nn.relu(y @ HW + HB).sum()
    return sgd_train(params, loss_fn)


def run_cnn():
    pixels = CNN_C * CNN_HW * CNN_HW
    rows = rows_of(pixels)
    x = make_input(rows, pixels).reshape(rows, CNN_C, CNN_HW, CNN_HW)
    fan = CNN_C * 9
    feat = CNN_O * (CNN_HW // 2) * (CNN_HW // 2)
    wcol = gen_weight(fan, CNN_O, fan)
    cw = wcol.T.reshape(CNN_O, CNN_C, 3, 3)
    cb = bias(CNN_O, 0.01)
    gamma = jnp.ones(CNN_O, dtype=DT)
    beta = jnp.zeros(CNN_O, dtype=DT)
    rmean = jnp.zeros(CNN_O, dtype=DT)
    rvar = jnp.ones(CNN_O, dtype=DT)
    wf = gen_weight(feat, CNN_CLS, feat)
    bf = bias(CNN_CLS, 0.01)

    def forward(t):
        y = jax.lax.conv_general_dilated(t, cw, (1, 1), ((1, 1), (1, 1)))
        y = y + cb[None, :, None, None]
        s = (gamma / jnp.sqrt(rvar + TB_EPS))[None, :, None, None]
        y = (y - rmean[None, :, None, None]) * s + beta[None, :, None, None]
        y = jax.nn.relu(y)
        y = y.reshape(rows, CNN_O, CNN_HW // 2, 2, CNN_HW // 2, 2).max((3, 5))
        y = y.reshape(rows, feat)
        return jax.nn.relu(y @ wf + bf).sum()
    step = compiled(forward, x)

    def run(iters):
        acc = 0.0
        for _ in range(iters):
            acc += float(sync(first(step, x)))
        return acc
    return run


def run_reduction():
    x = make_input(rows_of(RED_COLS), RED_COLS)

    def forward(t):
        h = (t * t).sum(1, keepdims=True)
        return t + h
    return runner(compiled(forward, x), x)


def run_matmul():
    x = make_input(rows_of(MLP_D), MLP_D)
    W, b = mlp_layer()

    def forward(t):
        return jax.nn.relu(t @ W + b)
    return runner(compiled(forward, x), x)


def run_attn():
    rows = rows_of(ATT_D)
    x = make_input(rows, ATT_D)
    dh = ATT_D // ATT_H
    scale = 1.0 / math.sqrt(dh)
    heads = [(gen_weight(ATT_D, dh, ATT_D), gen_weight(ATT_D, dh, ATT_D),
              gen_weight(ATT_D, dh, ATT_D), gen_weight(dh, ATT_D, ATT_D))
             for _ in range(ATT_H)]
    WQ = jnp.concatenate([hd[0] for hd in heads], axis=1)
    WK = jnp.concatenate([hd[1] for hd in heads], axis=1)
    WV = jnp.concatenate([hd[2] for hd in heads], axis=1)
    WO = jnp.concatenate([hd[3] for hd in heads], axis=0)

    def forward(t):
        return attention(t, rows, ATT_H, dh, WQ, WK, WV, scale) @ WO
    return runner(compiled(forward, x), x)


def report(warm, steady, acc):
    print("%s %d %d %d %d %f %f 0 %f 0 -1 -1 %s %.3f %.3f" % (
        mode, variant, k, n, iters, warm, steady, acc, DTNAME,
        COMPILE_MS[0], FIRST_MS[0] if FIRST_MS[0] is not None else -1.0))


def timed(run):
    t0 = time.time(); run(20); warm = time.time() - t0
    t0 = time.time(); acc = run(iters); steady = (time.time() - t0) / iters * 1e6
    report(warm, steady, acc)


MODELS = {6: run_mlp, 7: run_mlp_train, 8: run_block, 9: run_cnn,
          10: run_transformer_train, 11: run_reduction, 12: run_matmul,
          13: run_attn}

if variant in MODELS:
    timed(MODELS[variant]())
    sys.exit(0)

w = dev((np.arange(n) % 7) - 3.0)
b = dev(np.full((n,), 0.5))
sink = open(os.devnull, "w")


def chain(h, b):
    for _ in range(k):
        h = jax.nn.relu(h * b + b)
    return h


chain = compiled(chain, w, b)
add = compiled(lambda h, b: h + b, w, b)
mul = compiled(lambda h, b: h * b, w, b)
total = compiled(lambda h: h.sum(), w)


def step(h, b, i):
    h = first(chain, h, b)
    if variant in (1, 2):
        if i % 7 == 0:
            h = add(h, b)
    elif variant == 3:
        if float(total(h)) > 0.0:
            h = add(h, b)
    elif variant == 4:
        try:
            if i % 5 == 0:
                raise ValueError
            h = add(h, b)
        except ValueError:
            h = mul(h, b)
    elif variant == 5:
        if i % 50 == 0:
            print("step", file=sink)
        h = add(h, b)
    return h


def run(iters):
    h = w
    for i in range(iters):
        h = step(h, b, i)
    return float(sync(h).sum())


timed(run)
