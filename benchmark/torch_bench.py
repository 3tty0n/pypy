import math, os, sys, time, torch
import torch.nn.functional as F

mode, variant, k, n, iters = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5])
dev = "cuda"
DTNAME = os.environ.get("TORCH_DTYPE", "float64")
DT = {"float64": torch.float64, "float32": torch.float32,
      "float16": torch.float16}[DTNAME]

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


def rows_of(d):
    return max(n // d, 1)


def gen_weight(rows, cols, divisor):
    return torch.tensor([float((i * 7) % 13 - 6) / divisor for i in range(rows * cols)],
                        dtype=DT, device=dev).reshape(rows, cols)


def tb_weight(rows, cols):
    return gen_weight(rows, cols, TB_D)


def bias(size, value):
    return torch.full((size,), value, dtype=DT, device=dev)


def make_input(rows, cols):
    return torch.tensor([(i % 7) - 3.0 for i in range(rows * cols)],
                        dtype=DT, device=dev).reshape(rows, cols)


def compiled(fn):
    return torch.compile(fn, dynamic=False) if mode == "compile" else fn


def loop(step, x, iters):
    for _ in range(iters):
        x = step(x)
    torch.cuda.synchronize()
    return x.sum().item()


def layer_norm(t, g, b):
    mu = t.mean(1, keepdim=True)
    var = ((t - mu) * (t - mu)).mean(1, keepdim=True)
    return (t - mu) / torch.sqrt(var + TB_EPS) * g + b


def attention(h, rows, heads, dh, WQ, WK, WV, scale):
    def split(t):
        return t.reshape(rows, heads, dh).transpose(0, 1)
    q, k_, v = split(h @ WQ), split(h @ WK), split(h @ WV)
    p = torch.softmax(torch.bmm(q, k_.transpose(1, 2)) * scale, dim=2)
    return torch.bmm(p, v).transpose(0, 1).reshape(rows, heads * dh)


def sgd(params, out):
    out.backward()
    with torch.no_grad():
        for p in params:
            p -= LR * p.grad
            p.grad = None
    return out.item()


def mlp_layer():
    return gen_weight(MLP_D, MLP_D, MLP_D), bias(MLP_D, 0.01)


def run_mlp(iters):
    x = make_input(rows_of(MLP_D), MLP_D)
    layers = [mlp_layer(), mlp_layer(), mlp_layer()]

    def forward(y):
        for W, b in layers:
            y = torch.relu(y @ W + b)
        return y
    return loop(compiled(forward), x, iters)


def run_mlp_train(iters):
    x = make_input(rows_of(MLP_D), MLP_D)
    layers = [mlp_layer(), mlp_layer(), mlp_layer()]
    params = []
    for W, b in layers:
        W.requires_grad_(True)
        b.requires_grad_(True)
        params.append(W)
        params.append(b)

    def step():
        y = x
        for W, b in layers:
            y = torch.relu(y @ W + b)
        return y.sum()
    step = compiled(step)
    loss = 0.0
    for _ in range(iters):
        loss = sgd(params, step())
    torch.cuda.synchronize()
    return loss


def run_block(iters):
    rows = rows_of(TB_D)
    x = make_input(rows, TB_D)
    dh = TB_D // TB_H
    heads = [(tb_weight(TB_D, dh), tb_weight(TB_D, dh), tb_weight(TB_D, dh),
              tb_weight(dh, TB_D)) for _ in range(TB_H)]
    mlp = [(tb_weight(TB_D, TB_D), bias(TB_D, 0.01)) for _ in range(2)]
    g = torch.ones(TB_D, dtype=DT, device=dev)
    b = torch.zeros(TB_D, dtype=DT, device=dev)
    scale = 1.0 / math.sqrt(dh)
    WQ = torch.cat([hd[0] for hd in heads], dim=1)
    WK = torch.cat([hd[1] for hd in heads], dim=1)
    WV = torch.cat([hd[2] for hd in heads], dim=1)
    WO = torch.cat([hd[3] for hd in heads], dim=0)

    def forward(t):
        h = layer_norm(t, g, b)
        o = attention(h, rows, TB_H, dh, WQ, WK, WV, scale)
        t = t + o @ WO
        y = layer_norm(t, g, b)
        for W, bb in mlp:
            y = torch.relu(y @ W + bb)
        return t + y
    return loop(compiled(forward), x, iters)


def run_transformer_train(iters):
    rows = rows_of(TB_D)
    x = make_input(rows, TB_D)
    dh = TB_D // TB_H
    scale = 1.0 / math.sqrt(dh)
    params = []

    def param(t):
        t.requires_grad_(True)
        params.append(t)
        return t

    blocks = []
    for _ in range(TF_BLOCKS):
        WQ = param(torch.cat([tb_weight(TB_D, dh) for _ in range(TB_H)], dim=1))
        WK = param(torch.cat([tb_weight(TB_D, dh) for _ in range(TB_H)], dim=1))
        WV = param(torch.cat([tb_weight(TB_D, dh) for _ in range(TB_H)], dim=1))
        WO = param(torch.cat([tb_weight(dh, TB_D) for _ in range(TB_H)], dim=0))
        g1 = param(torch.ones(TB_D, dtype=DT, device=dev))
        b1 = param(torch.zeros(TB_D, dtype=DT, device=dev))
        g2 = param(torch.ones(TB_D, dtype=DT, device=dev))
        b2 = param(torch.zeros(TB_D, dtype=DT, device=dev))
        mlp = [(param(tb_weight(TB_D, TB_D)), param(bias(TB_D, 0.01)))
               for _ in range(2)]
        blocks.append((WQ, WK, WV, WO, g1, b1, g2, b2, mlp))
    HW = param(tb_weight(TB_D, TB_D))
    HB = param(bias(TB_D, 0.01))

    def block(t, bl):
        WQ, WK, WV, WO, g1, b1, g2, b2, mlp = bl
        h = layer_norm(t, g1, b1)
        o = attention(h, rows, TB_H, dh, WQ, WK, WV, scale)
        t = t + o @ WO
        y = layer_norm(t, g2, b2)
        for W, bb in mlp:
            y = torch.relu(y @ W + bb)
        return t + y

    def step():
        y = x
        for bl in blocks:
            y = block(y, bl)
        return torch.relu(y @ HW + HB).sum()
    step = compiled(step)
    loss = 0.0
    for _ in range(iters):
        loss = sgd(params, step())
    torch.cuda.synchronize()
    return loss


def run_cnn(iters):
    pixels = CNN_C * CNN_HW * CNN_HW
    rows = rows_of(pixels)
    x = make_input(rows, pixels).reshape(rows, CNN_C, CNN_HW, CNN_HW)
    fan = CNN_C * 9
    feat = CNN_O * (CNN_HW // 2) * (CNN_HW // 2)
    wcol = gen_weight(fan, CNN_O, fan)
    cw = wcol.t().reshape(CNN_O, CNN_C, 3, 3).contiguous()
    cb = bias(CNN_O, 0.01)
    gamma = torch.ones(CNN_O, dtype=DT, device=dev)
    beta = torch.zeros(CNN_O, dtype=DT, device=dev)
    rmean = torch.zeros(CNN_O, dtype=DT, device=dev)
    rvar = torch.ones(CNN_O, dtype=DT, device=dev)
    wf = gen_weight(feat, CNN_CLS, feat)
    bf = bias(CNN_CLS, 0.01)

    def forward(t):
        y = torch.conv2d(t, cw, cb, padding=1)
        y = F.batch_norm(y, rmean, rvar, gamma, beta, False, 0.0, TB_EPS)
        y = torch.max_pool2d(torch.relu(y), 2).reshape(rows, feat)
        return torch.relu(y @ wf + bf)
    step = compiled(forward)
    acc = 0.0
    with torch.no_grad():
        for _ in range(iters):
            acc += step(x).sum().item()
    torch.cuda.synchronize()
    return acc


def run_reduction(iters):
    x = make_input(rows_of(RED_COLS), RED_COLS)

    def forward(t):
        h = (t * t).sum(1, keepdim=True)
        return t + h
    return loop(compiled(forward), x, iters)


def run_matmul(iters):
    x = make_input(rows_of(MLP_D), MLP_D)
    W, b = mlp_layer()

    def forward(t):
        return torch.relu(t @ W + b)
    return loop(compiled(forward), x, iters)


def run_attn(iters):
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
        return attention(t, rows, ATT_H, dh, WQ, WK, WV, scale) @ WO
    return loop(compiled(forward), x, iters)


FIRST_MS = [None]


def report(warm, steady, acc, graphs=-1, breaks=-1):
    # compile_ms is -1: torch.compile compiles inside the first call, so
    # first_run_ms is compile plus one iteration and the split is not known.
    print("torch-%s %d %d %d %d %f %f 0 %f 0 %d %d %s -1 %.3f" % (
        mode, variant, k, n, iters, warm, steady, acc, graphs, breaks, DTNAME,
        FIRST_MS[0] if FIRST_MS[0] is not None else -1.0))


def timed(run):
    t0 = time.time(); run(1); torch.cuda.synchronize()
    FIRST_MS[0] = (time.time() - t0) * 1e3
    t0 = time.time(); run(19); warm = time.time() - t0 + FIRST_MS[0] / 1e3
    t0 = time.time(); acc = run(iters); steady = (time.time() - t0) / iters * 1e6
    return warm, steady, acc


MODELS = {6: run_mlp, 7: run_mlp_train, 8: run_block, 9: run_cnn,
          10: run_transformer_train, 11: run_reduction, 12: run_matmul,
          13: run_attn}

if variant in MODELS:
    warm, steady, acc = timed(MODELS[variant])
    report(warm, steady, acc)
    sys.exit(0)

w = torch.tensor([(i % 7) - 3.0 for i in range(n)], dtype=DT, device=dev)
b = torch.full((n,), 0.5, dtype=DT, device=dev)

sink = open(os.devnull, "w")


def step(h, b, i):
    for _ in range(k):
        h = torch.relu(h * b + b)
    if variant in (1, 2):
        if i % 7 == 0:
            h = h + b
    elif variant == 3:
        if h.sum().item() > 0.0:
            h = h + b
    elif variant == 4:
        try:
            if i % 5 == 0:
                raise ValueError
            h = h + b
        except ValueError:
            h = h * b
    elif variant == 5:
        if i % 50 == 0:
            print("step", file=sink)
        h = h + b
    return h


graphs = breaks = -1
if mode == "compile":
    import torch._dynamo
    ex = torch._dynamo.explain(step)(w, b, 0)
    graphs, breaks = ex.graph_count, ex.graph_break_count
    step = torch.compile(step, dynamic=False)


def run(iters):
    h = w
    for i in range(iters):
        h = step(h, b, i)
    torch.cuda.synchronize()
    return h.sum().item()


warm, steady, acc = timed(run)
report(warm, steady, acc, graphs, breaks)
