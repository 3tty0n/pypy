import math, sys, time, torch
import torch.nn.functional as F
mode, variant, k, n, iters = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5])
dev = "cuda"
import os
DTNAME = os.environ.get("TORCH_DTYPE", "float64")
DT = {"float64": torch.float64, "float32": torch.float32,
      "float16": torch.float16}[DTNAME]

MLP_D = 256
LR = 1e-06
TB_D = 64
TB_H = 4
TB_EPS = 1e-05
TF_BLOCKS = 2

def mlp_layer():
    W = torch.tensor([float((i * 7) % 13 - 6) / MLP_D for i in range(MLP_D * MLP_D)],
                     dtype=DT, device=dev).reshape(MLP_D, MLP_D)
    b = torch.full((MLP_D,), 0.01, dtype=DT, device=dev)
    return W, b

def run_mlp(iters):
    rows = n // MLP_D
    if rows <= 0:
        rows = 1
    x = torch.tensor([(i % 7) - 3.0 for i in range(rows * MLP_D)],
                     dtype=DT, device=dev).reshape(rows, MLP_D)
    layers = [mlp_layer(), mlp_layer(), mlp_layer()]
    def forward(y):
        for W, b in layers:
            y = torch.relu(y @ W + b)
        return y
    if mode == "compile":
        forward = torch.compile(forward, dynamic=False)
    h = x
    for _ in range(iters):
        h = forward(h)
    torch.cuda.synchronize()
    return h.sum().item()

def run_mlp_train(iters):
    rows = n // MLP_D
    if rows <= 0:
        rows = 1
    x = torch.tensor([(i % 7) - 3.0 for i in range(rows * MLP_D)],
                     dtype=DT, device=dev).reshape(rows, MLP_D)
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
    if mode == "compile":
        step = torch.compile(step, dynamic=False)
    loss = 0.0
    for _ in range(iters):
        out = step()
        out.backward()
        with torch.no_grad():
            for p in params:
                p -= LR * p.grad
                p.grad = None
        loss = out.item()
    torch.cuda.synchronize()
    return loss

def tb_weight(rows, cols):
    return torch.tensor([float((i * 7) % 13 - 6) / TB_D for i in range(rows * cols)],
                        dtype=DT, device=dev).reshape(rows, cols)

def run_block(iters):
    rows = n // TB_D
    if rows <= 0:
        rows = 1
    x = torch.tensor([(i % 7) - 3.0 for i in range(rows * TB_D)],
                     dtype=DT, device=dev).reshape(rows, TB_D)
    dh = TB_D // TB_H
    heads = [(tb_weight(TB_D, dh), tb_weight(TB_D, dh), tb_weight(TB_D, dh),
              tb_weight(dh, TB_D)) for _ in range(TB_H)]
    mlp = [(tb_weight(TB_D, TB_D),
            torch.full((TB_D,), 0.01, dtype=DT, device=dev))
           for _ in range(2)]
    g = torch.ones(TB_D, dtype=DT, device=dev)
    b = torch.zeros(TB_D, dtype=DT, device=dev)
    scale = 1.0 / math.sqrt(dh)

    def ln(t):
        mu = t.mean(1, keepdim=True)
        var = ((t - mu) * (t - mu)).mean(1, keepdim=True)
        return (t - mu) / torch.sqrt(var + TB_EPS) * g + b

    WQ = torch.cat([hd[0] for hd in heads], dim=1)
    WK = torch.cat([hd[1] for hd in heads], dim=1)
    WV = torch.cat([hd[2] for hd in heads], dim=1)
    WO = torch.cat([hd[3] for hd in heads], dim=0)

    def split(t):
        return t.reshape(rows, TB_H, dh).transpose(0, 1)

    def forward(t):
        h = ln(t)
        q, k, v = split(h @ WQ), split(h @ WK), split(h @ WV)
        p = torch.softmax(torch.bmm(q, k.transpose(1, 2)) * scale, dim=2)
        o = torch.bmm(p, v).transpose(0, 1).reshape(rows, TB_D)
        t = t + o @ WO
        y = ln(t)
        for W, bb in mlp:
            y = torch.relu(y @ W + bb)
        return t + y

    step = torch.compile(forward, dynamic=False) if mode == "compile" else forward
    for _ in range(iters):
        x = step(x)
    torch.cuda.synchronize()
    return x.sum().item()

def run_transformer_train(iters):
    rows = n // TB_D
    if rows <= 0:
        rows = 1
    x = torch.tensor([(i % 7) - 3.0 for i in range(rows * TB_D)],
                     dtype=DT, device=dev).reshape(rows, TB_D)
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
        mlp = [(param(tb_weight(TB_D, TB_D)),
                param(torch.full((TB_D,), 0.01, dtype=DT,
                                 device=dev)))
               for _ in range(2)]
        blocks.append((WQ, WK, WV, WO, g1, b1, g2, b2, mlp))
    HW = param(tb_weight(TB_D, TB_D))
    HB = param(torch.full((TB_D,), 0.01, dtype=DT, device=dev))

    def ln(t, g, b):
        mu = t.mean(1, keepdim=True)
        var = ((t - mu) * (t - mu)).mean(1, keepdim=True)
        return (t - mu) / torch.sqrt(var + TB_EPS) * g + b

    def split(t):
        return t.reshape(rows, TB_H, dh).transpose(0, 1)

    def block(t, bl):
        WQ, WK, WV, WO, g1, b1, g2, b2, mlp = bl
        h = ln(t, g1, b1)
        q, k, v = split(h @ WQ), split(h @ WK), split(h @ WV)
        p = torch.softmax(torch.bmm(q, k.transpose(1, 2)) * scale, dim=2)
        o = torch.bmm(p, v).transpose(0, 1).reshape(rows, TB_D)
        t = t + o @ WO
        y = ln(t, g2, b2)
        for W, bb in mlp:
            y = torch.relu(y @ W + bb)
        return t + y

    def step():
        y = x
        for bl in blocks:
            y = block(y, bl)
        return torch.relu(y @ HW + HB).sum()

    if mode == "compile":
        step = torch.compile(step, dynamic=False)
    loss = 0.0
    for _ in range(iters):
        out = step()
        out.backward()
        with torch.no_grad():
            for p in params:
                p -= LR * p.grad
                p.grad = None
        loss = out.item()
    torch.cuda.synchronize()
    return loss


CNN_C, CNN_HW, CNN_O, CNN_CLS = 3, 32, 8, 10

def run_cnn(iters):
    pixels = CNN_C * CNN_HW * CNN_HW
    rows = n // pixels
    if rows <= 0:
        rows = 1
    x = torch.tensor([(i % 7) - 3.0 for i in range(rows * pixels)],
                     dtype=DT, device=dev).reshape(rows, CNN_C, CNN_HW, CNN_HW)
    fan = CNN_C * 9
    feat = CNN_O * (CNN_HW // 2) * (CNN_HW // 2)
    wcol = torch.tensor([float((i * 7) % 13 - 6) / fan for i in range(fan * CNN_O)],
                        dtype=DT, device=dev).reshape(fan, CNN_O)
    cw = wcol.t().reshape(CNN_O, CNN_C, 3, 3).contiguous()
    cb = torch.full((CNN_O,), 0.01, dtype=DT, device=dev)
    gamma = torch.ones(CNN_O, dtype=DT, device=dev)
    beta = torch.zeros(CNN_O, dtype=DT, device=dev)
    rmean = torch.zeros(CNN_O, dtype=DT, device=dev)
    rvar = torch.ones(CNN_O, dtype=DT, device=dev)
    wf = torch.tensor([float((i * 7) % 13 - 6) / feat for i in range(feat * CNN_CLS)],
                      dtype=DT, device=dev).reshape(feat, CNN_CLS)
    bf = torch.full((CNN_CLS,), 0.01, dtype=DT, device=dev)

    def forward(t):
        y = torch.conv2d(t, cw, cb, padding=1)
        y = F.batch_norm(y, rmean, rvar, gamma, beta, False, 0.0, TB_EPS)
        y = torch.max_pool2d(torch.relu(y), 2).reshape(rows, feat)
        return torch.relu(y @ wf + bf)

    step = torch.compile(forward, dynamic=False) if mode == "compile" else forward
    acc = 0.0
    with torch.no_grad():
        for _ in range(iters):
            acc += step(x).sum().item()
    torch.cuda.synchronize()
    return acc

if variant == 10:
    t0 = time.time(); run_transformer_train(20); warm = time.time() - t0
    t0 = time.time(); acc = run_transformer_train(iters); steady = (time.time() - t0) / iters * 1e6
    print("torch-%s %d %d %d %d %f %f 0 %f 0 -1 -1 %s" % (mode, variant, k, n, iters,
        warm, steady, acc, DTNAME))
    sys.exit(0)

if variant == 9:
    t0 = time.time(); run_cnn(20); warm = time.time() - t0
    t0 = time.time(); acc = run_cnn(iters); steady = (time.time() - t0) / iters * 1e6
    print("torch-%s %d %d %d %d %f %f 0 %f 0 -1 -1 %s" % (mode, variant, k, n, iters,
        warm, steady, acc, DTNAME))
    sys.exit(0)

if variant == 8:
    t0 = time.time(); run_block(20); warm = time.time() - t0
    t0 = time.time(); acc = run_block(iters); steady = (time.time() - t0) / iters * 1e6
    print("torch-%s %d %d %d %d %f %f 0 %f 0 -1 -1 %s" % (mode, variant, k, n, iters,
        warm, steady, acc, DTNAME))
    sys.exit(0)

if variant == 7:
    t0 = time.time(); run_mlp_train(20); warm = time.time() - t0
    t0 = time.time(); acc = run_mlp_train(iters); steady = (time.time() - t0) / iters * 1e6
    print("torch-%s %d %d %d %d %f %f 0 %f 0 -1 -1 %s" % (mode, variant, k, n, iters,
        warm, steady, acc, DTNAME))
    sys.exit(0)

if variant == 6:
    t0 = time.time(); run_mlp(20); warm = time.time() - t0
    t0 = time.time(); acc = run_mlp(iters); steady = (time.time() - t0) / iters * 1e6
    print("torch-%s %d %d %d %d %f %f 0 %f 0 -1 -1 %s" % (mode, variant, k, n, iters,
        warm, steady, acc, DTNAME))
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

if mode == "compile":
    step = torch.compile(step, dynamic=False)

def run(iters):
    h = w
    for i in range(iters):
        h = step(h, b, i)
    torch.cuda.synchronize()
    return h.sum().item()

t0 = time.time(); run(20); warm = time.time() - t0
t0 = time.time(); acc = run(iters); steady = (time.time() - t0) / iters * 1e6
print("torch-%s %d %d %d %d %f %f 0 %f 0 %d %d %s" % (mode, variant, k, n, iters, warm, steady, acc, graphs, breaks, DTNAME))
