import sys, time, torch
mode, variant, k, n, iters = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5])
dev = "cuda"

MLP_D = 256

def mlp_layer():
    W = torch.tensor([float((i * 7) % 13 - 6) / MLP_D for i in range(MLP_D * MLP_D)],
                     dtype=torch.float64, device=dev).reshape(MLP_D, MLP_D)
    b = torch.full((MLP_D,), 0.01, dtype=torch.float64, device=dev)
    return W, b

def run_mlp(iters):
    rows = n // MLP_D
    if rows <= 0:
        rows = 1
    x = torch.tensor([(i % 7) - 3.0 for i in range(rows * MLP_D)],
                     dtype=torch.float64, device=dev).reshape(rows, MLP_D)
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

if variant == 6:
    t0 = time.time(); run_mlp(20); warm = time.time() - t0
    t0 = time.time(); acc = run_mlp(iters); steady = (time.time() - t0) / iters * 1e6
    print("torch-%s %d %d %d %d %f %f 0 %f 0 -1 -1" % (mode, variant, k, n, iters,
        warm, steady, acc))
    sys.exit(0)

w = torch.tensor([(i % 7) - 3.0 for i in range(n)], dtype=torch.float64, device=dev)
b = torch.full((n,), 0.5, dtype=torch.float64, device=dev)

import os
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
print("torch-%s %d %d %d %d %f %f 0 %f 0 %d %d" % (mode, variant, k, n, iters, warm, steady, acc, graphs, breaks))
