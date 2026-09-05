import sys, time, torch
mode, variant, k, n, iters = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5])
dev = "cuda"
w = torch.tensor([(i % 7) - 3.0 for i in range(n)], dtype=torch.float64, device=dev)
b = torch.full((n,), 0.5, dtype=torch.float64, device=dev)

def step(h, b, i):
    for _ in range(k):
        h = torch.relu(h * b + b)
    if variant in (1, 2):
        if i % 7 == 0:
            h = h + b
    return h

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
print("torch-%s %d %d %d %d %f %f 0 %f" % (mode, variant, k, n, iters, warm, steady, acc))
