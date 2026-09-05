import sys, time, _tensor
n, iters = int(sys.argv[1]), int(sys.argv[2])
w = _tensor.tensor([float((i % 7) - 3) for i in range(n)])
b = _tensor.tensor([0.5] * n)
def run(h, iters):
    for i in range(iters):
        h = (h * b + b).relu()
        h = (h * b + b).relu()
        h = (h * b + b).relu()
        h = (h * b + b).relu()
    return h
run(w, 30)
t0 = time.time()
h = run(w, iters)
s = h.sum().item()
print("applevel-unrolled n=%d iters=%d steady_us=%.1f checksum=%.6f" % (n, iters, (time.time() - t0) / iters * 1e6, s))
