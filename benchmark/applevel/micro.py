"""App-level twin of metatensor-bench for the headline microbenchmark points.

Takes the same argv and prints the same line, so the rows land in micro.tsv
next to the standalone ones.  What separates the two is the PyPy interpreter:
the standalone binary is a translated RPython program with no bytecode
dispatch, this one runs the same models as Python through _metatensor, which
is what torch_bench.py is compared against.

    micro.py MODE VARIANT K N ITERS      (VARIANT: 0, 8, 9, 13)
"""
import sys, os, time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import _metatensor
import cnn as cnn_app
import transformer as tf_app

ATT_D, ATT_H = 256, 8
WARMUP_ITERS = 10

VARIANTS = "0, 8, 9, 13"


def chain(k, n):
    h0 = _metatensor.tensor([float((i % 7) - 3) for i in range(n)])
    b = _metatensor.tensor([0.5] * n)

    def run(iters):
        h = h0
        for i in range(iters):
            for j in range(k):
                h = (h * b + b).relu()
        return h.sum().item()
    return run, n


def block(n):
    rows = max(1, n // tf_app.TB_D)
    model = tf_app.make_block()
    x0 = tf_app.make_input(rows, tf_app.TB_D)

    def run(iters):
        x = x0
        for i in range(iters):
            x = model(x)
        return x.sum().item()
    return run, rows * tf_app.TB_D


def attn(n):
    rows = max(1, n // ATT_D)
    model = tf_app.make_attn(ATT_D, ATT_H)
    x0 = tf_app.make_input(rows, ATT_D)

    def run(iters):
        x = x0
        for i in range(iters):
            x = model(x)
        return x.sum().item()
    return run, rows * ATT_D


def conv(n):
    pixels = cnn_app.CNN_C * cnn_app.CNN_HW * cnn_app.CNN_HW
    rows = max(1, n // pixels)
    model = cnn_app.make_cnn()
    x = cnn_app.make_input(rows, pixels)

    def run(iters):
        acc = 0.0
        for i in range(iters):
            acc += model(x).sum().item()
        return acc
    return run, rows * pixels


def build(variant, k, n):
    if variant == 0:
        return chain(k, n)
    if variant == 8:
        return block(n)
    if variant == 9:
        return conv(n)
    if variant == 13:
        return attn(n)
    raise SystemExit("micro.py: variant %d not implemented app-level (have %s)"
                     % (variant, VARIANTS))


def main(argv):
    if len(argv) != 6:
        print("usage: micro.py MODE VARIANT K N ITERS  (VARIANT: %s)" % VARIANTS)
        return 1
    mode, variant, k = argv[1], int(argv[2]), int(argv[3])
    n, iters = int(argv[4]), int(argv[5])
    dtype = os.environ.get('RTENSOR_DTYPE', 'float64')
    if dtype != 'float64':
        print("micro.py: only float64 app-level, got %s" % dtype)
        return 1

    run, eff_n = build(variant, k, n)
    t0 = time.time()
    run(WARMUP_ITERS)
    warm = time.time() - t0
    run(WARMUP_ITERS)

    kernels_before = _metatensor.kernel_count()
    launches_before = _metatensor.launch_count()
    t0 = time.time()
    acc = run(iters)
    steady = (time.time() - t0) / iters * 1e6
    kernels_n = _metatensor.kernel_count()
    launches = float(_metatensor.launch_count() - launches_before) / iters

    print("%s %d %d %d %d %f %f %d %f %d %f %s"
          % (mode, variant, k, eff_n, iters, warm, steady, kernels_n, acc,
             kernels_n - kernels_before, launches, dtype))
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv))
