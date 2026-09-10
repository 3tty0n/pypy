"""App-level twin of metatensor-bench for the microbenchmark points.

Takes the same argv and prints the same line, so the rows land in micro.tsv
next to the standalone ones.  What separates the two is the PyPy interpreter:
the standalone binary is a translated RPython program with no bytecode
dispatch, this one runs the same models as Python through _metatensor, which
is what torch_bench.py is compared against.

    micro.py MODE VARIANT K N ITERS      (VARIANT: 0..13, float64 only)
"""
import sys, os, time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import _metatensor
import cnn as cnn_app
import transformer as tf_app
from tensorpypy.optim import sgd_step

ATT_D, ATT_H = 256, 8
MLP_D = 256
LR = 1e-06
RED_COLS = 64
TF_BLOCKS = 2
WARMUP_ITERS = 10

VARIANTS = "0..13"

# Same coefficients as bench/common.py's MEM_* table: rough working-set sizes
# ("tensors of that shape held live at the peak") used to shrink n so the
# variant's memory stays within mem_target().  See that module for how they
# were measured; they only have to be right to within a small factor.
MEM_CHAIN = 6
MEM_MLP, MEM_MLP_TRAIN = 4, 12
MEM_CNN = 8
MEM_REDUCE, MEM_MATMUL = 4, 4
MEM_BLOCK_SQ, MEM_BLOCK_LIN = 10 * tf_app.TB_H, 8
MEM_TF_TRAIN_SQ, MEM_TF_TRAIN_LIN = 22 * tf_app.TB_H * TF_BLOCKS, 24
MEM_ATTN_SQ, MEM_ATTN_LIN = 10 * ATT_H, 8

DTYPE_BYTES = 8  # micro.py is float64-only


class _Cfg(object):
    capped = False
_cfg = _Cfg()


def _int_env(name, default):
    try:
        return int(os.environ.get(name, default))
    except ValueError:
        return int(default)


def mem_target():
    """Device bytes the benchmark is allowed to occupy, 0 when unknown."""
    if _int_env('RTENSOR_MAX_BYTES', '0') > 0:
        return _int_env('RTENSOR_MAX_BYTES', '0')
    total = _metatensor.mem_total()
    if total <= 0:
        return 0
    pct = _int_env('RTENSOR_MEM_FRAC_PCT', '50')
    if pct < 1:
        pct = 1
    elif pct > 95:
        pct = 95
    return total // 100 * pct


def _fits(rows, d, sq, lin, target):
    return DTYPE_BYTES * (sq * rows * rows + lin * rows * d) <= target


def _largest_fitting(rows, d, sq, lin, target):
    if _fits(rows, d, sq, lin, target):
        return rows
    lo, hi = 1, rows
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if _fits(mid, d, sq, lin, target):
            lo = mid
        else:
            hi = mid - 1
    return lo


def fit_rows(n, d, sq, lin):
    """Rows of width d for n elements, shrunk until the working set fits."""
    rows = n // d
    if rows <= 0:
        rows = 1
    target = mem_target()
    if target > 0:
        capped = _largest_fitting(rows, d, sq, lin, target)
        if capped < rows:
            rows = capped
            _cfg.capped = True
    return rows, rows * d


def fit_n(n, units):
    """Element count for the flat (chain) variants, fitted the same way."""
    target = mem_target()
    if target > 0:
        cap = target // (DTYPE_BYTES * units)
        if cap < 1:
            cap = 1
        if n > cap:
            n = cap
            _cfg.capped = True
    return n


_sink_fd = [-1]


def _sink():
    if _sink_fd[0] < 0:
        _sink_fd[0] = os.open('/dev/null', os.O_WRONLY, 0)
    return _sink_fd[0]


def chain(variant, k, n):
    n = fit_n(n, MEM_CHAIN)
    h0 = _metatensor.tensor([float((i % 7) - 3) for i in range(n)])
    b = _metatensor.tensor([0.5] * n)

    def run(iters):
        h = h0
        for i in range(iters):
            for j in range(k):
                h = (h * b + b).relu()
            if variant == 1:
                if i % 7 == 0:
                    h = h + b
            elif variant == 2:
                h = h.force()
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
                    os.write(_sink(), "step\n")
                h = h + b
        return h.sum().item()
    return run, n


def mlp_layer(requires_grad=False):
    w = tf_app.tb_weight(MLP_D, MLP_D, MLP_D, requires_grad)
    b = tf_app.tb_vector(0.01, MLP_D, requires_grad)
    return tf_app.nn.Linear(w, b)


def make_mlp(requires_grad=False):
    return tf_app.MLP([mlp_layer(requires_grad) for _ in range(3)])


def mlp_forward(n):
    rows, eff_n = fit_rows(n, MLP_D, 0, MEM_MLP)
    model = make_mlp()
    x0 = tf_app.make_input(rows, MLP_D)

    def run(iters):
        x = x0
        for i in range(iters):
            x = model(x)
        return x.sum().item()
    return run, eff_n


def mlp_train(n):
    # Standalone rebuilds the model fresh inside run_mlp_train on every call
    # (the JIT-warmup calls each train and discard their own model), so the
    # reported acc is a from-scratch model trained for exactly `iters` steps.
    # Building it here instead of once outside would let warmup steps leak
    # SGD updates into the measured run.
    rows, eff_n = fit_rows(n, MLP_D, 0, MEM_MLP_TRAIN)

    def run(iters):
        model = make_mlp(True)
        x0 = tf_app.make_input(rows, MLP_D)
        params = model.parameters()
        loss = 0.0
        for i in range(iters):
            out = model(x0).sum()
            out.backward()
            sgd_step(params, LR)
            loss = out.item()
        return loss
    return run, eff_n


def block(n):
    rows, eff_n = fit_rows(n, tf_app.TB_D, MEM_BLOCK_SQ, MEM_BLOCK_LIN)
    model = tf_app.make_block()
    x0 = tf_app.make_input(rows, tf_app.TB_D)

    def run(iters):
        x = x0
        for i in range(iters):
            x = model(x)
        return x.sum().item()
    return run, eff_n


def transformer_train(n):
    # Same rebuild-per-call reasoning as mlp_train: standalone's
    # run_transformer_train constructs the model from scratch every call.
    rows, eff_n = fit_rows(n, tf_app.TB_D, MEM_TF_TRAIN_SQ, MEM_TF_TRAIN_LIN)

    def run(iters):
        blocks = [tf_app.make_train_block() for _ in range(TF_BLOCKS)]
        head = tf_app.nn.Linear(
            tf_app.tb_weight(tf_app.TB_D, tf_app.TB_D, tf_app.TB_D, True),
            tf_app.tb_vector(0.01, tf_app.TB_D, True))
        x0 = tf_app.make_input(rows, tf_app.TB_D)
        params = []
        for blk in blocks:
            params.extend(blk.parameters())
        params.extend(head.parameters())
        loss = 0.0
        for i in range(iters):
            x = x0
            for blk in blocks:
                x = blk(x)
            out = head(x).sum()
            out.backward()
            sgd_step(params, LR)
            loss = out.item()
        return loss
    return run, eff_n


def attn(n):
    rows, eff_n = fit_rows(n, ATT_D, MEM_ATTN_SQ, MEM_ATTN_LIN)
    model = tf_app.make_attn(ATT_D, ATT_H)
    x0 = tf_app.make_input(rows, ATT_D)

    def run(iters):
        x = x0
        for i in range(iters):
            x = model(x)
        return x.sum().item()
    return run, eff_n


def conv(n):
    pixels = cnn_app.CNN_C * cnn_app.CNN_HW * cnn_app.CNN_HW
    rows, eff_n = fit_rows(n, pixels, 0, MEM_CNN)
    model = cnn_app.make_cnn()
    x0 = cnn_app.make_input(rows, pixels)

    def run(iters):
        acc = 0.0
        for i in range(iters):
            acc += model(x0).sum().item()
        return acc
    return run, eff_n


def reduction(n):
    rows, eff_n = fit_rows(n, RED_COLS, 0, MEM_REDUCE)
    data = [(i % 7) - 3.0 for i in range(rows * RED_COLS)]
    x0 = _metatensor.tensor(data, [rows, RED_COLS])

    def run(iters):
        x = x0
        for i in range(iters):
            h = (x * x).sum(1).reshape([rows, 1])
            x = x + h
        return x.sum().item()
    return run, eff_n


def matmul_variant(n):
    rows, eff_n = fit_rows(n, MLP_D, 0, MEM_MATMUL)
    layer = mlp_layer()
    x0 = tf_app.make_input(rows, MLP_D)

    def run(iters):
        x = x0
        for i in range(iters):
            x = layer(x)
        return x.sum().item()
    return run, eff_n


def build(variant, k, n):
    if variant <= 5:
        return chain(variant, k, n)
    if variant == 6:
        return mlp_forward(n)
    if variant == 7:
        return mlp_train(n)
    if variant == 8:
        return block(n)
    if variant == 9:
        return conv(n)
    if variant == 10:
        return transformer_train(n)
    if variant == 11:
        return reduction(n)
    if variant == 12:
        return matmul_variant(n)
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

    _cfg.capped = False
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

    # Only report a different n when the working set actually had to shrink;
    # the plain floor of n // d is left alone so rows stay comparable with
    # earlier results.
    report_n = n
    if _cfg.capped:
        report_n = eff_n
        sys.stderr.write("micro.py: variant %d n %d -> %d "
                          "(fitted to GPU memory)\n" % (variant, n, report_n))

    print("%s %d %d %d %d %f %f %d %f %d %f %s"
          % (mode, variant, k, report_n, iters, warm, steady, kernels_n, acc,
             kernels_n - kernels_before, launches, dtype))
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv))
