import os
from rpython.metatensor import core, device, nn

class Sink(object):
    fd = -1
sink = Sink()

class Cfg(object):
    dtype = core.F64
    # The element count the benchmark actually ran with, after the working set
    # has been fitted to the GPU (see fit_n / fit_rows).  entry_point reports
    # this instead of the requested n so the TSV records what was measured.
    eff_n = 0
    capped = False
cfg = Cfg()

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

def zeros(shape):
    return core.zeros(shape, cfg.dtype)

# Per-variant working-set models, in "tensors of that shape held live at the
# peak".  sq counts rows x rows tensors (attention scores and their gradients,
# summed over heads and blocks), lin counts rows x d tensors (activations,
# saved values and gradients).  They only have to be right to within a small
# factor: they set the size at which a variant is allowed to run, not what it
# computes.
#
# The quadratic coefficients come from peak device memory measured on this
# harness in float64, minus the ~250 MB a bare context and the kernels take:
#
#   variant 8  rows=4000   4522 MB   -> 9.3 per head
#   variant 13 rows=1000    638 MB   -> 10.5 per head
#   variant 10 rows=400     210 MB   -> 21.5 per head and block
#   variant 10 rows=4000    OOM at 14.9 GB live, consistent with 22 GB needed
#
# so a forward block holds about ten rows x rows tensors per head, and the
# training variant about twice that per head and block for the saved values
# and their gradients.
MEM_CHAIN = 6
MEM_MLP, MEM_MLP_TRAIN = 4, 12
MEM_CNN = 8
MEM_REDUCE, MEM_MATMUL = 4, 4
MEM_BLOCK_SQ, MEM_BLOCK_LIN = 10 * TB_H, 8
MEM_TF_TRAIN_SQ, MEM_TF_TRAIN_LIN = 22 * TB_H * TF_BLOCKS, 24
MEM_ATTN_SQ, MEM_ATTN_LIN = 10 * ATT_H, 8

def _int_env(name, default):
    try:
        return int(bench_env(name, default))
    except ValueError:
        return int(default)

def mem_target():
    """Device bytes the benchmark is allowed to occupy, 0 when unknown."""
    if _int_env('RTENSOR_MAX_BYTES', '0') > 0:
        return _int_env('RTENSOR_MAX_BYTES', '0')
    total = device.mem_total()
    if total <= 0:
        return 0
    pct = _int_env('RTENSOR_MEM_FRAC_PCT', '50')
    if pct < 1:
        pct = 1
    elif pct > 95:
        pct = 95
    return total // 100 * pct

def _fits(rows, d, sq, lin, item, target):
    return item * (sq * rows * rows + lin * rows * d) <= target

def _largest_fitting(rows, d, sq, lin, item, target):
    if _fits(rows, d, sq, lin, item, target):
        return rows
    lo, hi = 1, rows
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if _fits(mid, d, sq, lin, item, target):
            lo = mid
        else:
            hi = mid - 1
    return lo

def fit_rows(n, d, sq, lin):
    """Rows of width d for n elements, shrunk until the working set fits.

    sq/lin are the per-variant MEM_* coefficients above.  Variants whose cost
    is quadratic in the row count (attention) pass sq > 0; without a cap they
    outgrow any GPU as n rises.
    """
    rows = n // d
    if rows <= 0:
        rows = 1
    target = mem_target()
    if target > 0:
        item = core.DTYPE_BYTES[cfg.dtype]
        capped = _largest_fitting(rows, d, sq, lin, item, target)
        if capped < rows:
            rows = capped
            cfg.capped = True
    cfg.eff_n = rows * d
    return rows

def fit_n(n, units):
    """Element count for the flat (chain) variants, fitted the same way."""
    target = mem_target()
    if target > 0:
        item = core.DTYPE_BYTES[cfg.dtype]
        cap = target // (item * units)
        if cap < 1:
            cap = 1
        if n > cap:
            n = cap
            cfg.capped = True
    cfg.eff_n = n
    return n

def make_mlp_layer(d):
    w = zeros([d, d])
    for i in range(d * d):
        w.host[i] = float((i * 7) % 13 - 6) / d
    b = zeros([d])
    for i in range(d):
        b.host[i] = 0.01
    device.dev(w)
    device.dev(b)
    return nn.Linear(nn.Tensor(w), nn.Tensor(b))

def make_mlp_input(rows, d):
    x = zeros([rows, d])
    for i in range(rows * d):
        x.host[i] = (i % 7) - 3.0
    device.dev(x)
    return nn.Tensor(x)

def make_lr():
    lr = nn.Tensor(core.from_list([-LR], cfg.dtype))
    device.dev(lr.t)
    return lr

def bench_env(name, default):
    value = os.environ.get(name)
    if value is None:
        return default
    return value

def emit(mode, variant, k, n, iters, warm, steady, kernels_n, acc, compiled,
         launches, dtname):
    print '%s %d %d %d %d %f %f %d %f %d %f %s' % (mode, variant, k, n,
        iters, warm, steady, kernels_n, acc, compiled, launches, dtname)
