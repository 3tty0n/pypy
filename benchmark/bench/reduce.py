from rpython.rlib import jit
from rpython.metatensor import core, device, ops, runtime
from rpython.metatensor.ops import (tensor_add, tensor_div, tensor_mul,
    tensor_sum, tensor_item)
from bench.common import (MEM_MATMUL, MEM_REDUCE, MLP_D, RED_COLS, cfg,
    fit_rows, make_mlp_input, make_mlp_layer, zeros)

reduction_driver = jit.JitDriver(greens=[], reds='auto', is_recursive=True)
matmul_driver = jit.JitDriver(greens=[], reds='auto', is_recursive=True)

def run_reduction(n, iters):
    """x <- 0.5*x + 2h/(1+h) with h = mean(x*x) over the row.

    Row reduction, broadcast back over the row, elementwise combine - the
    structure this variant exists to measure - but bounded: the earlier
    x += rowsum(x*x) doubles every iteration and is inf long before the
    timed loop ends, so its accumulator carried no information and could not
    be compared across systems or dtypes.  h >= 0 and 2h/(1+h) < 2, so |x|
    stays below 4 here, and the fixed point at 2+sqrt(3) is strongly
    attracting, which is what makes every system and every dtype land on the
    same accumulator instead of amplifying its own rounding.
    """
    rows = fit_rows(n, RED_COLS, 0, MEM_REDUCE)
    x = zeros([rows, RED_COLS])
    for i in range(rows * RED_COLS):
        x.host[i] = (i % 7) - 3.0
    device.dev(x)
    i = 0
    while i < iters:
        reduction_driver.jit_merge_point()
        half = runtime.scalar_of(0.5, cfg.dtype)
        one = runtime.scalar_of(1.0, cfg.dtype)
        two = runtime.scalar_of(2.0, cfg.dtype)
        inv = runtime.scalar_of(1.0 / RED_COLS, cfg.dtype)
        h = tensor_mul(ops.sum(tensor_mul(x, x, core.BC_NONE), 1), inv,
                       core.BC_R_SCALAR)
        g = tensor_mul(
            tensor_div(h, tensor_add(h, one, core.BC_R_SCALAR), core.BC_NONE),
            two, core.BC_R_SCALAR)
        x = tensor_add(tensor_mul(x, half, core.BC_R_SCALAR), g,
                       core.BC_R_COL)
        i += 1
    return tensor_item(tensor_sum(x, -1))

def run_matmul(n, iters):
    rows = fit_rows(n, MLP_D, 0, MEM_MATMUL)
    layer = make_mlp_layer(MLP_D)
    x = make_mlp_input(rows, MLP_D)
    i = 0
    while i < iters:
        matmul_driver.jit_merge_point()
        x = layer.forward(x)
        i += 1
    return tensor_item(tensor_sum(x.t, -1))
