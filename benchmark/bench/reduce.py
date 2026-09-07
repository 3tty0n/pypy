from rpython.rlib import jit
from rpython.metatensor import core, device, ops
from rpython.metatensor.ops import tensor_add, tensor_mul, tensor_sum, tensor_item
from bench.common import (MLP_D, RED_COLS, make_mlp_input, make_mlp_layer,
    rows_of, zeros)

reduction_driver = jit.JitDriver(greens=[], reds='auto', is_recursive=True)
matmul_driver = jit.JitDriver(greens=[], reds='auto', is_recursive=True)

def run_reduction(n, iters):
    rows = rows_of(n, RED_COLS)
    x = zeros([rows, RED_COLS])
    for i in range(rows * RED_COLS):
        x.host[i] = (i % 7) - 3.0
    device.dev(x)
    i = 0
    while i < iters:
        reduction_driver.jit_merge_point()
        h = ops.sum(tensor_mul(x, x, core.BC_NONE), 1)
        x = tensor_add(x, h, core.BC_R_COL)
        i += 1
    return tensor_item(tensor_sum(x, -1))

def run_matmul(n, iters):
    rows = rows_of(n, MLP_D)
    layer = make_mlp_layer(MLP_D)
    x = make_mlp_input(rows, MLP_D)
    i = 0
    while i < iters:
        matmul_driver.jit_merge_point()
        x = layer.forward(x)
        i += 1
    return tensor_item(tensor_sum(x.t, -1))
