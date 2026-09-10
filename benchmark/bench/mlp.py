from rpython.rlib import jit
from rpython.metatensor import nn
from rpython.metatensor.ops import tensor_sum, tensor_item
from bench.common import (MEM_MLP, MEM_MLP_TRAIN, MLP_D, fit_rows, make_lr,
    make_mlp_input, make_mlp_layer)

mlp_driver = jit.JitDriver(greens=[], reds='auto', is_recursive=True)
train_driver = jit.JitDriver(greens=[], reds='auto', is_recursive=True)

def make_mlp(d):
    return nn.MLP([make_mlp_layer(d), make_mlp_layer(d),
                           make_mlp_layer(d)])

def run_mlp(n, iters):
    rows = fit_rows(n, MLP_D, 0, MEM_MLP)
    mlp = make_mlp(MLP_D)
    x = make_mlp_input(rows, MLP_D)
    i = 0
    while i < iters:
        mlp_driver.jit_merge_point()
        y = mlp.forward(x)
        h = y
        x = h
        i += 1
    return tensor_item(tensor_sum(x.t, -1))

def run_mlp_train(n, iters):
    rows = fit_rows(n, MLP_D, 0, MEM_MLP_TRAIN)
    mlp = make_mlp(MLP_D)
    x = make_mlp_input(rows, MLP_D)
    params = mlp.parameters()
    lr = make_lr()
    loss = 0.0
    i = 0
    while i < iters:
        train_driver.jit_merge_point()
        out = mlp.forward(x).sum()
        out.backward()
        nn.sgd_step(params, lr)
        loss = out.item()
        i += 1
    return loss
