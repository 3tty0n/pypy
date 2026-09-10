import sys, os, time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                '..', '..', 'lib_pypy'))

import _metatensor
from tensorpypy import nn
from tensorpypy.models import MLP, TransformerBlock

TB_D = 64
TB_H = 4
TB_EPS = 1e-05


def tb_weight(nrows, ncols, div=TB_D, requires_grad=False):
    data = [float((i * 7) % 13 - 6) / div for i in range(nrows * ncols)]
    return _metatensor.tensor(data, [nrows, ncols], requires_grad)


def tb_vector(v, d=TB_D, requires_grad=False):
    return _metatensor.tensor([v] * d, None, requires_grad)


def tb_qkv(d=TB_D, heads=TB_H, requires_grad=False):
    dh = d // heads
    data = [0.0] * (d * d)
    for r in range(d):
        for h in range(heads):
            for c in range(dh):
                data[r * d + h * dh + c] = float(((r * dh + c) * 7) % 13 - 6) / d
    return _metatensor.tensor(data, [d, d], requires_grad)


def tb_proj(d=TB_D, heads=TB_H, requires_grad=False):
    dh = d // heads
    data = [0.0] * (d * d)
    for h in range(heads):
        for r in range(dh):
            for c in range(d):
                data[(h * dh + r) * d + c] = float(((r * d + c) * 7) % 13 - 6) / d
    return _metatensor.tensor(data, [d, d], requires_grad)


def make_attn(d, heads):
    return nn.MultiheadAttention(tb_qkv(d, heads), tb_qkv(d, heads),
                                 tb_qkv(d, heads), tb_proj(d, heads), heads)


def make_mlp_layers(requires_grad=False):
    layers = []
    for i in range(2):
        layers.append(nn.Linear(tb_weight(TB_D, TB_D, TB_D, requires_grad),
                                tb_vector(0.01, TB_D, requires_grad)))
    return layers


def make_block():
    attn = nn.MultiheadAttention(tb_qkv(), tb_qkv(), tb_qkv(), tb_proj(), TB_H)
    return TransformerBlock(attn, tb_vector(1.0), tb_vector(0.0),
                                       tb_vector(1.0), tb_vector(0.0),
                                       MLP(make_mlp_layers()), TB_EPS)


def make_train_block():
    attn = nn.MultiheadAttention(tb_qkv(TB_D, TB_H, True), tb_qkv(TB_D, TB_H, True),
                                 tb_qkv(TB_D, TB_H, True), tb_proj(TB_D, TB_H, True),
                                 TB_H)
    return TransformerBlock(attn, tb_vector(1.0, TB_D, True), tb_vector(0.0, TB_D, True),
                                       tb_vector(1.0, TB_D, True), tb_vector(0.0, TB_D, True),
                                       MLP(make_mlp_layers(True)), TB_EPS)


def make_input(nrows, d):
    data = [(i % 7) - 3.0 for i in range(nrows * d)]
    return _metatensor.tensor(data, [nrows, d])


def main():
    rows, iters = int(sys.argv[1]), int(sys.argv[2])
    warmup_block = make_block()
    h = make_input(rows, TB_D)
    for i in range(10):
        h = warmup_block(h)

    block = make_block()
    h = make_input(rows, TB_D)
    t0 = time.time()
    for i in range(iters):
        h = block(h)
    steady_us = (time.time() - t0) / iters * 1e6

    print("applevel-transformer rows=%d iters=%d steady_us=%.1f checksum=%.6f" %
          (rows, iters, steady_us, h.sum().item()))


if __name__ == '__main__':
    main()
