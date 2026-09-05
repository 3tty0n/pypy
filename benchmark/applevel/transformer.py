import sys, os, time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                '..', '..', 'lib_pypy'))

import _tensor, tensorlite

rows, iters = int(sys.argv[1]), int(sys.argv[2])
TB_D = 64
TB_H = 4
TB_EPS = 1e-05


def tb_weight(nrows, ncols):
    data = [float((i * 7) % 13 - 6) / TB_D for i in range(nrows * ncols)]
    return _tensor.tensor(data, [nrows, ncols])


def tb_vector(v):
    return _tensor.tensor([v] * TB_D)


def make_block():
    dh = TB_D // TB_H
    heads = []
    for i in range(TB_H):
        heads.append(tensorlite.Head(tb_weight(TB_D, dh), tb_weight(TB_D, dh),
                                     tb_weight(TB_D, dh), tb_weight(dh, TB_D)))
    layers = []
    for i in range(2):
        layers.append(tensorlite.Linear(tb_weight(TB_D, TB_D), tb_vector(0.01)))
    return tensorlite.TransformerBlock(heads, tb_vector(1.0), tb_vector(0.0),
                                       tb_vector(1.0), tb_vector(0.0),
                                       tensorlite.MLP(layers), TB_EPS)


def make_input(nrows, d):
    data = [(i % 7) - 3.0 for i in range(nrows * d)]
    return _tensor.tensor(data, [nrows, d])


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
