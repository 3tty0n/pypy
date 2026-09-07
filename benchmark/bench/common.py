import os
from rpython.metatensor import core, device, nn

class Sink(object):
    fd = -1
sink = Sink()

class Cfg(object):
    dtype = core.F64
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

def rows_of(n, d):
    rows = n // d
    if rows <= 0:
        rows = 1
    return rows

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
