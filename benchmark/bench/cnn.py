from rpython.rlib import jit
from rpython.metatensor import device, nn
from bench.common import (CNN_C, CNN_CLS, CNN_HW, CNN_O, TB_EPS,
    make_mlp_input, rows_of, zeros)

cnn_driver = jit.JitDriver(greens=[], reds='auto', is_recursive=True)

def cnn_weight(rows, cols):
    w = zeros([rows, cols])
    for i in range(rows * cols):
        w.host[i] = float((i * 7) % 13 - 6) / rows
    device.dev(w)
    return nn.Tensor(w)

def cnn_bias(m):
    t = zeros([m])
    for i in range(m):
        t.host[i] = 0.01
    device.dev(t)
    return nn.Tensor(t)

def make_cnn():
    fan = CNN_C * 9
    feat = CNN_O * (CNN_HW // 2) * (CNN_HW // 2)
    conv = nn.Conv2d(cnn_weight(fan, CNN_O), cnn_bias(CNN_O),
                             CNN_C, CNN_HW, CNN_HW, CNN_O)
    fc = nn.Linear(cnn_weight(feat, CNN_CLS), cnn_bias(CNN_CLS))
    return nn.CNN(conv, nn.BatchNorm2d(CNN_O, TB_EPS),
                          nn.MaxPool2d(CNN_O, CNN_HW, CNN_HW), fc)

def run_cnn(n, iters):
    pixels = CNN_C * CNN_HW * CNN_HW
    rows = rows_of(n, pixels)
    cnn = make_cnn()
    x = make_mlp_input(rows, pixels)
    acc = 0.0
    i = 0
    while i < iters:
        cnn_driver.jit_merge_point()
        acc += cnn.forward(x).sum().item()
        i += 1
    return acc
