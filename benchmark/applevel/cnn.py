import sys, os, time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                '..', '..', 'lib_pypy'))

import _tensor, tensorlite

images, iters = int(sys.argv[1]), int(sys.argv[2])
TB_EPS = 1e-05
CNN_C, CNN_HW, CNN_O, CNN_CLS = 3, 32, 8, 10


def cnn_weight(nrows, ncols):
    data = [float((i * 7) % 13 - 6) / nrows for i in range(nrows * ncols)]
    return _tensor.tensor(data, [nrows, ncols])


def cnn_bias(m):
    return _tensor.tensor([0.01] * m)


def make_cnn():
    fan = CNN_C * 9
    feat = CNN_O * (CNN_HW // 2) * (CNN_HW // 2)
    conv = tensorlite.Conv2d(cnn_weight(fan, CNN_O), cnn_bias(CNN_O),
                             CNN_C, CNN_HW, CNN_HW)
    fc = tensorlite.Linear(cnn_weight(feat, CNN_CLS), cnn_bias(CNN_CLS))
    return tensorlite.CNN(conv, tensorlite.BatchNorm2d(CNN_O, eps=TB_EPS),
                          tensorlite.MaxPool2d(CNN_O, CNN_HW, CNN_HW), fc)


def make_input(nrows, pixels):
    data = [(i % 7) - 3.0 for i in range(nrows * pixels)]
    return _tensor.tensor(data, [nrows, pixels])


pixels = CNN_C * CNN_HW * CNN_HW

warmup_cnn = make_cnn()
x = make_input(images, pixels)
for i in range(10):
    warmup_cnn(x).sum().item()

cnn = make_cnn()
x = make_input(images, pixels)
acc = 0.0
t0 = time.time()
for i in range(iters):
    acc += cnn(x).sum().item()
steady_us = (time.time() - t0) / iters * 1e6

print("applevel-cnn images=%d iters=%d steady_us=%.1f checksum=%.6f" %
      (images, iters, steady_us, acc))
