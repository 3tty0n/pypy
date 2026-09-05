import sys, os, time
from rpython.rlib import jit
from rpython.rlib import rtensor, rtensor_nn
from rpython.rlib.rtensor import (tensor_add, tensor_mul, tensor_relu,
    tensor_sum, tensor_item, tensor_force, new_tensor)

class Sink(object):
    fd = -1
sink = Sink()

driver = jit.JitDriver(greens=['k', 'variant'], reds='auto', is_recursive=True)
mlp_driver = jit.JitDriver(greens=[], reds='auto', is_recursive=True)
train_driver = jit.JitDriver(greens=[], reds='auto', is_recursive=True)
block_driver = jit.JitDriver(greens=[], reds='auto', is_recursive=True)
cnn_driver = jit.JitDriver(greens=[], reds='auto', is_recursive=True)

MLP_D = 256
LR = 1e-06
TB_D = 64
TB_H = 4
TB_EPS = 1e-05

def make_mlp_layer(d):
    w = rtensor.zeros([d, d])
    for i in range(d * d):
        w.host[i] = float((i * 7) % 13 - 6) / d
    b = rtensor.zeros([d])
    for i in range(d):
        b.host[i] = 0.01
    rtensor.dev(w)
    rtensor.dev(b)
    return rtensor_nn.Linear(rtensor_nn.Tensor(w), rtensor_nn.Tensor(b))

def make_mlp(d):
    return rtensor_nn.MLP([make_mlp_layer(d), make_mlp_layer(d),
                           make_mlp_layer(d)])

def make_mlp_input(rows, d):
    x = rtensor.zeros([rows, d])
    for i in range(rows * d):
        x.host[i] = (i % 7) - 3.0
    rtensor.dev(x)
    return rtensor_nn.Tensor(x)

def run_mlp(n, iters):
    rows = n // MLP_D
    if rows <= 0:
        rows = 1
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
    rows = n // MLP_D
    if rows <= 0:
        rows = 1
    mlp = make_mlp(MLP_D)
    x = make_mlp_input(rows, MLP_D)
    params = mlp.parameters()
    lr = rtensor_nn.Tensor(rtensor.from_list([-LR]))
    rtensor.dev(lr.t)
    loss = 0.0
    i = 0
    while i < iters:
        train_driver.jit_merge_point()
        out = mlp.forward(x).sum()
        out.backward()
        rtensor_nn.sgd_step(params, lr)
        loss = out.item()
        i += 1
    return loss

def tb_weight(rows, cols):
    w = rtensor.zeros([rows, cols])
    for i in range(rows * cols):
        w.host[i] = float((i * 7) % 13 - 6) / TB_D
    rtensor.dev(w)
    return rtensor_nn.Tensor(w)

def tb_vector(v):
    t = rtensor.zeros([TB_D])
    for i in range(TB_D):
        t.host[i] = v
    rtensor.dev(t)
    return rtensor_nn.Tensor(t)

def make_block():
    dh = TB_D // TB_H
    heads = []
    for i in range(TB_H):
        heads.append(rtensor_nn.Head(tb_weight(TB_D, dh), tb_weight(TB_D, dh),
                                     tb_weight(TB_D, dh), tb_weight(dh, TB_D)))
    layers = []
    for i in range(2):
        layers.append(rtensor_nn.Linear(tb_weight(TB_D, TB_D),
                                        tb_vector(0.01)))
    return rtensor_nn.TransformerBlock(heads, tb_vector(1.0), tb_vector(0.0),
                                       tb_vector(1.0), tb_vector(0.0),
                                       rtensor_nn.MLP(layers), TB_EPS)

def run_block(n, iters):
    rows = n // TB_D
    if rows <= 0:
        rows = 1
    block = make_block()
    x = make_mlp_input(rows, TB_D)
    i = 0
    while i < iters:
        block_driver.jit_merge_point()
        x = rtensor_nn.Tensor(block.forward(x).t)
        i += 1
    return tensor_item(tensor_sum(x.t, -1))

CNN_C, CNN_HW, CNN_O, CNN_CLS = 3, 32, 8, 10

def cnn_weight(rows, cols):
    w = rtensor.zeros([rows, cols])
    for i in range(rows * cols):
        w.host[i] = float((i * 7) % 13 - 6) / rows
    rtensor.dev(w)
    return rtensor_nn.Tensor(w)

def cnn_bias(m):
    t = rtensor.zeros([m])
    for i in range(m):
        t.host[i] = 0.01
    rtensor.dev(t)
    return rtensor_nn.Tensor(t)

def make_cnn():
    fan = CNN_C * 9
    feat = CNN_O * (CNN_HW // 2) * (CNN_HW // 2)
    conv = rtensor_nn.Conv2d(cnn_weight(fan, CNN_O), cnn_bias(CNN_O),
                             CNN_C, CNN_HW, CNN_HW, CNN_O)
    fc = rtensor_nn.Linear(cnn_weight(feat, CNN_CLS), cnn_bias(CNN_CLS))
    return rtensor_nn.CNN(conv, rtensor_nn.BatchNorm2d(CNN_O, TB_EPS),
                          rtensor_nn.MaxPool2d(CNN_O, CNN_HW, CNN_HW), fc)

def run_cnn(n, iters):
    pixels = CNN_C * CNN_HW * CNN_HW
    rows = n // pixels
    if rows <= 0:
        rows = 1
    cnn = make_cnn()
    x = make_mlp_input(rows, pixels)
    acc = 0.0
    i = 0
    while i < iters:
        cnn_driver.jit_merge_point()
        acc += cnn.forward(x).sum().item()
        i += 1
    return acc

def run_model(variant, n, iters):
    if variant == 9:
        return run_cnn(n, iters)
    if variant == 7:
        return run_mlp_train(n, iters)
    if variant == 8:
        return run_block(n, iters)
    return run_mlp(n, iters)

def make_inputs(n):
    w = new_tensor(n)
    b = new_tensor(n)
    for i in range(n):
        w.host[i] = (i % 7) - 3.0
        b.host[i] = 0.5
    rtensor.dev(w)
    rtensor.dev(b)
    return w, b

def run(variant, k, h, b, iters):
    i = 0
    while i < iters:
        driver.jit_merge_point(k=k, variant=variant)
        j = 0
        while j < k:
            h = tensor_relu(tensor_add(tensor_mul(h, b, 0), b, 0))
            j += 1
        if variant == 1:
            if i % 7 == 0:
                h = tensor_add(h, b, 0)
        elif variant == 2:
            h = tensor_force(h)
            if i % 7 == 0:
                h = tensor_add(h, b, 0)
        elif variant == 3:
            if tensor_item(tensor_sum(h, -1)) > 0.0:
                h = tensor_add(h, b, 0)
        elif variant == 4:
            try:
                if i % 5 == 0:
                    raise ValueError
                h = tensor_add(h, b, 0)
            except ValueError:
                h = tensor_mul(h, b, 0)
        elif variant == 5:
            if i % 50 == 0:
                os.write(sink.fd, "step\n")
            h = tensor_add(h, b, 0)
        i += 1
    return tensor_item(tensor_sum(h, -1))

def entry_point(argv):
    if len(argv) != 6:
        print 'usage: rtensor-bench MODE VARIANT K N ITERS  (MODE: fused|eager|nojit, VARIANT: 0..9)'
        return 1
    mode = argv[1]
    variant = int(argv[2])
    k = int(argv[3])
    n = int(argv[4])
    iters = int(argv[5])
    jit.set_user_param(None, 'threshold=3,function_threshold=3,trace_eagerness=2')
    if mode == 'eager':
        jit.set_user_param(None, 'enable_opts=intbounds:rewrite:virtualize:'
                                 'string:pure:earlyforce:heap:unroll')
    elif mode == 'nojit':
        jit.set_user_param(None, 'off')
    rtensor.init_device()
    sink.fd = os.open('/dev/null', os.O_WRONLY, 0)
    if variant >= 6:
        run_model(variant, n, 20)
        t0 = time.time()
        run_model(variant, n, 20)
        warm = time.time() - t0
        run_model(variant, n, 30)
        run_model(variant, n, 30)
        before = rtensor.counter.n
        launches_before = rtensor.launch_count()
        t0 = time.time()
        acc = run_model(variant, n, iters)
        rtensor.sync_device()
        steady = (time.time() - t0) / iters * 1e6
        launches = float(rtensor.launch_count() - launches_before) / iters
        rtensor.reset_device()
        print '%s %d %d %d %d %f %f %d %f %d %f' % (mode, variant, k, n, iters,
            warm, steady, rtensor.counter.n, acc, rtensor.counter.n - before,
            launches)
        return 0
    w, b = make_inputs(n)
    t0 = time.time()
    run(variant, k, w, b, 20)
    warm = time.time() - t0
    run(variant, k, w, b, 30)
    run(variant, k, w, b, 30)
    before = rtensor.counter.n
    launches_before = rtensor.launch_count()
    t0 = time.time()
    acc = run(variant, k, w, b, iters)
    rtensor.sync_device()
    steady = (time.time() - t0) / iters * 1e6
    launches = float(rtensor.launch_count() - launches_before) / iters
    rtensor.reset_device()
    print '%s %d %d %d %d %f %f %d %f %d %f' % (mode, variant, k, n, iters,
        warm, steady, rtensor.counter.n, acc, rtensor.counter.n - before,
        launches)
    return 0

def target(*args):
    return entry_point, None

if __name__ == '__main__':
    entry_point(sys.argv)
