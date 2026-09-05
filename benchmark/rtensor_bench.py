import sys, os, time
from rpython.rlib import jit
from rpython.rlib import rtensor
from rpython.rlib.rtensor import (tensor_add, tensor_mul, tensor_relu,
    tensor_sum, tensor_item, tensor_force, new_tensor)

class Sink(object):
    fd = -1
sink = Sink()

driver = jit.JitDriver(greens=['k', 'variant'], reds='auto')

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
            h = tensor_relu(tensor_add(tensor_mul(h, b, 0), b))
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
        print 'usage: rtensor-bench MODE VARIANT K N ITERS  (MODE: fused|eager|nojit, VARIANT: 0..5)'
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
