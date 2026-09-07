import os
from rpython.rlib import jit
from rpython.metatensor import device
from rpython.metatensor.ops import (tensor_add, tensor_mul, tensor_relu,
    tensor_sum, tensor_item, tensor_force)
from bench.common import sink, zeros

driver = jit.JitDriver(greens=['k', 'variant'], reds='auto', is_recursive=True)

def make_inputs(n):
    w = zeros([n])
    b = zeros([n])
    for i in range(n):
        w.host[i] = (i % 7) - 3.0
        b.host[i] = 0.5
    device.dev(w)
    device.dev(b)
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
