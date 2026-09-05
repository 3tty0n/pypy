from rpython.jit.metainterp.test.support import LLJitMixin
from rpython.rlib.jit import JitDriver
from rpython.rlib import rtensor
from rpython.rlib.rtensor import (tensor_add, tensor_mul, tensor_relu,
    tensor_sum, tensor_item, tensor_size, from_list)

rtensor.init_device()

class TestTensor(LLJitMixin):

    def test_fuse_chain_into_one_launch(self):
        driver = JitDriver(greens=[], reds=['n', 'w', 'b', 'acc'])
        def f(n):
            w = from_list([1.0, -2.0, 3.0, -4.0])
            b = from_list([0.5, 0.5, 0.5, 0.5])
            acc = 0.0
            while n > 0:
                driver.jit_merge_point(n=n, acc=acc, w=w, b=b)
                h = tensor_relu(tensor_add(tensor_mul(w, b), b))
                acc += tensor_item(tensor_sum(h))
                n -= 1
            return acc
        import os
        before = rtensor.counter.n
        res = self.meta_interp(f, [10])
        assert res == f(10)
        if 'RTENSOR_CPU' not in os.environ:
            assert rtensor.counter.n > before
        self.check_simple_loop(call_r=1, call_pure_r=0, call_f=1)

    def test_guard_inside_region_resumes(self):
        driver = JitDriver(greens=[], reds=['n', 'w', 'b', 'acc'])
        def f(n):
            w = from_list([1.0, -2.0, 3.0, -4.0])
            b = from_list([0.5, 0.5, 0.5, 0.5])
            acc = 0.0
            while n > 0:
                driver.jit_merge_point(n=n, acc=acc, w=w, b=b)
                t = tensor_mul(w, b)
                if n % 7 == 0:
                    t = tensor_add(t, b)
                h = tensor_relu(t)
                acc += tensor_item(tensor_sum(h))
                n -= 1
            return acc
        res = self.meta_interp(f, [30])
        assert res == f(30)
        from rpython.jit.metainterp.test.support import get_stats
        assert get_stats().compiled_count >= 2
        self.check_resops(call_pure_r=0)

    def test_loop_carried_tensor(self):
        driver = JitDriver(greens=[], reds=['n', 'h', 'b'])
        def f(n):
            h = from_list([1.0, 2.0, 3.0])
            b = from_list([0.5, -0.5, 0.25])
            while n > 0:
                driver.jit_merge_point(n=n, h=h, b=b)
                h = tensor_relu(tensor_add(h, b))
                n -= 1
            return tensor_item(tensor_sum(h))
        res = self.meta_interp(f, [12])
        assert res == f(12)
        self.check_simple_loop(call_r=1)

class TestTensorMeta(LLJitMixin):

    def test_size_of_virtual_does_not_force(self):
        driver = JitDriver(greens=[], reds=['n', 'w', 'b', 'acc'])
        def f(n):
            w = from_list([1.0, -2.0, 3.0, -4.0])
            b = from_list([0.5, 0.5, 0.5, 0.5])
            acc = 0.0
            while n > 0:
                driver.jit_merge_point(n=n, acc=acc, w=w, b=b)
                h = tensor_relu(tensor_add(tensor_mul(w, b), b))
                if tensor_size(h) > 2:
                    h = tensor_add(h, b)
                if tensor_size(tensor_sum(h)) == 1:
                    acc += tensor_item(tensor_sum(h))
                n -= 1
            return acc
        res = self.meta_interp(f, [10])
        assert res == f(10)
        self.check_simple_loop(call_r=1, call_pure_r=0)

    def test_kernel_cache_reuses_compiled_kernel(self):
        driver = JitDriver(greens=[], reds=['n', 'w', 'b', 'acc'])
        def f(n):
            w = from_list([1.0, -2.0, 3.0, -4.0])
            b = from_list([0.5, 0.5, 0.5, 0.5])
            acc = 0.0
            while n > 0:
                driver.jit_merge_point(n=n, acc=acc, w=w, b=b)
                acc += tensor_item(tensor_sum(tensor_relu(tensor_mul(tensor_mul(w, b), b))))
                n -= 1
            return acc
        before = rtensor.counter.n
        res = self.meta_interp(f, [10])
        assert res == f(10)
        import os
        if 'RTENSOR_CPU' not in os.environ:
            assert rtensor.counter.n - before == 1


def test_gpu_launch_matches_cpu():
    k = rtensor.build_kernel(2, [rtensor.MUL, rtensor.ADD, rtensor.RELU, rtensor.SUM],
                             [0, 2, 3, 4], [1, 1, -1, -1])
    if k.fn == 0:
        import py
        py.test.skip("no CUDA device / nvcc")
    a = from_list([1.0, -2.0, 3.0, -4.0]); b = from_list([0.5, 0.5, 0.5, 0.5])
    gpu = rtensor.launch_gpu(k, [a, b])
    assert gpu and rtensor.host(gpu)[0] == 3.0
    k.fn = 0
    assert rtensor.host(rtensor.launch(k, a, b, rtensor.NULLTENSOR))[0] == 3.0
    assert '"tt.reduce"(%v4)' in rtensor.to_ttir(k, 'k')
    k2 = rtensor.build_kernel(2, [rtensor.ADD, rtensor.RELU], [0, 2], [1, -1])
    assert k2.fn != 0
    r = rtensor.launch_gpu(k2, [from_list([1.0, -2.0, 3.0]), from_list([0.5, 0.5, -4.0])])
    assert not r.host and r.dptr != 0
    assert [rtensor.host(r)[i] for i in range(3)] == [1.5, 0.0, 0.0]

def test_tile_ir_text():
    k = rtensor.build_kernel(2, [rtensor.MUL, rtensor.ADD, rtensor.RELU, rtensor.SUM],
                             [0, 2, 3, 4], [1, 1, -1, -1])
    out = rtensor.launch(k, from_list([1.0, -2.0]), from_list([0.5, 0.5]),
                         rtensor.NULLTENSOR)
    assert rtensor.host(out)[0] == 1.0
    text = rtensor.to_tile_ir(k, 'k0', 2)
    assert 'arith.mulf %v0, %v1' in text
    assert 'cuda_tile.reduce add %v4' in text
    assert text.count('cuda_tile.load_ptr_tko') == 2
