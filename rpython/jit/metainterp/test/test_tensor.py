from rpython.jit.metainterp.test.support import LLJitMixin
from rpython.rlib.jit import JitDriver
from rpython.rlib import rtensor, rtensor_nn
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
                h = tensor_relu(tensor_add(tensor_mul(w, b, 0), b, 0))
                acc += tensor_item(tensor_sum(h, -1))
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
                t = tensor_mul(w, b, 0)
                if n % 7 == 0:
                    t = tensor_add(t, b, 0)
                h = tensor_relu(t)
                acc += tensor_item(tensor_sum(h, -1))
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
                h = tensor_relu(tensor_add(h, b, 0))
                n -= 1
            return tensor_item(tensor_sum(h, -1))
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
                h = tensor_relu(tensor_add(tensor_mul(w, b, 0), b, 0))
                if tensor_size(h) > 2:
                    h = tensor_add(h, b, 0)
                if tensor_size(tensor_sum(h, -1)) == 1:
                    acc += tensor_item(tensor_sum(h, -1))
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
                acc += tensor_item(tensor_sum(tensor_relu(tensor_mul(tensor_mul(w, b, 0), b, 0)), -1))
                n -= 1
            return acc
        before = rtensor.counter.n
        res = self.meta_interp(f, [10])
        assert res == f(10)
        import os
        if 'RTENSOR_CPU' not in os.environ:
            assert rtensor.counter.n - before == 1


class TestMultiOutput(LLJitMixin):

    def test_intermediate_forced_later_becomes_extra_output(self):
        driver = JitDriver(greens=[], reds=['n', 'w', 'b', 'h', 'acc'])
        def f(n):
            w = from_list([1.0, -2.0, 3.0, -4.0])
            b = from_list([0.5, 0.5, 0.5, 0.5])
            h = w
            acc = 0.0
            while n > 0:
                driver.jit_merge_point(n=n, acc=acc, w=w, b=b, h=h)
                h = tensor_relu(tensor_add(tensor_mul(h, b, 0), w, 0))
                acc += tensor_item(tensor_sum(h, -1))
                n -= 1
            return acc
        before = rtensor.counter.n
        res = self.meta_interp(f, [10])
        assert res == f(10)
        self.check_simple_loop(call_r=2, call_f=1)
        import os
        if 'RTENSOR_CPU' not in os.environ:
            assert rtensor.counter.n - before == 1
        assert any(',o' in k for k in rtensor.kernel_cache.kernels)


class TestShapeSpecialization(LLJitMixin):

    def setup_method(self, meth):
        rtensor.policy.static = True
        rtensor.policy.seen = []

    def test_static_size_kernel_then_demotion(self):
        driver = JitDriver(greens=[], reds=['n', 'size', 'w', 'b', 'acc'])
        def f(n, size):
            w = rtensor.zeros([size])
            b = rtensor.zeros([size])
            for i in range(size):
                w.host[i] = i - 2.0
                b.host[i] = 0.5
            acc = 0.0
            while n > 0:
                driver.jit_merge_point(n=n, size=size, acc=acc, w=w, b=b)
                h = rtensor.relu(rtensor.add(rtensor.mul(w, b), b))
                if rtensor.tensor_shape(h, 0) == size:
                    acc += rtensor.item(rtensor.sum(h))
                n -= 1
            return acc
        res = self.meta_interp(f, [10, 8])
        assert res == f(10, 8)
        self.check_simple_loop(call_r=1)
        self.check_resops(guard_value=2)
        keys = [k for k in rtensor.kernel_cache.kernels if k.split(',')[1] == '8']
        assert keys
        assert rtensor.policy.static
        for size in (16, 24, 40):
            assert self.meta_interp(f, [10, size]) == f(10, size)
        assert not rtensor.policy.static
        res = self.meta_interp(f, [10, 48])
        assert res == f(10, 48)
        self.check_resops(guard_value=0)
        assert any(k.split(',')[1] == '0' for k in rtensor.kernel_cache.kernels)


def test_gpu_launch_matches_cpu():
    k = rtensor.build_kernel(2, [rtensor.MUL, rtensor.ADD, rtensor.RELU, rtensor.SUM],
                             [0, 2, 3, 4], [1, 1, -1, -1],
                             [0, 0, 0, rtensor.AXIS_ALL])
    if k.fn == 0:
        import py
        py.test.skip("no CUDA device / nvcc")
    a = from_list([1.0, -2.0, 3.0, -4.0]); b = from_list([0.5, 0.5, 0.5, 0.5])
    gpu = rtensor.launch_gpu(k, [a, b])
    assert gpu and rtensor.host(gpu)[0] == 3.0
    k.fn = 0
    assert rtensor.host(rtensor.launch(k, a, b, rtensor.NULLTENSOR))[0] == 3.0
    assert '"tt.reduce"(%v4)' in rtensor.to_ttir(k, 'k')
    k2 = rtensor.build_kernel(2, [rtensor.ADD, rtensor.RELU], [0, 2], [1, -1],
                                   [0, 0])
    assert k2.fn != 0
    r = rtensor.launch_gpu(k2, [from_list([1.0, -2.0, 3.0]), from_list([0.5, 0.5, -4.0])])
    assert not r.host and r.dptr != 0
    assert [rtensor.host(r)[i] for i in range(3)] == [1.5, 0.0, 0.0]

def test_tile_ir_text():
    k = rtensor.build_kernel(2, [rtensor.MUL, rtensor.ADD, rtensor.RELU, rtensor.SUM],
                             [0, 2, 3, 4], [1, 1, -1, -1],
                             [0, 0, 0, rtensor.AXIS_ALL])
    out = rtensor.launch(k, from_list([1.0, -2.0]), from_list([0.5, 0.5]),
                         rtensor.NULLTENSOR)
    assert rtensor.host(out)[0] == 1.0
    text = rtensor.to_tile_ir(k, 'k0', 2)
    assert 'arith.mulf %v0, %v1' in text
    assert 'cuda_tile.reduce add %v4' in text
    assert text.count('cuda_tile.load_ptr_tko') == 2


class TestBroadcast(LLJitMixin):

    def setup_method(self, meth):
        rtensor.policy.static = True
        rtensor.policy.seen = []

    def test_row_broadcast_fused(self):
        driver = JitDriver(greens=[], reds=['n', 'x', 'v', 'acc'])
        def f(n):
            x = rtensor.zeros([3, 4])
            v = rtensor.zeros([4])
            for i in range(12):
                x.host[i] = i - 5.0
            for i in range(4):
                v.host[i] = i * 0.5
            acc = 0.0
            while n > 0:
                driver.jit_merge_point(n=n, acc=acc, x=x, v=v)
                h = rtensor.relu(rtensor.add(x, v))
                acc += rtensor.item(rtensor.sum(h))
                n -= 1
            return acc
        expect = 0.0
        for i in range(12):
            val = (i - 5.0) + (i % 4) * 0.5
            if val > 0.0:
                expect += val
        before = rtensor.counter.n
        res = self.meta_interp(f, [10])
        assert res == expect * 10
        self.check_simple_loop(call_r=1)
        assert any('0:0:1:1' in k for k in rtensor.kernel_cache.kernels)
        import os
        if 'RTENSOR_CPU' not in os.environ:
            assert rtensor.counter.n > before

    def test_scalar_broadcast_fused(self):
        driver = JitDriver(greens=[], reds=['n', 'x', 's', 'acc'])
        def f(n):
            x = rtensor.zeros([3, 4])
            for i in range(12):
                x.host[i] = i - 5.0
            s = from_list([2.0])
            acc = 0.0
            while n > 0:
                driver.jit_merge_point(n=n, acc=acc, x=x, s=s)
                h = rtensor.relu(rtensor.mul(x, s))
                acc += rtensor.item(rtensor.sum(h))
                n -= 1
            return acc
        expect = 0.0
        for i in range(12):
            val = (i - 5.0) * 2.0
            if val > 0.0:
                expect += val
        res = self.meta_interp(f, [10])
        assert res == expect * 10
        self.check_simple_loop(call_r=1)

    def test_sum_axis0_root(self):
        driver = JitDriver(greens=[], reds=['n', 'x', 'w', 'acc'])
        def f(n):
            x = rtensor.zeros([3, 4])
            w = rtensor.zeros([4])
            for i in range(12):
                x.host[i] = i - 5.0
            for i in range(4):
                w.host[i] = 1.0 * (1 << i)
            acc = 0.0
            while n > 0:
                driver.jit_merge_point(n=n, acc=acc, x=x, w=w)
                r = rtensor.sum(rtensor.relu(x), 0)
                acc += rtensor.item(rtensor.sum(rtensor.mul(r, w)))
                n -= 1
            return acc
        vals = [max(i - 5.0, 0.0) for i in range(12)]
        expect = 0.0
        for c in range(4):
            for r in range(3):
                expect += vals[r * 4 + c] * (1 << c)
        res = self.meta_interp(f, [10])
        assert res == expect * 10

    def test_sum_axis1_root(self):
        driver = JitDriver(greens=[], reds=['n', 'x', 'w', 'acc'])
        def f(n):
            x = rtensor.zeros([3, 4])
            w = rtensor.zeros([3])
            for i in range(12):
                x.host[i] = i - 5.0
            for i in range(3):
                w.host[i] = 1.0 * (1 << i)
            acc = 0.0
            while n > 0:
                driver.jit_merge_point(n=n, acc=acc, x=x, w=w)
                r = rtensor.sum(rtensor.relu(x), 1)
                acc += rtensor.item(rtensor.sum(rtensor.mul(r, w)))
                n -= 1
            return acc
        vals = [max(i - 5.0, 0.0) for i in range(12)]
        expect = 0.0
        for r in range(3):
            for c in range(4):
                expect += vals[r * 4 + c] * (1 << r)
        res = self.meta_interp(f, [10])
        assert res == expect * 10

    def test_broadcast_guard_inside_region_resumes(self):
        driver = JitDriver(greens=[], reds=['n', 'x', 'v', 'acc'])
        def f(n):
            x = rtensor.zeros([3, 4])
            v = rtensor.zeros([4])
            for i in range(12):
                x.host[i] = i - 5.0
            for i in range(4):
                v.host[i] = i * 0.5
            acc = 0.0
            while n > 0:
                driver.jit_merge_point(n=n, acc=acc, x=x, v=v)
                t = rtensor.mul(x, v)
                if n % 7 == 0:
                    t = rtensor.add(t, v)
                acc += rtensor.item(rtensor.sum(rtensor.relu(t)))
                n -= 1
            return acc
        res = self.meta_interp(f, [30])
        assert res == f(30)
        from rpython.jit.metainterp.test.support import get_stats
        assert get_stats().compiled_count >= 2

    def test_reshape_then_elementwise(self):
        driver = JitDriver(greens=[], reds=['n', 'm', 'v', 'acc'])
        def f(n):
            base = rtensor.zeros([12])
            for i in range(12):
                base.host[i] = i - 5.0
            m = rtensor.reshape(base, [3, 4])
            v = rtensor.zeros([4])
            for i in range(4):
                v.host[i] = i * 0.5
            acc = 0.0
            while n > 0:
                driver.jit_merge_point(n=n, acc=acc, m=m, v=v)
                h = rtensor.relu(rtensor.add(m, v))
                acc += rtensor.item(rtensor.sum(h))
                n -= 1
            return acc
        expect = 0.0
        for i in range(12):
            val = (i - 5.0) + (i % 4) * 0.5
            if val > 0.0:
                expect += val
        res = self.meta_interp(f, [10])
        assert res == expect * 10
        self.check_simple_loop(call_r=1)

    def test_inplace_keeps_fusion(self):
        driver = JitDriver(greens=[], reds=['n', 'h', 'b'])
        def f(n):
            h = from_list([1.0, 2.0, 3.0])
            b = from_list([0.5, -0.5, 0.25])
            while n > 0:
                driver.jit_merge_point(n=n, h=h, b=b)
                h = rtensor.relu(rtensor.add_(h, b))
                n -= 1
            return rtensor.item(rtensor.sum(h))
        res = self.meta_interp(f, [12])
        assert res == f(12)
        self.check_simple_loop(call_r=1)


def test_reshape_shares_storage():
    t = from_list([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    v = rtensor.reshape(t, [2, 3])
    assert v.size == 6 and v.shape[0] == 2 and v.shape[1] == 3
    assert v.host is t.host and v.dptr == t.dptr
    t.host[0] = 9.0
    assert rtensor.host(v)[0] == 9.0

def test_sum_axis_elements():
    x = rtensor.zeros([3, 4])
    for i in range(12):
        x.host[i] = i - 5.0
    vals = [i - 5.0 for i in range(12)]
    r0 = rtensor.sum(x, 0)
    assert r0.size == 4 and r0.shape[0] == 4
    for c in range(4):
        assert rtensor.host(r0)[c] == sum([vals[r * 4 + c] for r in range(3)])
    r1 = rtensor.sum(x, 1)
    assert r1.size == 3 and r1.shape[0] == 3
    for r in range(3):
        assert rtensor.host(r1)[r] == sum([vals[r * 4 + c] for c in range(4)])
    assert rtensor.item(rtensor.sum(x)) == sum(vals)


B, D = 2, 3
LR = 0.01
BACKWARD_CALL_R = 5

def _fill_matrix(w, rows, cols, seed):
    for i in range(rows):
        for j in range(cols):
            w.host[i * cols + j] = ((seed + i * 3 - j * 2) % 5 - 2) * 0.25

def _fill_bias(b, seed):
    for j in range(D):
        b.host[j] = 0.125 * (j + 1) + seed * 0.0625

def _make_layer(seed):
    w = rtensor.zeros([D, D])
    b = rtensor.zeros([D])
    _fill_matrix(w, D, D, seed)
    _fill_bias(b, seed)
    return rtensor_nn.Linear(rtensor_nn.Tensor(w), rtensor_nn.Tensor(b))

def _make_mlp():
    return rtensor_nn.MLP([_make_layer(1), _make_layer(2), _make_layer(3)])

def _cpu_matrix(seed):
    return [[((seed + i * 3 - j * 2) % 5 - 2) * 0.25 for j in range(D)]
            for i in range(D)]

def _cpu_bias(seed):
    return [0.125 * (j + 1) + seed * 0.0625 for j in range(D)]

def _cpu_layer(x, seed):
    w = _cpu_matrix(seed)
    b = _cpu_bias(seed)
    rows = len(x)
    y = [[0.0] * D for _ in range(rows)]
    for i in range(rows):
        for j in range(D):
            acc = 0.0
            for k in range(D):
                acc += x[i][k] * w[k][j]
            v = acc + b[j]
            y[i][j] = v if v > 0.0 else 0.0
    return y

def _cpu_mlp_step(x):
    x = _cpu_layer(x, 1)
    x = _cpu_layer(x, 2)
    x = _cpu_layer(x, 3)
    return x

def _cpu_expect(n):
    x0 = [[(i - j) * 0.5 for j in range(D)] for i in range(B)]
    y = _cpu_mlp_step(x0)
    return sum([sum(row) for row in y]) * n


class TestTensorNN(LLJitMixin):

    def test_mlp_forward_fuses_bias_relu(self):
        driver = JitDriver(greens=[], reds=['n', 'x', 'mlp', 'acc'])
        def f(n):
            x0 = rtensor.zeros([B, D])
            for i in range(B):
                for j in range(D):
                    x0.host[i * D + j] = (i - j) * 0.5
            mlp = _make_mlp()
            x = rtensor_nn.Tensor(x0)
            acc = 0.0
            while n > 0:
                driver.jit_merge_point(n=n, x=x, mlp=mlp, acc=acc)
                y = mlp.forward(x)
                acc += y.sum().item()
                n -= 1
            return acc
        res = self.meta_interp(f, [5])
        assert res == _cpu_expect(5)
        self.check_simple_loop(call_r=6)


AB, AD = 2, 3
AW = [1.0, -2.0, 0.5]
ABIAS = [0.25, 0.5, -1.0]

def _autograd_ref():
    loss = 0.0
    gw = [0.0] * AD
    gb = [0.0] * AD
    for i in range(AB):
        for j in range(AD):
            x = i * AD + j - 2.5
            v = x * AW[j] + ABIAS[j]
            if v > 0.0:
                loss += v
                gw[j] += x
                gb[j] += 1.0
    return loss, gw, gb

def _autograd_setup():
    x = rtensor.zeros([AB, AD])
    for i in range(AB * AD):
        x.host[i] = i - 2.5
    w = rtensor.zeros([AD])
    b = rtensor.zeros([AD])
    for j in range(AD):
        w.host[j] = AW[j]
        b.host[j] = ABIAS[j]
    return x, w, b

def test_autograd_elementwise_matches_reference():
    x, w, b = _autograd_setup()
    xt = rtensor_nn.Tensor(x)
    wt = rtensor_nn.Tensor(w, True)
    bt = rtensor_nn.Tensor(b, True)
    loss = xt.mul(wt).add(bt).relu().sum()
    loss.backward()
    eloss, egw, egb = _autograd_ref()
    assert loss.item() == eloss
    for j in range(AD):
        assert rtensor.host(wt.grad.t)[j] == egw[j]
        assert rtensor.host(bt.grad.t)[j] == egb[j]

def _mlp_train_ref(iters, lr):
    x = [[(i - j) * 0.5 for j in range(D)] for i in range(B)]
    ws = [_cpu_matrix(1), _cpu_matrix(2)]
    bs = [_cpu_bias(1), _cpu_bias(2)]
    losses = []
    for _ in range(iters):
        acts = [x]
        pre = []
        h = x
        for l in range(2):
            y = [[sum([h[i][k] * ws[l][k][j] for k in range(D)]) + bs[l][j]
                  for j in range(D)] for i in range(B)]
            r = [[v if v > 0.0 else 0.0 for v in row] for row in y]
            pre.append(y)
            acts.append(r)
            h = r
        losses.append(sum([sum(row) for row in h]))
        g = [[1.0] * D for _ in range(B)]
        for l in (1, 0):
            g = [[g[i][j] if pre[l][i][j] > 0.0 else 0.0 for j in range(D)]
                 for i in range(B)]
            gb = [sum([g[i][j] for i in range(B)]) for j in range(D)]
            gw = [[sum([acts[l][i][k] * g[i][j] for i in range(B)])
                   for j in range(D)] for k in range(D)]
            gx = [[sum([g[i][j] * ws[l][k][j] for j in range(D)])
                   for k in range(D)] for i in range(B)]
            for k in range(D):
                for j in range(D):
                    ws[l][k][j] -= lr * gw[k][j]
            for j in range(D):
                bs[l][j] -= lr * gb[j]
            g = gx
    return losses


class TestAutograd(LLJitMixin):

    def setup_method(self, meth):
        rtensor.policy.static = True
        rtensor.policy.seen = []

    def test_backward_of_relu_mul_add_fuses(self):
        driver = JitDriver(greens=[], reds=['n', 'x', 'w', 'b', 'acc'])
        def f(n):
            x, w, b = _autograd_setup()
            acc = 0.0
            while n > 0:
                driver.jit_merge_point(n=n, x=x, w=w, b=b, acc=acc)
                xt = rtensor_nn.Tensor(x)
                wt = rtensor_nn.Tensor(w, True)
                bt = rtensor_nn.Tensor(b, True)
                loss = xt.mul(wt).add(bt).relu().sum()
                loss.backward()
                acc += (loss.item() + 3.0 * wt.grad.sum().item() +
                        7.0 * bt.grad.sum().item())
                n -= 1
            return acc
        eloss, egw, egb = _autograd_ref()
        expect = (eloss + 3.0 * sum(egw) + 7.0 * sum(egb)) * 10
        res = self.meta_interp(f, [10])
        assert res == expect
        self.check_simple_loop(call_r=BACKWARD_CALL_R, call_f=3)

    def test_mlp_train_step_matches_reference(self):
        driver = JitDriver(greens=[], reds=['n', 'x', 'mlp', 'lr', 'loss', 'prev'])
        def f(n):
            x0 = rtensor.zeros([B, D])
            for i in range(B):
                for j in range(D):
                    x0.host[i * D + j] = (i - j) * 0.5
            mlp = rtensor_nn.MLP([_make_layer(1), _make_layer(2)])
            x = rtensor_nn.Tensor(x0)
            lr = rtensor_nn.Tensor(rtensor.from_list([-LR]))
            params = mlp.parameters()
            loss = 0.0
            prev = 1e30
            while n > 0:
                driver.jit_merge_point(n=n, x=x, mlp=mlp, lr=lr, loss=loss,
                                       prev=prev)
                y = mlp.forward(x)
                out = y.reshape([B * D]).sum()
                out.backward()
                rtensor_nn.sgd_step(mlp.parameters(), lr)
                loss = out.item()
                if loss >= prev:
                    return -1.0
                prev = loss
                n -= 1
            return loss
        losses = _mlp_train_ref(6, LR)
        assert losses[0] > losses[-1]
        res = self.meta_interp(f, [6])
        assert abs(res - losses[-1]) < 1e-9
