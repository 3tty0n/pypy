import os
os.environ.setdefault('RTENSOR_CPU', '1')


class AppTestTensor(object):
    spaceconfig = dict(usemodules=['_metatensor'])

    def test_construct(self):
        import _metatensor
        t = _metatensor.tensor([1.0, 2.0, 3.0])
        assert t.shape == (3,)
        assert t.size == 3
        assert t.requires_grad is False
        m = _metatensor.tensor([[1.0, 2.0], [3.0, 4.0]])
        assert m.shape == (2, 2)

    def test_zeros(self):
        import _metatensor
        z = _metatensor.zeros([2, 3])
        assert z.shape == (2, 3)
        assert z.sum().item() == 0.0

    def test_add_mul_broadcast(self):
        import _metatensor
        a = _metatensor.tensor([1.0, 2.0, 3.0])
        b = _metatensor.tensor([10.0])
        c = a.add(b)
        assert c.sum().item() == 36.0
        d = a.mul(_metatensor.tensor([2.0]))
        assert d.sum().item() == 12.0
        e = a + b
        assert e.sum().item() == 36.0

    def test_sum_axis(self):
        import _metatensor
        m = _metatensor.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        assert m.sum().item() == 21.0
        s0 = m.sum(0)
        w0 = _metatensor.tensor([1.0, 2.0, 3.0])
        assert s0.mul(w0).sum().item() == 46.0
        s1 = m.sum(1)
        w1 = _metatensor.tensor([1.0, 2.0])
        assert s1.mul(w1).sum().item() == 36.0

    def test_matmul(self):
        import _metatensor
        x = _metatensor.tensor([[1.0, 2.0], [3.0, 4.0]])
        w = _metatensor.tensor([[1.0, 0.0], [0.0, 1.0]])
        y = x.matmul(w)
        assert y.sum().item() == 10.0

    def test_reshape(self):
        import _metatensor
        t = _metatensor.tensor([1.0, 2.0, 3.0, 4.0])
        r = t.reshape([2, 2])
        assert r.shape == (2, 2)
        assert r.sum().item() == 10.0

    def test_shape_mismatch(self):
        import _metatensor
        a = _metatensor.tensor([1.0, 2.0, 3.0])
        b = _metatensor.tensor([1.0, 2.0])
        raises(ValueError, a.add, b)
        raises(ValueError, a.reshape, [3, 3])

    def test_add__inplace_updates_view(self):
        import _metatensor
        t = _metatensor.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        v = t.reshape([2, 3])
        b = _metatensor.tensor([1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
        r = t.add_(b)
        assert r is t
        assert v.sum().item() == 27.0

    def test_add__on_requires_grad_leaf_raises(self):
        import _metatensor
        a = _metatensor.tensor([1.0, 2.0], requires_grad=True)
        b = _metatensor.tensor([0.5, 0.5])
        raises(ValueError, a.add_, b)
        raises(ValueError, a.mul_, b)

    def test_backward(self):
        import _metatensor
        x = _metatensor.tensor([1.0, -2.0, 3.0], requires_grad=True)
        w = _metatensor.tensor([2.0, 1.0, -1.0], requires_grad=True)
        b = _metatensor.tensor([0.5, 0.5, 0.5], requires_grad=True)
        z = x.mul(w).add(b).relu()
        loss = z.sum()
        assert loss.item() == 2.5
        assert x.grad is None
        loss.backward()
        assert x.grad.sum().item() == 2.0
        assert w.grad.sum().item() == 1.0
        assert b.grad.sum().item() == 1.0

    def test_mlp_training_step(self):
        import _metatensor
        from tensorpypy import nn
        from tensorpypy.models import MLP
        from tensorpypy import optim

        w1 = _metatensor.tensor([[1.0, -1.0], [0.5, 0.5]], requires_grad=True)
        b1 = _metatensor.tensor([0.0, 0.0], requires_grad=True)
        w2 = _metatensor.tensor([[1.0], [1.0]], requires_grad=True)
        b2 = _metatensor.tensor([0.0], requires_grad=True)
        mlp = MLP([nn.Linear(w1, b1),
                               nn.Linear(w2, b2)])
        x = _metatensor.tensor([[1.0, 2.0]])
        target = _metatensor.tensor([[0.0]])

        def loss_fn():
            y = mlp(x)
            d = y.add(target.mul(_metatensor.tensor([-1.0])))
            return d.mul(d).sum()

        loss0 = loss_fn().item()
        for i in range(5):
            loss = loss_fn()
            loss.backward()
            optim.sgd_step(mlp.parameters(), 0.05)
        loss1 = loss_fn().item()
        assert loss1 < loss0

    def test_sub_div_exp_sqrt_max(self):
        import _metatensor
        a = _metatensor.tensor([1.0, 4.0, 9.0])
        b = _metatensor.tensor([1.0, 2.0, 3.0])
        assert a.sub(b).sum().item() == 8.0
        assert (a - b).sum().item() == 8.0
        assert a.div(b).sum().item() == 6.0
        assert (a / b).sum().item() == 6.0
        assert a.sqrt().sum().item() == 6.0
        assert abs(b.exp().sum().item() - 30.19287485057736) < 1e-12
        assert a.max().item() == 9.0
        m = _metatensor.tensor([[1.0, 5.0], [7.0, 2.0]])
        assert m.max(1).sum().item() == 12.0
        assert m.max(0).sum().item() == 12.0

    def test_matmul_transpose_b(self):
        import _metatensor
        x = _metatensor.tensor([[1.0, 2.0], [3.0, 4.0]])
        w = _metatensor.tensor([[1.0, 10.0], [100.0, 1000.0]])
        assert x.matmul(w, True).sum().item() == 6464.0

    def test_softmax_layernorm(self):
        import _metatensor, math
        from tensorpypy.functional import softmax, layer_norm
        x = _metatensor.tensor([[1.0, 2.0, 3.0], [1.0, 1.0, 1.0]])
        s = softmax(x)
        assert abs(s.sum().item() - 2.0) < 1e-12
        e = [math.exp(v - 3.0) for v in [1.0, 2.0, 3.0]]
        tot = sum(e)
        pick = _metatensor.tensor([[1.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
        assert abs(s.mul(pick).sum().item() - e[0] / tot) < 1e-12
        one = _metatensor.tensor([1.0, 1.0, 1.0])
        zero = _metatensor.tensor([0.0, 0.0, 0.0])
        y = layer_norm(x, one, zero)
        assert abs(y.sum().item()) < 1e-9
        std = math.sqrt(2.0 / 3.0)
        assert abs(y.mul(pick).sum().item() + 1.0 / std) < 1e-4

    def test_cnn_forward(self):
        import _metatensor, math
        from tensorpypy import nn
        from tensorpypy.models import CNN
        C, H, W, O, NC = 2, 4, 4, 3, 5
        fan = C * 9
        feat = O * (H // 2) * (W // 2)
        wc = [float((i * 7) % 13 - 6) / fan for i in range(fan * O)]
        wf = [float((i * 7) % 13 - 6) / feat for i in range(feat * NC)]
        x = [float(i % 5) - 2.0 for i in range(C * H * W)]
        cnn = CNN(
            nn.Conv2d(_metatensor.tensor(wc).reshape([fan, O]),
                              _metatensor.tensor([0.01] * O), C, H, W),
            nn.BatchNorm2d(O),
            nn.MaxPool2d(O, H, W),
            nn.Linear(_metatensor.tensor(wf).reshape([feat, NC]),
                              _metatensor.tensor([0.01] * NC)))
        y = cnn(_metatensor.tensor(x).reshape([1, C * H * W]))
        assert y.shape == (1, NC)

        def at(c, h, w):
            if 0 <= h < H and 0 <= w < W:
                return x[c * H * W + h * W + w]
            return 0.0

        inv = 1.0 / math.sqrt(1.0 + 1e-5)
        planes = []
        for o in range(O):
            plane = []
            for h in range(H):
                row = []
                for w in range(W):
                    acc = 0.01
                    for c in range(C):
                        for r in range(3):
                            for s in range(3):
                                acc += (at(c, h + r - 1, w + s - 1) *
                                        wc[(c * 9 + r * 3 + s) * O + o])
                    row.append(max(acc * inv, 0.0))
                plane.append(row)
            planes.append(plane)
        flat = []
        for o in range(O):
            for oh in range(H // 2):
                for ow in range(W // 2):
                    p = planes[o]
                    flat.append(max(p[2 * oh][2 * ow], p[2 * oh][2 * ow + 1],
                                    p[2 * oh + 1][2 * ow],
                                    p[2 * oh + 1][2 * ow + 1]))
        exp = 0.0
        for j in range(NC):
            acc = 0.01
            for i in range(feat):
                acc += flat[i] * wf[i * NC + j]
            exp += max(acc, 0.0)
        assert abs(y.sum().item() - exp) < 1e-9

    def test_backward_new_ops(self):
        import _metatensor, math
        x = _metatensor.tensor([1.0, 4.0, 9.0], requires_grad=True)
        x.sqrt().sum().backward()
        assert abs(x.grad.sum().item() - (0.5 + 0.25 + 1.0 / 6.0)) < 1e-12

        e = _metatensor.tensor([0.0, 1.0], requires_grad=True)
        e.exp().sum().backward()
        assert abs(e.grad.sum().item() - (1.0 + math.e)) < 1e-12

        a = _metatensor.tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
        b = _metatensor.tensor([2.0], requires_grad=True)
        a.div(b).sum().backward()
        assert abs(a.grad.sum().item() - 2.0) < 1e-12
        assert abs(b.grad.sum().item() + 2.5) < 1e-12

        s = _metatensor.tensor([[1.0, 2.0]], requires_grad=True)
        t = _metatensor.tensor([[0.25, 0.5]], requires_grad=True)
        s.sub(t).sum().backward()
        assert s.grad.sum().item() == 2.0
        assert t.grad.sum().item() == -2.0

        m = _metatensor.tensor([[1.0, 5.0], [7.0, 2.0]], requires_grad=True)
        m.max(1).sum().backward()
        assert m.grad.sum().item() == 2.0

        p = _metatensor.tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
        q = _metatensor.tensor([[1.0, 10.0], [100.0, 1000.0]])
        p.matmul(q, True).sum().backward()
        assert p.grad.sum().item() == 2222.0

        h = _metatensor.tensor([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]],
                           requires_grad=True)
        h.head_split(2).head_merge(2).sum().backward()
        assert h.grad.sum().item() == 8.0

        u = _metatensor.tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
        eye = _metatensor.tensor([[1.0, 0.0], [0.0, 1.0]])
        u.bmm(eye, 1).sum().backward()
        assert u.grad.sum().item() == 4.0

    def test_backward_softmax_rows(self):
        import _metatensor
        from tensorpypy.functional import softmax
        x = _metatensor.tensor([[1.0, 2.0, 3.0], [0.5, -1.0, 2.5]],
                           requires_grad=True)
        w = _metatensor.tensor([[1.0, -2.0, 0.5], [0.25, 1.5, -1.0]])
        softmax(x).mul(w).sum().backward()
        g = x.grad
        rows = g.mul(_metatensor.tensor([1.0, 1.0, 1.0])).sum(1)
        assert abs(rows.mul(_metatensor.tensor([1.0, 1.0])).sum().item()) < 1e-12

    def test_dtype_and_astype(self):
        import _metatensor
        a = _metatensor.tensor([1.0, 2.0, 3.0])
        assert a.dtype == "float64"
        b = _metatensor.tensor([1.0, 2.0, 3.0], dtype="float32")
        assert b.dtype == "float32"
        assert abs(b.sum().item() - 6.0) < 1e-4
        c = b.astype("float64")
        assert c.dtype == "float64"
        assert abs(c.sum().item() - 6.0) < 1e-9
        h = a.astype("float16")
        assert h.dtype == "float16"
        assert h.shape == (3,)
        z = _metatensor.zeros([2, 2], False, "float32")
        assert z.dtype == "float32"
        raises(ValueError, _metatensor.tensor, [1.0], None, False, "float8")
        raises(ValueError, b.add, a)

    def test_gelu(self):
        import _metatensor, math
        from tensorpypy.functional import gelu
        x = _metatensor.tensor([-2.0, -0.5, 0.0, 0.5, 2.0])
        y = gelu(x)
        want = 0.0
        for v in [-2.0, -0.5, 0.0, 0.5, 2.0]:
            z = math.sqrt(2.0 / math.pi) * (v + 0.044715 * v ** 3)
            want += 0.5 * v * (1.0 + math.tanh(z))
        assert abs(y.sum().item() - want) < 1e-9

    def test_gpt2_causal(self):
        import _metatensor
        from tensorpypy.models import CausalSelfAttention, GPT2MLP, GPT2Block, GPT2
        v, d, h, t = 4, 4, 2, 3

        def mat(rows, cols, k):
            data = [float(((i * k) % 7) - 3) / 8.0 for i in range(rows * cols)]
            return _metatensor.tensor(data, [rows, cols])

        def vec(n, val):
            return _metatensor.tensor([val] * n)

        mask = [0.0] * (h * t * t)
        for head in range(h):
            for i in range(t):
                for j in range(i + 1, t):
                    mask[(head * t + i) * t + j] = -1e9
        mask = _metatensor.tensor(mask, [h * t, t])
        wqkv = []
        for r in range(d):
            for k in [3, 5, 2]:
                wqkv.extend([float(((i * k) % 7) - 3) / 8.0
                             for i in range(r * d, (r + 1) * d)])
        attn = CausalSelfAttention(
            _metatensor.tensor(wqkv, [d, 3 * d]), vec(3 * d, 0.0),
            mat(d, d, 4), vec(d, 0.0), h, mask)
        mlp = GPT2MLP(mat(d, 4 * d, 3), vec(4 * d, 0.0),
                                 mat(4 * d, d, 5), vec(d, 0.0))
        block = GPT2Block(attn, vec(d, 1.0), vec(d, 0.0),
                                     vec(d, 1.0), vec(d, 0.0), mlp)
        model = GPT2(mat(v, d, 3), [block], vec(d, 1.0),
                                vec(d, 0.0))
        pos = mat(t, d, 7)
        pick = _metatensor.tensor([1.0] + [0.0] * (t * v - 1), [t, v])

        def run(last):
            idx = _metatensor.tensor([1.0, 2.0, float(last)])
            return model(idx, pos)

        a = run(0)
        b = run(3)
        assert a.shape == (t, v)
        assert abs(a.mul(pick).sum().item() -
                   b.mul(pick).sum().item()) < 1e-12
        assert abs(a.sum().item() - b.sum().item()) > 1e-9

    def test_attn_strided_slices(self):
        import _metatensor
        rows, d, h = 3, 4, 2

        def gen(k):
            return [float(((i * k) % 11) - 5) / 8.0 for i in range(rows * d)]

        q, k, v = gen(3), gen(5), gen(2)
        fused = []
        for r in range(rows):
            for part in (q, k, v):
                fused.extend(part[r * d:(r + 1) * d])
        qkv = _metatensor.tensor(fused, [rows, 3 * d])
        qt = _metatensor.tensor(q, [rows, d])
        kt = _metatensor.tensor(k, [rows, d])
        vt = _metatensor.tensor(v, [rows, d])
        s0 = qt.attn_scores(kt, h).tolist()
        s1 = qkv.attn_scores(qkv, h, d, 0, d).tolist()
        assert len(s0) == len(s1)
        for a, b in zip(s0, s1):
            assert abs(a - b) < 1e-9
        p0 = _metatensor.tensor(s0, [h * rows, rows])
        c0 = p0.attn_context(vt, h).tolist()
        c1 = p0.attn_context(qkv, h, d, 2 * d).tolist()
        for a, b in zip(c0, c1):
            assert abs(a - b) < 1e-9

    def test_take_rows(self):
        import _metatensor
        table = _metatensor.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        idx = _metatensor.tensor([2.0, 0.0, 2.0])
        r = table.take(idx)
        assert r.shape == (3, 2)
        assert r.sum().item() == 25.0
        pick = _metatensor.tensor([[1.0, 0.0], [0.0, 0.0], [0.0, 0.0]])
        assert r.mul(pick).sum().item() == 5.0
