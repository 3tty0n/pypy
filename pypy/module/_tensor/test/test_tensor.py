import os
os.environ.setdefault('RTENSOR_CPU', '1')


class AppTestTensor(object):
    spaceconfig = dict(usemodules=['_tensor'])

    def test_construct(self):
        import _tensor
        t = _tensor.tensor([1.0, 2.0, 3.0])
        assert t.shape == (3,)
        assert t.size == 3
        assert t.requires_grad is False
        m = _tensor.tensor([[1.0, 2.0], [3.0, 4.0]])
        assert m.shape == (2, 2)

    def test_zeros(self):
        import _tensor
        z = _tensor.zeros([2, 3])
        assert z.shape == (2, 3)
        assert z.sum().item() == 0.0

    def test_add_mul_broadcast(self):
        import _tensor
        a = _tensor.tensor([1.0, 2.0, 3.0])
        b = _tensor.tensor([10.0])
        c = a.add(b)
        assert c.sum().item() == 36.0
        d = a.mul(_tensor.tensor([2.0]))
        assert d.sum().item() == 12.0
        e = a + b
        assert e.sum().item() == 36.0

    def test_sum_axis(self):
        import _tensor
        m = _tensor.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        assert m.sum().item() == 21.0
        s0 = m.sum(0)
        w0 = _tensor.tensor([1.0, 2.0, 3.0])
        assert s0.mul(w0).sum().item() == 46.0
        s1 = m.sum(1)
        w1 = _tensor.tensor([1.0, 2.0])
        assert s1.mul(w1).sum().item() == 36.0

    def test_matmul(self):
        import _tensor
        x = _tensor.tensor([[1.0, 2.0], [3.0, 4.0]])
        w = _tensor.tensor([[1.0, 0.0], [0.0, 1.0]])
        y = x.matmul(w)
        assert y.sum().item() == 10.0

    def test_reshape(self):
        import _tensor
        t = _tensor.tensor([1.0, 2.0, 3.0, 4.0])
        r = t.reshape([2, 2])
        assert r.shape == (2, 2)
        assert r.sum().item() == 10.0

    def test_shape_mismatch(self):
        import _tensor
        a = _tensor.tensor([1.0, 2.0, 3.0])
        b = _tensor.tensor([1.0, 2.0])
        raises(ValueError, a.add, b)
        raises(ValueError, a.reshape, [3, 3])

    def test_backward(self):
        import _tensor
        x = _tensor.tensor([1.0, -2.0, 3.0], requires_grad=True)
        w = _tensor.tensor([2.0, 1.0, -1.0], requires_grad=True)
        b = _tensor.tensor([0.5, 0.5, 0.5], requires_grad=True)
        z = x.mul(w).add(b).relu()
        loss = z.sum()
        assert loss.item() == 2.5
        assert x.grad is None
        loss.backward()
        assert x.grad.sum().item() == 2.0
        assert w.grad.sum().item() == 1.0
        assert b.grad.sum().item() == 1.0

    def test_mlp_training_step(self):
        import _tensor
        import tensorlite

        w1 = _tensor.tensor([[1.0, -1.0], [0.5, 0.5]], requires_grad=True)
        b1 = _tensor.tensor([0.0, 0.0], requires_grad=True)
        w2 = _tensor.tensor([[1.0], [1.0]], requires_grad=True)
        b2 = _tensor.tensor([0.0], requires_grad=True)
        mlp = tensorlite.MLP([tensorlite.Linear(w1, b1),
                               tensorlite.Linear(w2, b2)])
        x = _tensor.tensor([[1.0, 2.0]])
        target = _tensor.tensor([[0.0]])

        def loss_fn():
            y = mlp(x)
            d = y.add(target.mul(_tensor.tensor([-1.0])))
            return d.mul(d).sum()

        loss0 = loss_fn().item()
        for i in range(5):
            loss = loss_fn()
            loss.backward()
            tensorlite.sgd_step(mlp.parameters(), 0.05)
        loss1 = loss_fn().item()
        assert loss1 < loss0

    def test_sub_div_exp_sqrt_max(self):
        import _tensor
        a = _tensor.tensor([1.0, 4.0, 9.0])
        b = _tensor.tensor([1.0, 2.0, 3.0])
        assert a.sub(b).sum().item() == 8.0
        assert (a - b).sum().item() == 8.0
        assert a.div(b).sum().item() == 6.0
        assert (a / b).sum().item() == 6.0
        assert a.sqrt().sum().item() == 6.0
        assert abs(b.exp().sum().item() - 30.19287485057736) < 1e-12
        assert a.max().item() == 9.0
        m = _tensor.tensor([[1.0, 5.0], [7.0, 2.0]])
        assert m.max(1).sum().item() == 12.0
        assert m.max(0).sum().item() == 12.0

    def test_matmul_transpose_b(self):
        import _tensor
        x = _tensor.tensor([[1.0, 2.0], [3.0, 4.0]])
        w = _tensor.tensor([[1.0, 10.0], [100.0, 1000.0]])
        assert x.matmul(w, True).sum().item() == 6464.0

    def test_softmax_layernorm(self):
        import _tensor, tensorlite, math
        x = _tensor.tensor([[1.0, 2.0, 3.0], [1.0, 1.0, 1.0]])
        s = tensorlite.softmax(x)
        assert abs(s.sum().item() - 2.0) < 1e-12
        e = [math.exp(v - 3.0) for v in [1.0, 2.0, 3.0]]
        tot = sum(e)
        pick = _tensor.tensor([[1.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
        assert abs(s.mul(pick).sum().item() - e[0] / tot) < 1e-12
        one = _tensor.tensor([1.0, 1.0, 1.0])
        zero = _tensor.tensor([0.0, 0.0, 0.0])
        y = tensorlite.layernorm(x, one, zero)
        assert abs(y.sum().item()) < 1e-9
        std = math.sqrt(2.0 / 3.0)
        assert abs(y.mul(pick).sum().item() + 1.0 / std) < 1e-4

    def test_cnn_forward(self):
        import _tensor, tensorlite, math
        C, H, W, O, NC = 2, 4, 4, 3, 5
        fan = C * 9
        feat = O * (H // 2) * (W // 2)
        wc = [float((i * 7) % 13 - 6) / fan for i in range(fan * O)]
        wf = [float((i * 7) % 13 - 6) / feat for i in range(feat * NC)]
        x = [float(i % 5) - 2.0 for i in range(C * H * W)]
        cnn = tensorlite.CNN(
            tensorlite.Conv2d(_tensor.tensor(wc).reshape([fan, O]),
                              _tensor.tensor([0.01] * O), C, H, W),
            tensorlite.BatchNorm2d(O),
            tensorlite.MaxPool2d(O, H, W),
            tensorlite.Linear(_tensor.tensor(wf).reshape([feat, NC]),
                              _tensor.tensor([0.01] * NC)))
        y = cnn(_tensor.tensor(x).reshape([1, C * H * W]))
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
        import _tensor, math
        x = _tensor.tensor([1.0, 4.0, 9.0], requires_grad=True)
        x.sqrt().sum().backward()
        assert abs(x.grad.sum().item() - (0.5 + 0.25 + 1.0 / 6.0)) < 1e-12

        e = _tensor.tensor([0.0, 1.0], requires_grad=True)
        e.exp().sum().backward()
        assert abs(e.grad.sum().item() - (1.0 + math.e)) < 1e-12

        a = _tensor.tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
        b = _tensor.tensor([2.0], requires_grad=True)
        a.div(b).sum().backward()
        assert abs(a.grad.sum().item() - 2.0) < 1e-12
        assert abs(b.grad.sum().item() + 2.5) < 1e-12

        s = _tensor.tensor([[1.0, 2.0]], requires_grad=True)
        t = _tensor.tensor([[0.25, 0.5]], requires_grad=True)
        s.sub(t).sum().backward()
        assert s.grad.sum().item() == 2.0
        assert t.grad.sum().item() == -2.0

        m = _tensor.tensor([[1.0, 5.0], [7.0, 2.0]], requires_grad=True)
        m.max(1).sum().backward()
        assert m.grad.sum().item() == 2.0

        p = _tensor.tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
        q = _tensor.tensor([[1.0, 10.0], [100.0, 1000.0]])
        p.matmul(q, True).sum().backward()
        assert p.grad.sum().item() == 2222.0

        h = _tensor.tensor([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]],
                           requires_grad=True)
        h.head_split(2).head_merge(2).sum().backward()
        assert h.grad.sum().item() == 8.0

        u = _tensor.tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
        eye = _tensor.tensor([[1.0, 0.0], [0.0, 1.0]])
        u.bmm(eye, 1).sum().backward()
        assert u.grad.sum().item() == 4.0

    def test_backward_softmax_rows(self):
        import _tensor, tensorlite
        x = _tensor.tensor([[1.0, 2.0, 3.0], [0.5, -1.0, 2.5]],
                           requires_grad=True)
        w = _tensor.tensor([[1.0, -2.0, 0.5], [0.25, 1.5, -1.0]])
        tensorlite.softmax(x).mul(w).sum().backward()
        g = x.grad
        rows = g.mul(_tensor.tensor([1.0, 1.0, 1.0])).sum(1)
        assert abs(rows.mul(_tensor.tensor([1.0, 1.0])).sum().item()) < 1e-12
