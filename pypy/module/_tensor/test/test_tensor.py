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
