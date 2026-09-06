import os
os.environ.setdefault('RTENSOR_CPU', '1')


class AppTestBert(object):
    spaceconfig = dict(usemodules=['_metatensor'])

    def test_gelu_erf_matches_math_erf(self):
        import _metatensor, math
        from tensorpypy.functional import gelu_erf
        xs = [-6.0, -3.0, -1.0, -0.25, 0.0, 0.25, 1.0, 3.0, 6.0]
        y = gelu_erf(_metatensor.tensor(xs)).tolist()
        for i, x in enumerate(xs):
            want = 0.5 * x * (1.0 + math.erf(x / math.sqrt(2.0)))
            assert abs(y[i] - want) < 1e-6

    def test_bert_is_bidirectional(self):
        import _metatensor
        from tensorpypy.functional import gelu_erf
        from tensorpypy.models import (CausalSelfAttention, GPT2MLP,
                                       BertBlock, Bert)
        v, d, h, t = 5, 4, 2, 3

        def mat(rows, cols, k):
            data = [float(((i * k) % 7) - 3) / 8.0 for i in range(rows * cols)]
            return _metatensor.tensor(data, [rows, cols])

        def vec(n, val):
            return _metatensor.tensor([val] * n)

        wqkv = []
        for r in range(d):
            for k in [3, 5, 2]:
                wqkv.extend([float(((i * k) % 7) - 3) / 8.0
                             for i in range(r * d, (r + 1) * d)])
        attn = CausalSelfAttention(_metatensor.tensor(wqkv, [d, 3 * d]),
                                   vec(3 * d, 0.0), mat(d, d, 4),
                                   vec(d, 0.0), h, None)
        mlp = GPT2MLP(mat(d, 4 * d, 3), vec(4 * d, 0.0), mat(4 * d, d, 5),
                      vec(d, 0.0), gelu_erf)
        block = BertBlock(attn, vec(d, 1.0), vec(d, 0.0), mlp, vec(d, 1.0),
                          vec(d, 0.0))
        model = Bert(mat(v, d, 3), mat(t, d, 7), vec(d, 1.0), vec(d, 0.0),
                     [block], mat(d, d, 5), vec(d, 0.0), vec(d, 1.0),
                     vec(d, 0.0), vec(v, 0.0))
        pick = _metatensor.tensor([1.0] + [0.0] * (t * v - 1), [t, v])

        def run(last):
            return model(_metatensor.tensor([1.0, 2.0, float(last)]))

        a = run(0)
        b = run(4)
        assert a.shape == (t, v)
        assert abs(a.mul(pick).sum().item() -
                   b.mul(pick).sum().item()) > 1e-9
