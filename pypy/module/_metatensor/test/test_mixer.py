import os
os.environ.setdefault('RTENSOR_CPU', '1')


class AppTestMixer(object):
    spaceconfig = dict(usemodules=['_metatensor'])

    def test_token_mixing_matches_reference(self):
        import _metatensor
        from tensorpypy.functional import gelu_erf
        from tensorpypy.models import GPT2MLP, MixerBlock
        t, ch, hid = 3, 2, 4

        def mat(rows, cols, k):
            return [float(((i * k) % 7) - 3) / 8.0 for i in range(rows * cols)]

        x = mat(t, ch, 3)
        w1 = mat(hid, t, 5)
        w2 = mat(t, hid, 2)
        b1 = [0.25, -0.5, 0.75, 0.0]
        b2 = [0.1, -0.1, 0.2]
        zeros = _metatensor.tensor([0.0] * (ch * ch), [ch, ch])
        mlp = GPT2MLP(zeros, _metatensor.tensor([0.0] * ch), zeros,
                      _metatensor.tensor([0.0] * ch), gelu_erf)
        block = MixerBlock(
            _metatensor.tensor([1.0] * ch), _metatensor.tensor([0.0] * ch),
            _metatensor.tensor(w1, [hid, t]),
            _metatensor.tensor(b1, [hid, 1]),
            _metatensor.tensor(w2, [t, hid]),
            _metatensor.tensor(b2, [t, 1]),
            _metatensor.tensor([1.0] * ch), _metatensor.tensor([0.0] * ch),
            mlp)
        got = block(_metatensor.tensor(x, [t, ch])).tolist()
        import math
        norm = []
        for i in range(t):
            row = x[i * ch:(i + 1) * ch]
            mu = sum(row) / ch
            var = sum((v - mu) ** 2 for v in row) / ch
            norm.extend([(v - mu) / math.sqrt(var + 1e-6) for v in row])
        u = []
        for i in range(hid):
            for j in range(ch):
                acc = b1[i]
                for p in range(t):
                    acc += w1[i * t + p] * norm[p * ch + j]
                u.append(0.5 * acc * (1.0 + math.erf(acc / math.sqrt(2.0))))
        want = []
        for i in range(t):
            for j in range(ch):
                acc = b2[i]
                for p in range(hid):
                    acc += w2[i * hid + p] * u[p * ch + j]
                want.append(x[i * ch + j] + acc)
        for a, e in zip(got, want):
            assert abs(a - e) < 1e-6
