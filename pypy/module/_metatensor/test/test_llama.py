import os
os.environ.setdefault('RTENSOR_CPU', '1')


class AppTestLlama(object):
    spaceconfig = dict(usemodules=['_metatensor'])

    def test_llama_rmsnorm(self):
        import _metatensor, math
        from tensorpypy.functional import rms_norm, silu
        rows, cols, eps = 3, 4, 1e-5
        data = [[0.5, -1.5, 2.0, 0.25], [1.0, 1.0, 1.0, 1.0],
                [-3.0, 0.5, 0.125, 2.0]]
        g = [0.5, 1.0, 1.5, 2.0]
        flat = [v for row in data for v in row]
        y = rms_norm(_metatensor.tensor(flat, [rows, cols]),
                               _metatensor.tensor(g), eps).tolist()
        for i in range(rows):
            d = math.sqrt(sum(v * v for v in data[i]) / cols + eps)
            for j in range(cols):
                assert abs(y[i * cols + j] - data[i][j] / d * g[j]) < 1e-9

    def test_llama_silu(self):
        import _metatensor, math
        from tensorpypy.functional import rms_norm, silu
        xs = [-4.0, -0.5, 0.0, 0.5, 3.0, 7.0]
        y = silu(_metatensor.tensor(xs)).tolist()
        for i, x in enumerate(xs):
            assert abs(y[i] - x / (1.0 + math.exp(-x))) < 1e-9

    def test_llama_rope(self):
        import _metatensor, math
        heads, dh, seq, theta = 2, 4, 3, 10000.0
        d = heads * dh
        half = dh // 2
        inv = [1.0 / theta ** (2.0 * i / dh) for i in range(half)]
        cos, sin = [], []
        for pos in range(seq):
            c = [math.cos(pos * f) for f in inv]
            s = [math.sin(pos * f) for f in inv]
            cos.extend((c + c) * heads)
            sin.extend(([-v for v in s] + s) * heads)
        x = [(i * 37 % 17) * 0.1 - 0.8 for i in range(seq * d)]
        xt = _metatensor.tensor(x, [seq, d])
        y = xt.mul(_metatensor.tensor(cos, [seq, d])).add(
            xt.rot_half(dh).mul(
                _metatensor.tensor(sin, [seq, d]))).tolist()
        for pos in range(seq):
            for h in range(heads):
                o = pos * d + h * dh
                for j in range(dh):
                    rot = (-x[o + j + half] if j < half else x[o + j - half])
                    want = (x[o + j] * cos[pos * d + h * dh + j] +
                            rot * abs(sin[pos * d + h * dh + j]))
                    assert abs(y[o + j] - want) < 1e-6

    def test_rot_half_involution(self):
        import _metatensor
        x = [float(i) for i in range(12)]
        t = _metatensor.tensor(x, [2, 6])
        assert t.rot_half(6).rot_half(6).tolist() == x
        assert t.rot_half(6).tolist() == [3.0, 4.0, 5.0, 0.0, 1.0, 2.0,
                                          9.0, 10.0, 11.0, 6.0, 7.0, 8.0]
