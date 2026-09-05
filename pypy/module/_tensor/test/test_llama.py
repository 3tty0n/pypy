import os
os.environ.setdefault('RTENSOR_CPU', '1')


class AppTestLlama(object):
    spaceconfig = dict(usemodules=['_tensor'])

    def test_llama_rmsnorm(self):
        import _tensor, tensorlite, math
        rows, cols, eps = 3, 4, 1e-5
        data = [[0.5, -1.5, 2.0, 0.25], [1.0, 1.0, 1.0, 1.0],
                [-3.0, 0.5, 0.125, 2.0]]
        g = [0.5, 1.0, 1.5, 2.0]
        flat = [v for row in data for v in row]
        y = tensorlite.rmsnorm(_tensor.tensor(flat, [rows, cols]),
                               _tensor.tensor(g), eps).tolist()
        for i in range(rows):
            d = math.sqrt(sum(v * v for v in data[i]) / cols + eps)
            for j in range(cols):
                assert abs(y[i * cols + j] - data[i][j] / d * g[j]) < 1e-9

    def test_llama_silu(self):
        import _tensor, tensorlite, math
        xs = [-4.0, -0.5, 0.0, 0.5, 3.0, 7.0]
        y = tensorlite.silu(_tensor.tensor(xs)).tolist()
        for i, x in enumerate(xs):
            assert abs(y[i] - x / (1.0 + math.exp(-x))) < 1e-9

    def test_llama_rope(self):
        import _tensor, tensorlite, math
        heads, dh, seq, theta = 2, 4, 3, 10000.0
        d = heads * dh
        half = dh // 2
        inv = [1.0 / theta ** (2.0 * i / dh) for i in range(half)]
        cos, sin = [], []
        for pos in range(seq):
            c = [math.cos(pos * f) for f in inv]
            s = [math.sin(pos * f) for f in inv]
            cos.extend((c + c) * heads)
            sin.extend((s + s) * heads)
        p = [0.0] * (d * d)
        for h in range(heads):
            o = h * dh
            for j in range(dh):
                if j < half:
                    p[(o + j + half) * d + o + j] = -1.0
                else:
                    p[(o + j - half) * d + o + j] = 1.0
        x = [(i * 37 % 17) * 0.1 - 0.8 for i in range(seq * d)]
        y = tensorlite.rope(_tensor.tensor(x, [seq, d]),
                            _tensor.tensor(cos, [seq, d]),
                            _tensor.tensor(sin, [seq, d]),
                            _tensor.tensor(p, [d, d])).tolist()
        for pos in range(seq):
            for h in range(heads):
                o = pos * d + h * dh
                for j in range(dh):
                    rot = (-x[o + j + half] if j < half else x[o + j - half])
                    want = (x[o + j] * cos[pos * d + h * dh + j] +
                            rot * sin[pos * d + h * dh + j])
                    assert abs(y[o + j] - want) < 1e-6
