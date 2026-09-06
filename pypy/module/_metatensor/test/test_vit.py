import os
os.environ.setdefault('RTENSOR_CPU', '1')


class AppTestViT(object):
    spaceconfig = dict(usemodules=['_metatensor'])

    def test_patch_gather_with_cls_row(self):
        import _metatensor
        size, k, d = 4, 2, 2
        grid = size // k
        patch = 3 * k * k
        pixels = [float(i % 17) for i in range(3 * size * size)]
        table = pixels + [0.0] * k
        img = _metatensor.tensor(table, [len(table) // k, k])
        idx = [float(3 * size * grid)] * (3 * k)
        for ph in range(grid):
            for pw in range(grid):
                for c in range(3):
                    for kh in range(k):
                        idx.append(float(((c * size) + ph * k + kh) * grid +
                                         pw))
        rows = grid * grid + 1
        got = img.take(_metatensor.tensor(idx)).reshape(
            [rows, patch]).tolist()
        assert got[:patch] == [0.0] * patch
        for ph in range(grid):
            for pw in range(grid):
                want = []
                for c in range(3):
                    for a in range(k):
                        for b in range(k):
                            want.append(pixels[(c * size + ph * k + a) *
                                               size + pw * k + b])
                off = (1 + ph * grid + pw) * patch
                assert got[off:off + patch] == want
        w = _metatensor.tensor([0.0] * (patch * d), [patch, d])
        emb = _metatensor.tensor([1.0] * (rows * d), [rows, d])
        x = img.take(_metatensor.tensor(idx)).reshape(
            [rows, patch]).matmul(w).add(emb)
        assert x.shape == (rows, d)
        assert x.sum().item() == rows * d
