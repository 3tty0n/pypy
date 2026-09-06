import os
os.environ.setdefault('RTENSOR_CPU', '1')


class AppTestResNet(object):
    spaceconfig = dict(usemodules=['_metatensor'])

    def test_im2col_stride_pad(self):
        import _metatensor
        c, h, w, k, pad, stride = 2, 5, 5, 3, 1, 2
        vals = [float((i * 7) % 13) - 6.0 for i in range(c * h * w)]
        got = _metatensor.tensor(vals, [1, c * h * w]).im2col(
            c, h, w, k, pad, stride).tolist()
        oh = (h + 2 * pad - k) // stride + 1
        ow = (w + 2 * pad - k) // stride + 1
        want = []
        for ph in range(oh):
            for pw in range(ow):
                for ci in range(c):
                    for a in range(k):
                        for b in range(k):
                            ih = ph * stride + a - pad
                            iw = pw * stride + b - pad
                            if 0 <= ih < h and 0 <= iw < w:
                                want.append(vals[(ci * h + ih) * w + iw])
                            else:
                                want.append(0.0)
        assert got == want

    def test_maxpool_3x3_stride2_pad1(self):
        import _metatensor
        c, h, w, k, stride, pad = 2, 5, 5, 3, 2, 1
        vals = [float((i * 5) % 11) for i in range(c * h * w)]
        got = _metatensor.tensor(vals, [1, c * h * w]).maxpool2(
            c, h, w, k, stride, pad).tolist()
        oh = (h + 2 * pad - k) // stride + 1
        want = []
        for ci in range(c):
            for ph in range(oh):
                for pw in range(oh):
                    m = None
                    for a in range(k):
                        for b in range(k):
                            ih = ph * stride + a - pad
                            iw = pw * stride + b - pad
                            if 0 <= ih < h and 0 <= iw < w:
                                v = vals[(ci * h + ih) * w + iw]
                                if m is None or v > m:
                                    m = v
                    want.append(m)
        assert got == want

    def test_im2col_nhwc_stride_pad(self):
        import _metatensor
        c, h, w, k, pad, stride = 2, 5, 5, 3, 1, 2
        vals = [float((i * 7) % 13) - 6.0 for i in range(h * w * c)]
        got = _metatensor.tensor(vals, [h * w, c]).im2col_nhwc(
            c, h, w, k, pad, stride).tolist()
        oh = (h + 2 * pad - k) // stride + 1
        ow = (w + 2 * pad - k) // stride + 1
        want = []
        for ph in range(oh):
            for pw in range(ow):
                for a in range(k):
                    for b in range(k):
                        for ci in range(c):
                            ih = ph * stride + a - pad
                            iw = pw * stride + b - pad
                            if 0 <= ih < h and 0 <= iw < w:
                                want.append(vals[(ih * w + iw) * c + ci])
                            else:
                                want.append(0.0)
        assert got == want

    def test_maxpool_nhwc_3x3_stride2_pad1(self):
        import _metatensor
        c, h, w, k, stride, pad = 2, 5, 5, 3, 2, 1
        vals = [float((i * 5) % 11) for i in range(h * w * c)]
        got = _metatensor.tensor(vals, [h * w, c]).maxpool2_nhwc(
            c, h, w, k, stride, pad).tolist()
        oh = (h + 2 * pad - k) // stride + 1
        want = []
        for ph in range(oh):
            for pw in range(oh):
                for ci in range(c):
                    m = None
                    for a in range(k):
                        ih = min(max(ph * stride + a - pad, 0), h - 1)
                        for b in range(k):
                            iw = min(max(pw * stride + b - pad, 0), w - 1)
                            v = vals[(ih * w + iw) * c + ci]
                            if m is None or v > m:
                                m = v
                    want.append(m)
        assert got == want

    def test_resnet_block_nhwc_matches_reference(self):
        import _metatensor
        from tensorpypy.nn import BatchNorm2d, Conv2d
        from tensorpypy.models import ResNetBlock
        c, o, h, k = 2, 2, 4, 3
        x = [float((i * 3) % 7) - 3.0 for i in range(h * h * c)]
        w1 = [float((i * 5) % 9) / 8.0 - 0.5 for i in range(k * k * c * o)]
        w2 = [float((i * 7) % 9) / 8.0 - 0.5 for i in range(k * k * o * o)]
        g = [1.25, 0.75]
        b = [0.1, -0.2]
        mean = [0.05, -0.1]
        var = [0.9, 1.3]
        eps = 1e-5

        def mk_bn():
            return BatchNorm2d(o, g, b, mean, var, eps, None, True)

        conv1 = Conv2d(_metatensor.tensor(w1, [k * k * c, o]), None, c, h, h,
                       k, 1, 1, True)
        conv2 = Conv2d(_metatensor.tensor(w2, [k * k * o, o]), None, o, h, h,
                       k, 1, 1, True)
        block = ResNetBlock(conv1, mk_bn(), conv2, mk_bn())
        got = block(_metatensor.tensor(x, [h * h, c])).tolist()

        def conv(src, wt, ci):
            out = []
            for ph in range(h):
                for pw in range(h):
                    for oc in range(o):
                        acc = 0.0
                        for a in range(k):
                            for bb in range(k):
                                for cc in range(ci):
                                    ih = ph + a - 1
                                    iw = pw + bb - 1
                                    if 0 <= ih < h and 0 <= iw < h:
                                        acc += (src[(ih * h + iw) * ci + cc] *
                                                wt[((a * k + bb) * ci + cc) *
                                                   o + oc])
                        out.append(acc)
            return out

        def bn(src):
            import math
            out = []
            for i in range(len(src)):
                ci = i % o
                s = g[ci] / math.sqrt(var[ci] + eps)
                out.append(src[i] * s + b[ci] - mean[ci] * s)
            return out

        y = bn(conv(x, w1, c))
        y = [v if v > 0.0 else 0.0 for v in y]
        y = bn(conv(y, w2, o))
        want = [max(y[i] + x[i], 0.0) for i in range(len(y))]
        for a, e in zip(got, want):
            assert abs(a - e) < 1e-9
