import common

from tensorpypy.nn import BatchNorm2d, Conv2d, MaxPool2d
from tensorpypy.models import ResNet, ResNetBlock


def build(cfg, buf, dtype, batch):
    w = common.Weights(cfg, buf, dtype)
    eps = cfg['eps']

    def bn(name, c):
        return BatchNorm2d(c, w.raw(name + '.g')[0], w.raw(name + '.b')[0],
                           w.raw(name + '.m')[0], w.raw(name + '.v')[0], eps,
                           dtype, True)

    s = cfg['image_size']
    pixels, _ = w.raw('image')
    x = common.tensor(pixels * batch, [batch * s * s, 3], False, dtype)
    ow, ok = cfg['stem'][0][0], cfg['stem'][1]
    conv1 = Conv2d(w.get('conv1.w'), None, 3, s, s, ok, 2, ok // 2, True)
    pool = MaxPool2d(ow, conv1.oh, conv1.ow, 3, 2, 1, True)
    h = pool.oh
    blocks = []
    for li, n, shape, k, stride, down in cfg['layers']:
        p = 'l%d.%d.' % (li, n)
        o, c = shape[0], shape[1]
        c1 = Conv2d(w.get(p + 'conv1.w'), None, c, h, h, k, stride, k // 2,
                    True)
        c2 = Conv2d(w.get(p + 'conv2.w'), None, o, c1.oh, c1.ow, k, 1, k // 2,
                    True)
        d = bnd = None
        if down:
            d = Conv2d(w.get(p + 'down.w'), None, c, h, h, 1, stride, 0, True)
            bnd = bn(p + 'bnd', o)
        blocks.append(ResNetBlock(c1, bn(p + 'bn1', o), c2,
                                  bn(p + 'bn2', o), d, bnd))
        h = c2.oh
    hw = h * h
    avg = [0.0] * (batch * batch * hw)
    for i in range(batch):
        for j in range(hw):
            avg[i * (batch * hw) + i * hw + j] = 1.0 / hw
    mean = common.tensor(avg, [batch, batch * hw], False, dtype)
    model = ResNet(conv1, bn('bn1', ow), pool, blocks, shape[0], hw,
                   w.get('fc.w'), w.get('fc.b'), mean)
    return model, x


def main():
    outdir, iters, warmup, dtype = common.argv()
    batch = common.batch_argv()
    cfg, buf = common.load(outdir)
    model, x = build(cfg, buf, dtype, batch)
    logits, acc, steady_us = common.timed(model, (x,), iters, warmup)
    flat = logits.tolist()[:cfg['classes']]
    common.dump(outdir, flat)
    common.report(
        'resnet pypy model=%s batch=%d classes=%d dtype=%s iters=%d '
        'steady_us=%.1f checksum=%.6f' %
        (cfg['source'], batch, cfg['classes'], dtype, iters, steady_us, acc),
        common.top5(flat))


if __name__ == '__main__':
    main()
