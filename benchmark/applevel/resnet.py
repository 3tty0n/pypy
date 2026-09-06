import array, json, os, sys, time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                '..', '..', 'lib_pypy'))

import _metatensor
from tensorpypy.nn import BatchNorm2d, Conv2d, MaxPool2d
from tensorpypy.models import ResNet, ResNetBlock


def load(outdir):
    cfg = json.load(open(os.path.join(outdir, 'index.json')))
    buf = array.array('f')
    path = os.path.join(outdir, 'weights.bin')
    buf.fromfile(open(path, 'rb'), os.path.getsize(path) // 4)
    if sys.byteorder != 'little':
        buf.byteswap()
    return cfg, buf


def build(cfg, buf, dtype, batch):
    index = cfg['index']
    eps = cfg['eps']

    def raw(name):
        off, shape = index[name]
        n = 1
        for d in shape:
            n *= d
        return list(buf[off:off + n]), shape

    def get(name):
        values, shape = raw(name)
        return _metatensor.tensor(values, shape, False, dtype)

    def bn(name, c):
        return BatchNorm2d(c, raw(name + '.g')[0], raw(name + '.b')[0],
                           raw(name + '.m')[0], raw(name + '.v')[0], eps,
                           dtype)

    s = cfg['image_size']
    pixels, _ = raw('image')
    x = _metatensor.tensor(pixels * batch, [batch, len(pixels)], False, dtype)
    ow, ok = cfg['stem'][0][0], cfg['stem'][1]
    conv1 = Conv2d(get('conv1.w'), None, 3, s, s, ok, 2, ok // 2)
    pool = MaxPool2d(ow, conv1.oh, conv1.ow, 3, 2, 1)
    h = pool.oh
    blocks = []
    for li, n, shape, k, stride, down in cfg['layers']:
        p = 'l%d.%d.' % (li, n)
        o, c = shape[0], shape[1]
        c1 = Conv2d(get(p + 'conv1.w'), None, c, h, h, k, stride, k // 2)
        c2 = Conv2d(get(p + 'conv2.w'), None, o, c1.oh, c1.ow, k, 1, k // 2)
        d = bnd = None
        if down:
            d = Conv2d(get(p + 'down.w'), None, c, h, h, 1, stride, 0)
            bnd = bn(p + 'bnd', o)
        blocks.append(ResNetBlock(c1, bn(p + 'bn1', o), c2,
                                  bn(p + 'bn2', o), d, bnd))
        h = c2.oh
    model = ResNet(conv1, bn('bn1', ow), pool, blocks, shape[0], h * h,
                   get('fc.w'), get('fc.b'))
    return model, x


def main():
    outdir = sys.argv[1]
    iters = int(sys.argv[2]) if len(sys.argv) > 2 else 20
    warmup = int(sys.argv[3]) if len(sys.argv) > 3 else 10
    batch = int(sys.argv[4]) if len(sys.argv) > 4 else 1
    dtype = os.environ.get('RTENSOR_DTYPE', 'float32')
    cfg, buf = load(outdir)
    model, x = build(cfg, buf, dtype, batch)
    for i in range(warmup):
        logits = model(x)
    logits.sum().item()
    t0 = time.time()
    for i in range(iters):
        logits = model(x)
    acc = logits.sum().item()
    steady_us = (time.time() - t0) / iters * 1e6
    flat = logits.tolist()[:cfg['classes']]
    order = sorted(range(len(flat)), key=lambda j: -flat[j])[:5]
    out = array.array('f', flat)
    if sys.byteorder != 'little':
        out.byteswap()
    out.tofile(open(os.path.join(outdir, 'logits_pypy.bin'), 'wb'))
    print('resnet pypy model=%s batch=%d classes=%d dtype=%s iters=%d '
          'steady_us=%.1f checksum=%.6f' %
          (cfg['source'], batch, cfg['classes'], dtype, iters, steady_us,
           acc))
    print('argmax %s' % ' '.join([str(a) for a in order]))


main()
