import array, json, os, sys, time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                '..', '..', 'lib_pypy'))

import _metatensor
from tensorpypy.functional import gelu_erf
from tensorpypy.models import GPT2MLP, MixerBlock, Mixer


def load(outdir):
    cfg = json.load(open(os.path.join(outdir, 'index.json')))
    buf = array.array('f')
    path = os.path.join(outdir, 'weights.bin')
    buf.fromfile(open(path, 'rb'), os.path.getsize(path) // 4)
    if sys.byteorder != 'little':
        buf.byteswap()
    return cfg, buf


def build(cfg, buf, dtype):
    index = cfg['index']

    def raw(name):
        off, shape = index[name]
        n = 1
        for d in shape:
            n *= d
        return list(buf[off:off + n]), shape

    def get(name):
        values, shape = raw(name)
        return _metatensor.tensor(values, shape, False, dtype)

    eps = cfg['eps']
    t = cfg['tokens']
    pixels, _ = raw('image')
    img = _metatensor.tensor(pixels, [1, len(pixels)], False, dtype)
    blocks = []
    for i in range(cfg['n_layer']):
        p = 'h.%d.' % i
        mlp = GPT2MLP(get(p + 'mlp.fc.w'), get(p + 'mlp.fc.b'),
                      get(p + 'mlp.proj.w'), get(p + 'mlp.proj.b'), gelu_erf)
        blocks.append(MixerBlock(
            get(p + 'ln_1.g'), get(p + 'ln_1.b'), get(p + 'tok.fc.w'),
            get(p + 'tok.fc.b'), get(p + 'tok.proj.w'), get(p + 'tok.proj.b'),
            get(p + 'ln_2.g'), get(p + 'ln_2.b'), mlp, eps))
    mean = _metatensor.tensor([1.0 / t] * t, [1, t], False, dtype)
    model = Mixer(cfg['image_size'], cfg['patch_size'], get('wp'), get('bp'),
                  blocks, get('ln_f.g'), get('ln_f.b'), mean, get('head.w'),
                  get('head.b'), eps)
    return model, img


def main():
    outdir = sys.argv[1]
    iters = int(sys.argv[2]) if len(sys.argv) > 2 else 20
    warmup = int(sys.argv[3]) if len(sys.argv) > 3 else 10
    dtype = os.environ.get('RTENSOR_DTYPE', 'float32')
    cfg, buf = load(outdir)
    model, img = build(cfg, buf, dtype)
    for i in range(warmup):
        logits = model(img)
    logits.sum().item()
    t0 = time.time()
    for i in range(iters):
        logits = model(img)
    acc = logits.sum().item()
    steady_us = (time.time() - t0) / iters * 1e6
    flat = logits.tolist()
    order = sorted(range(len(flat)), key=lambda j: -flat[j])[:5]
    out = array.array('f', flat)
    if sys.byteorder != 'little':
        out.byteswap()
    out.tofile(open(os.path.join(outdir, 'logits_pypy.bin'), 'wb'))
    print('mixer pypy layers=%d embd=%d tokens=%d classes=%d dtype=%s '
          'iters=%d steady_us=%.1f checksum=%.6f' %
          (cfg['n_layer'], cfg['n_embd'], cfg['tokens'], cfg['classes'],
           dtype, iters, steady_us, acc))
    print('argmax %s' % ' '.join([str(a) for a in order]))


main()
