import common

from tensorpypy.functional import gelu_erf
from tensorpypy.models import GPT2MLP, MixerBlock, Mixer


def build(cfg, buf, dtype):
    w = common.Weights(cfg, buf, dtype)
    eps = cfg['eps']
    t = cfg['tokens']
    pixels, _ = w.raw('image')
    img = common.tensor(pixels, [1, len(pixels)], False, dtype)
    blocks = []
    for i in range(cfg['n_layer']):
        p = 'h.%d.' % i
        mlp = GPT2MLP(w.get(p + 'mlp.fc.w'), w.get(p + 'mlp.fc.b'),
                      w.get(p + 'mlp.proj.w'), w.get(p + 'mlp.proj.b'),
                      gelu_erf)
        blocks.append(MixerBlock(
            w.get(p + 'ln_1.g'), w.get(p + 'ln_1.b'), w.get(p + 'tok.fc.w'),
            w.get(p + 'tok.fc.b'), w.get(p + 'tok.proj.w'),
            w.get(p + 'tok.proj.b'), w.get(p + 'ln_2.g'),
            w.get(p + 'ln_2.b'), mlp, eps))
    mean = common.tensor([1.0 / t] * t, [1, t], False, dtype)
    model = Mixer(cfg['image_size'], cfg['patch_size'], w.get('wp'),
                  w.get('bp'), blocks, w.get('ln_f.g'), w.get('ln_f.b'), mean,
                  w.get('head.w'), w.get('head.b'), eps)
    return model, img


def main():
    outdir, iters, warmup, dtype = common.argv()
    cfg, buf = common.load(outdir)
    model, img = build(cfg, buf, dtype)
    logits, acc, steady_us = common.timed(model, (img,), iters, warmup)
    flat = logits.tolist()
    common.dump(outdir, flat)
    common.report(
        'mixer pypy layers=%d embd=%d tokens=%d classes=%d dtype=%s '
        'iters=%d steady_us=%.1f checksum=%.6f' %
        (cfg['n_layer'], cfg['n_embd'], cfg['tokens'], cfg['classes'],
         dtype, iters, steady_us, acc),
        common.top5(flat))


if __name__ == '__main__':
    main()
