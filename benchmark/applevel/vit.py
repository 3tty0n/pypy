import common

from tensorpypy.functional import gelu_erf
from tensorpypy.models import CausalSelfAttention, GPT2MLP, GPT2Block, ViT


def patch_index(cfg):
    k = cfg['patch_size']
    s = cfg['image_size']
    grid = s // k
    zero = 3 * s * grid
    idx = [float(zero)] * (3 * k)
    for ph in range(grid):
        for pw in range(grid):
            for c in range(3):
                for kh in range(k):
                    idx.append(float(((c * s) + ph * k + kh) * grid + pw))
    return idx


def build(cfg, buf, dtype):
    w = common.Weights(cfg, buf, dtype)
    h = cfg['n_head']
    eps = cfg['eps']
    k = cfg['patch_size']
    pixels, _ = w.raw('image')
    pixels.extend([0.0] * k)
    img = common.tensor(pixels, [len(pixels) // k, k], False, dtype)
    idx = patch_index(cfg)
    idx = common.tensor(idx, [len(idx)], False, dtype)
    blocks = []
    for i in range(cfg['n_layer']):
        p = 'h.%d.' % i
        attn = CausalSelfAttention(
            w.cat([p + 'attn.q.w', p + 'attn.k.w', p + 'attn.v.w']),
            w.cat([p + 'attn.q.b', p + 'attn.k.b', p + 'attn.v.b']),
            w.get(p + 'attn.proj.w'), w.get(p + 'attn.proj.b'), h, None)
        mlp = GPT2MLP(w.get(p + 'mlp.fc.w'), w.get(p + 'mlp.fc.b'),
                      w.get(p + 'mlp.proj.w'), w.get(p + 'mlp.proj.b'),
                      gelu_erf)
        blocks.append(GPT2Block(attn, w.get(p + 'ln_1.g'),
                                w.get(p + 'ln_1.b'), w.get(p + 'ln_2.g'),
                                w.get(p + 'ln_2.b'), mlp, eps))
    model = ViT(idx, cfg['tokens'], cfg['patch'], w.get('wp'), w.get('emb'),
                blocks, w.get('ln_f.g'), w.get('ln_f.b'), w.get('head.w'),
                w.get('head.b'), common.tensor([0.0], [1], False, dtype), eps)
    return model, img


def main():
    outdir, iters, warmup, dtype = common.argv()
    cfg, buf = common.load(outdir)
    model, img = build(cfg, buf, dtype)
    logits, acc, steady_us = common.timed(model, (img,), iters, warmup)
    flat = logits.tolist()
    common.dump(outdir, flat)
    common.report(
        'vit pypy layers=%d embd=%d heads=%d tokens=%d classes=%d '
        'dtype=%s iters=%d steady_us=%.1f checksum=%.6f' %
        (cfg['n_layer'], cfg['n_embd'], cfg['n_head'], cfg['tokens'],
         cfg['classes'], dtype, iters, steady_us, acc),
        common.top5(flat))


if __name__ == '__main__':
    main()
