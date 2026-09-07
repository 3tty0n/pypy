import common

from tensorpypy.models import CausalSelfAttention, GPT2MLP, GPT2Block, GPT2


def build_model(cfg, buf, dtype, mask=None):
    w = common.Weights(cfg, buf, dtype)
    h = cfg['n_head']
    eps = cfg['eps']
    blocks = []
    for i in range(cfg['n_layer']):
        p = 'h.%d.' % i
        attn = CausalSelfAttention(
            w.cat([p + 'attn.q.w', p + 'attn.k.w', p + 'attn.v.w']),
            w.cat([p + 'attn.q.b', p + 'attn.k.b', p + 'attn.v.b']),
            w.get(p + 'attn.proj.w'), w.get(p + 'attn.proj.b'), h, mask)
        mlp = GPT2MLP(w.get(p + 'mlp.fc.w'), w.get(p + 'mlp.fc.b'),
                      w.get(p + 'mlp.proj.w'), w.get(p + 'mlp.proj.b'))
        blocks.append(GPT2Block(
            attn, w.get(p + 'ln_1.g'), w.get(p + 'ln_1.b'),
            w.get(p + 'ln_2.g'), w.get(p + 'ln_2.b'), mlp, eps))
    return GPT2(w.get('wte'), blocks, w.get('ln_f.g'), w.get('ln_f.b'), eps)


def build(cfg, buf, dtype):
    t = cfg['seq']
    d = cfg['n_embd']
    idx = common.tensor([float(tok) for tok in cfg['tokens']], [t], False,
                        dtype)
    wpe_off = cfg['index']['wpe'][0]
    pos = common.tensor(list(buf[wpe_off:wpe_off + t * d]), [t, d], False,
                        dtype)
    mask = common.causal_mask(cfg['n_head'], t, dtype)
    return build_model(cfg, buf, dtype, mask), idx, pos


def main():
    outdir, iters, warmup, dtype = common.argv()
    cfg, buf = common.load(outdir)
    model, idx, pos = build(cfg, buf, dtype)
    logits, acc, steady_us = common.timed(model, (idx, pos), iters, warmup)
    flat = logits.tolist()
    common.dump(outdir, flat)
    common.report(
        'gpt2 pypy layers=%d embd=%d heads=%d seq=%d vocab=%d dtype=%s '
        'iters=%d steady_us=%.1f checksum=%.6f' %
        (cfg['n_layer'], cfg['n_embd'], cfg['n_head'], cfg['seq'],
         cfg['vocab'], dtype, iters, steady_us, acc),
        common.token_argmax(flat, cfg['seq'], cfg['vocab']))


if __name__ == '__main__':
    main()
