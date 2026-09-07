import common

from tensorpypy.models import LlamaAttention, LlamaMLP, LlamaBlock, Llama


def build(cfg, buf, dtype):
    w = common.Weights(cfg, buf, dtype)
    t = cfg['seq']
    h = cfg['n_head']
    eps = cfg['eps']
    idx = common.tensor([float(tok) for tok in cfg['tokens']],
                                    [t], False, dtype)
    mask = common.causal_mask(h, t, dtype)
    dh = cfg['head_dim']
    cvals, cshape = w.raw('rope.cos')
    svals, _ = w.raw('rope.sin')
    d = cshape[1]
    c3, s3 = [], []
    for r in range(cshape[0]):
        row = cvals[r * d:(r + 1) * d]
        srow = svals[r * d:(r + 1) * d]
        srow = [-v if (i % dh) < dh // 2 else v
                for i, v in enumerate(srow)]
        c3.extend(row + row + [1.0] * d)
        s3.extend(srow + srow + [0.0] * d)
    cos = common.tensor(c3, [cshape[0], 3 * d], False, dtype)
    sin = common.tensor(s3, [cshape[0], 3 * d], False, dtype)
    blocks = []
    for i in range(cfg['n_layer']):
        pre = 'h.%d.' % i
        attn = LlamaAttention(
            w.cat([pre + 'attn.q.w', pre + 'attn.k.w', pre + 'attn.v.w']),
            w.get(pre + 'attn.proj.w'), h, mask, cos, sin, dh)
        mlp = LlamaMLP(w.get(pre + 'mlp.gate.w'), w.get(pre + 'mlp.up.w'),
                       w.get(pre + 'mlp.down.w'))
        blocks.append(LlamaBlock(attn, w.get(pre + 'norm1.g'),
                                 w.get(pre + 'norm2.g'), mlp, eps))
    head = None if cfg['tied'] else w.get('lm_head.w')
    return Llama(w.get('wte'), blocks, w.get('norm_f.g'), head, eps), idx


def main():
    outdir, iters, warmup, dtype = common.argv()
    cfg, buf = common.load(outdir)
    model, idx = build(cfg, buf, dtype)
    logits, acc, steady_us = common.timed(model, (idx,), iters, warmup)
    flat = logits.tolist()
    common.dump(outdir, flat)
    common.report(
        'llama pypy layers=%d embd=%d heads=%d kv=%d seq=%d vocab=%d '
        'dtype=%s iters=%d steady_us=%.1f checksum=%.6f' %
        (cfg['n_layer'], cfg['n_embd'], cfg['n_head'], cfg['n_kv_head'],
         cfg['seq'], cfg['vocab'], dtype, iters, steady_us, acc),
        common.token_argmax(flat, cfg['seq'], cfg['vocab']))


if __name__ == '__main__':
    main()
