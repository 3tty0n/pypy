import common
import rope

from tensorpypy.models import LlamaAttention, LlamaMLP, LlamaBlock, Llama


def rope3(cfg, w, dtype, t):
    """cos and sin as [t, 3d] tables for the fused q|k|v rows: the rotation
    for q and k, identity for v.  The first seq rows are the exported ones;
    decoding past them computes the rest with the exporter's formula."""
    dh = cfg['head_dim']
    if t == cfg['seq']:
        cvals, cshape = w.raw('rope.cos')
        svals, _ = w.raw('rope.sin')
        d = cshape[1]
    else:
        d = cfg['n_embd']
        cvals, svals = rope.rope_tables(t, dh, cfg['n_head'],
                                        cfg['rope_theta'])
    c3, s3 = [], []
    for r in range(t):
        row = list(cvals[r * d:(r + 1) * d])
        srow = svals[r * d:(r + 1) * d]
        srow = [-v if (i % dh) < dh // 2 else v
                for i, v in enumerate(srow)]
        c3.extend(row + row + [1.0] * d)
        s3.extend(srow + srow + [0.0] * d)
    return (common.tensor(c3, [t, 3 * d], False, dtype),
            common.tensor(s3, [t, 3 * d], False, dtype))


def build(cfg, buf, dtype, t=None):
    w = common.Weights(cfg, buf, dtype)
    if t is None:
        t = cfg['seq']
    h = cfg['n_head']
    eps = cfg['eps']
    idx = common.tensor([float(tok) for tok in cfg['tokens'][:t]],
                                    [t], False, dtype)
    mask = common.causal_mask(h, t, dtype)
    dh = cfg['head_dim']
    cos, sin = rope3(cfg, w, dtype, t)
    blocks = []
    for i in range(cfg['n_layer']):
        pre = 'h.%d.' % i
        bqkv = None
        if cfg.get('qkv_bias'):
            bqkv = w.cat([pre + 'attn.q.b', pre + 'attn.k.b',
                          pre + 'attn.v.b'])
        attn = LlamaAttention(
            w.cat([pre + 'attn.q.w', pre + 'attn.k.w', pre + 'attn.v.w']),
            w.get(pre + 'attn.proj.w'), h, mask, cos, sin, dh, bqkv)
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
