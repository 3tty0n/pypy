import common

from tensorpypy.functional import gelu_erf
from tensorpypy.models import CausalSelfAttention, GPT2MLP, BertBlock, Bert


def build(cfg, buf, dtype):
    w = common.Weights(cfg, buf, dtype)
    h = cfg['n_head']
    eps = cfg['eps']
    idx = common.tensor([float(tok) for tok in cfg['tokens']],
                                    [cfg['seq']], False, dtype)
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
        blocks.append(BertBlock(attn, w.get(p + 'ln_1.g'),
                                w.get(p + 'ln_1.b'), mlp, w.get(p + 'ln_2.g'),
                                w.get(p + 'ln_2.b'), eps))
    model = Bert(w.get('wte'), w.get('emb'), w.get('emb.g'), w.get('emb.b'),
                 blocks, w.get('mlm.dense.w'), w.get('mlm.dense.b'),
                 w.get('mlm.ln.g'), w.get('mlm.ln.b'), w.get('mlm.bias'), eps)
    return model, idx


def main():
    outdir, iters, warmup, dtype = common.argv()
    cfg, buf = common.load(outdir)
    model, idx = build(cfg, buf, dtype)
    logits, acc, steady_us = common.timed(model, (idx,), iters, warmup)
    flat = logits.tolist()
    common.dump(outdir, flat)
    common.report(
        'bert pypy layers=%d embd=%d heads=%d seq=%d vocab=%d dtype=%s '
        'iters=%d steady_us=%.1f checksum=%.6f' %
        (cfg['n_layer'], cfg['n_embd'], cfg['n_head'], cfg['seq'],
         cfg['vocab'], dtype, iters, steady_us, acc),
        common.token_argmax(flat, cfg['seq'], cfg['vocab']))


if __name__ == '__main__':
    main()
