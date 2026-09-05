import argparse, json, math, os, struct


def rope_tables(seq, dh, heads, theta):
    cos, sin = [], []
    half = dh // 2
    inv = [1.0 / (theta ** (2.0 * i / dh)) for i in range(half)]
    for pos in range(seq):
        row = [math.cos(pos * f) for f in inv]
        row = row + row
        srow = [math.sin(pos * f) for f in inv]
        srow = srow + srow
        cos.extend(row * heads)
        sin.extend(srow * heads)
    return cos, sin


def rope_perm(dh, heads):
    d = dh * heads
    p = [0.0] * (d * d)
    half = dh // 2
    for h in range(heads):
        o = h * dh
        for j in range(dh):
            if j < half:
                p[(o + j + half) * d + o + j] = -1.0
            else:
                p[(o + j - half) * d + o + j] = 1.0
    return p


def hf_weights(model_id, cfg):
    import torch
    from transformers import AutoModelForCausalLM
    m = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.float32)
    c = m.config
    sd = m.state_dict()
    d = c.hidden_size
    heads = c.num_attention_heads
    kv = c.num_key_value_heads
    dh = getattr(c, 'head_dim', None) or d // heads
    rp = getattr(c, 'rope_parameters', None)
    theta = rp['rope_theta'] if rp else c.rope_theta
    cfg.update({'vocab': c.vocab_size, 'n_embd': d, 'n_head': heads,
                'n_kv_head': kv, 'head_dim': dh, 'n_layer':
                c.num_hidden_layers, 'eps': c.rms_norm_eps,
                'ff': c.intermediate_size, 'rope_theta': theta,
                'tied': bool(c.tie_word_embeddings)})
    w = {}

    def put(key, t):
        w[key] = (list(t.shape), t.contiguous().view(-1).float().tolist())

    def putT(key, t):
        put(key, t.t())

    def rep(t):
        n = heads // kv
        return t.view(kv, dh, -1).repeat_interleave(n, 0).reshape(heads * dh,
                                                                  -1)

    put('wte', sd['model.embed_tokens.weight'])
    if not c.tie_word_embeddings:
        put('lm_head.w', sd['lm_head.weight'])
    put('norm_f.g', sd['model.norm.weight'])
    for i in range(c.num_hidden_layers):
        pre = 'model.layers.%d.' % i
        putT('h.%d.attn.q.w' % i, sd[pre + 'self_attn.q_proj.weight'])
        putT('h.%d.attn.k.w' % i, rep(sd[pre + 'self_attn.k_proj.weight']))
        putT('h.%d.attn.v.w' % i, rep(sd[pre + 'self_attn.v_proj.weight']))
        putT('h.%d.attn.proj.w' % i, sd[pre + 'self_attn.o_proj.weight'])
        putT('h.%d.mlp.gate.w' % i, sd[pre + 'mlp.gate_proj.weight'])
        putT('h.%d.mlp.up.w' % i, sd[pre + 'mlp.up_proj.weight'])
        putT('h.%d.mlp.down.w' % i, sd[pre + 'mlp.down_proj.weight'])
        put('h.%d.norm1.g' % i, sd[pre + 'input_layernorm.weight'])
        put('h.%d.norm2.g' % i, sd[pre + 'post_attention_layernorm.weight'])
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_id)
    ids = tok('The quick brown fox jumps over the lazy dog. ' * 40)
    cfg['tokens'] = ids['input_ids'][:cfg['seq_hint']]
    return w


def write(outdir, cfg, weights, seq):
    if not os.path.isdir(outdir):
        os.makedirs(outdir)
    heads, dh = cfg['n_head'], cfg['head_dim']
    cos, sin = rope_tables(seq, dh, heads, cfg['rope_theta'])
    d = heads * dh
    weights['rope.cos'] = ([seq, d], cos)
    weights['rope.sin'] = ([seq, d], sin)
    weights['rope.p'] = ([d, d], rope_perm(dh, heads))
    index = {}
    off = 0
    with open(os.path.join(outdir, 'weights.bin'), 'wb') as f:
        for name in sorted(weights):
            shape, data = weights[name]
            f.write(struct.pack('<%df' % len(data), *data))
            index[name] = [off, shape]
            off += len(data)
    toks = cfg.get('tokens') or []
    while len(toks) < seq:
        toks.append((len(toks) * 137 + 11) % cfg['vocab'])
    cfg['tokens'] = toks[:seq]
    cfg['seq'] = seq
    cfg.pop('seq_hint', None)
    cfg['index'] = index
    with open(os.path.join(outdir, 'index.json'), 'w') as f:
        json.dump(cfg, f)
    print('wrote %s: %d tensors, %d floats' % (outdir, len(index), off))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('outdir')
    ap.add_argument('--model', default='HuggingFaceTB/SmolLM2-135M')
    ap.add_argument('--seq', type=int, default=64)
    a = ap.parse_args()
    cfg = {'source': a.model, 'seq_hint': a.seq}
    write(a.outdir, cfg, hf_weights(a.model, cfg), a.seq)


main()
