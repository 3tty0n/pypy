import argparse, json, math, os, struct, sys


def _rand(n, seed, scale):
    out = []
    s = seed
    for i in range(n):
        s = (s * 1103515245 + 12345) & 0x7fffffff
        out.append(((s / float(0x7fffffff)) - 0.5) * 2.0 * scale)
    return out


def random_weights(cfg):
    v, p, d, l = cfg['vocab'], cfg['npos'], cfg['n_embd'], cfg['n_layer']
    w = {}
    w['wte'] = ([v, d], _rand(v * d, 1, 0.02))
    w['wpe'] = ([p, d], _rand(p * d, 2, 0.02))
    seed = 3
    for i in range(l):
        for name, shape in [('attn.q', [d, d]), ('attn.k', [d, d]),
                            ('attn.v', [d, d]), ('attn.proj', [d, d]),
                            ('mlp.fc', [d, 4 * d]), ('mlp.proj', [4 * d, d])]:
            n = shape[0] * shape[1]
            w['h.%d.%s.w' % (i, name)] = (shape, _rand(n, seed, 0.05))
            seed += 1
            w['h.%d.%s.b' % (i, name)] = ([shape[1]], [0.0] * shape[1])
        for name in ['ln_1', 'ln_2']:
            w['h.%d.%s.g' % (i, name)] = ([d], [1.0] * d)
            w['h.%d.%s.b' % (i, name)] = ([d], [0.0] * d)
    w['ln_f.g'] = ([d], [1.0] * d)
    w['ln_f.b'] = ([d], [0.0] * d)
    return w


def hf_weights(model_id, cfg):
    from transformers import GPT2LMHeadModel
    m = GPT2LMHeadModel.from_pretrained(model_id)
    sd = m.state_dict()
    c = m.config
    cfg.update({'vocab': c.vocab_size, 'npos': c.n_positions,
                'n_embd': c.n_embd, 'n_head': c.n_head,
                'n_layer': c.n_layer, 'eps': c.layer_norm_epsilon})
    d = c.n_embd
    w = {}

    def put(key, t):
        w[key] = (list(t.shape), t.contiguous().view(-1).float().tolist())

    put('wte', sd['transformer.wte.weight'])
    put('wpe', sd['transformer.wpe.weight'])
    for i in range(c.n_layer):
        pre = 'transformer.h.%d.' % i
        aw = sd[pre + 'attn.c_attn.weight']
        ab = sd[pre + 'attn.c_attn.bias']
        for j, name in enumerate(['q', 'k', 'v']):
            put('h.%d.attn.%s.w' % (i, name), aw[:, j * d:(j + 1) * d])
            put('h.%d.attn.%s.b' % (i, name), ab[j * d:(j + 1) * d])
        put('h.%d.attn.proj.w' % i, sd[pre + 'attn.c_proj.weight'])
        put('h.%d.attn.proj.b' % i, sd[pre + 'attn.c_proj.bias'])
        put('h.%d.mlp.fc.w' % i, sd[pre + 'mlp.c_fc.weight'])
        put('h.%d.mlp.fc.b' % i, sd[pre + 'mlp.c_fc.bias'])
        put('h.%d.mlp.proj.w' % i, sd[pre + 'mlp.c_proj.weight'])
        put('h.%d.mlp.proj.b' % i, sd[pre + 'mlp.c_proj.bias'])
        put('h.%d.ln_1.g' % i, sd[pre + 'ln_1.weight'])
        put('h.%d.ln_1.b' % i, sd[pre + 'ln_1.bias'])
        put('h.%d.ln_2.g' % i, sd[pre + 'ln_2.weight'])
        put('h.%d.ln_2.b' % i, sd[pre + 'ln_2.bias'])
    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(model_id)
        ids = tok('The quick brown fox jumps over the lazy dog. ' * 20)
        cfg['tokens'] = ids['input_ids'][:cfg['seq_hint']]
    except Exception as exc:
        print('tokenizer unavailable (%s), using synthetic ids' % exc)
    put('ln_f.g', sd['transformer.ln_f.weight'])
    put('ln_f.b', sd['transformer.ln_f.bias'])
    return w


def write(outdir, cfg, weights, seq):
    if not os.path.isdir(outdir):
        os.makedirs(outdir)
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
    ap.add_argument('--random', action='store_true')
    ap.add_argument('--model', default='sshleifer/tiny-gpt2')
    ap.add_argument('--seq', type=int, default=64)
    ap.add_argument('--vocab', type=int, default=1024)
    ap.add_argument('--n-embd', type=int, default=256)
    ap.add_argument('--n-head', type=int, default=8)
    ap.add_argument('--n-layer', type=int, default=4)
    ap.add_argument('--npos', type=int, default=1024)
    a = ap.parse_args()
    cfg = {'vocab': a.vocab, 'npos': a.npos, 'n_embd': a.n_embd,
           'n_head': a.n_head, 'n_layer': a.n_layer, 'eps': 1e-5,
           'source': 'random'}
    cfg['seq_hint'] = a.seq
    if a.random:
        w = random_weights(cfg)
    else:
        cfg['source'] = a.model
        w = hf_weights(a.model, cfg)
    write(a.outdir, cfg, w, a.seq)


main()
