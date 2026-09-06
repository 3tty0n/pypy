import argparse, json, os, struct


def hf_weights(model_id, cfg, seq):
    from transformers import BertForMaskedLM, AutoTokenizer
    m = BertForMaskedLM.from_pretrained(model_id)
    sd = m.state_dict()
    c = m.config
    cfg.update({'vocab': c.vocab_size, 'n_embd': c.hidden_size,
                'n_head': c.num_attention_heads,
                'n_layer': c.num_hidden_layers,
                'n_inner': c.intermediate_size,
                'eps': c.layer_norm_eps})
    w = {}

    def put(key, t, transpose=False):
        if transpose:
            t = t.t()
        w[key] = (list(t.shape), t.contiguous().view(-1).float().tolist())

    try:
        tok = AutoTokenizer.from_pretrained(model_id)
    except Exception:
        tok = AutoTokenizer.from_pretrained('bert-base-uncased')
    ids = tok('The quick brown fox jumps over the lazy dog. ' * 20)
    ids = ids['input_ids'][:seq]
    while len(ids) < seq:
        ids.append(tok.pad_token_id)
    cfg['tokens'] = ids
    pre = 'bert.embeddings.'
    put('wte', sd[pre + 'word_embeddings.weight'])
    emb = (sd[pre + 'position_embeddings.weight'][:seq] +
           sd[pre + 'token_type_embeddings.weight'][0])
    put('emb', emb)
    put('emb.g', sd[pre + 'LayerNorm.weight'])
    put('emb.b', sd[pre + 'LayerNorm.bias'])
    for i in range(c.num_hidden_layers):
        p = 'bert.encoder.layer.%d.' % i
        for src, dst in [('attention.self.query', 'attn.q'),
                         ('attention.self.key', 'attn.k'),
                         ('attention.self.value', 'attn.v'),
                         ('attention.output.dense', 'attn.proj'),
                         ('intermediate.dense', 'mlp.fc'),
                         ('output.dense', 'mlp.proj')]:
            put('h.%d.%s.w' % (i, dst), sd[p + src + '.weight'], True)
            put('h.%d.%s.b' % (i, dst), sd[p + src + '.bias'])
        for src, dst in [('attention.output.LayerNorm', 'ln_1'),
                         ('output.LayerNorm', 'ln_2')]:
            put('h.%d.%s.g' % (i, dst), sd[p + src + '.weight'])
            put('h.%d.%s.b' % (i, dst), sd[p + src + '.bias'])
    put('mlm.dense.w', sd['cls.predictions.transform.dense.weight'], True)
    put('mlm.dense.b', sd['cls.predictions.transform.dense.bias'])
    put('mlm.ln.g', sd['cls.predictions.transform.LayerNorm.weight'])
    put('mlm.ln.b', sd['cls.predictions.transform.LayerNorm.bias'])
    put('mlm.bias', sd['cls.predictions.bias'])
    return w


def write(outdir, cfg, weights):
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
    cfg['index'] = index
    with open(os.path.join(outdir, 'index.json'), 'w') as f:
        json.dump(cfg, f)
    print('wrote %s: %d tensors, %d floats' % (outdir, len(index), off))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('outdir')
    ap.add_argument('--model', default='prajjwal1/bert-tiny')
    ap.add_argument('--seq', type=int, default=64)
    a = ap.parse_args()
    cfg = {'source': a.model, 'seq': a.seq}
    write(a.outdir, cfg, hf_weights(a.model, cfg, a.seq))


main()
