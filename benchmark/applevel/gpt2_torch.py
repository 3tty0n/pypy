import array, json, math, os, sys, time

import torch


def load(outdir):
    cfg = json.load(open(os.path.join(outdir, 'index.json')))
    buf = array.array('f')
    path = os.path.join(outdir, 'weights.bin')
    buf.fromfile(open(path, 'rb'), os.path.getsize(path) // 4)
    if sys.byteorder != 'little':
        buf.byteswap()
    return cfg, torch.tensor(buf, dtype=torch.float32)


class Model(torch.nn.Module):
    def __init__(self, cfg, flat, dtype, dev):
        super().__init__()
        index = cfg['index']

        def get(name):
            off, shape = index[name]
            n = 1
            for d in shape:
                n *= d
            return flat[off:off + n].view(*shape).to(dev, dtype)

        self.cfg = cfg
        self.h = cfg['n_head']
        self.eps = cfg['eps']
        self.wte = get('wte')
        self.ln_f = (get('ln_f.g'), get('ln_f.b'))
        self.layers = []
        for i in range(cfg['n_layer']):
            p = 'h.%d.' % i
            self.layers.append(tuple(get(p + k) for k in [
                'ln_1.g', 'ln_1.b', 'attn.q.w', 'attn.q.b', 'attn.k.w',
                'attn.k.b', 'attn.v.w', 'attn.v.b', 'attn.proj.w',
                'attn.proj.b', 'ln_2.g', 'ln_2.b', 'mlp.fc.w', 'mlp.fc.b',
                'mlp.proj.w', 'mlp.proj.b']))
        t = cfg['seq']
        d = cfg['n_embd']
        off = index['wpe'][0]
        self.pos = flat[off:off + t * d].view(t, d).to(dev, dtype)
        m = torch.zeros(t, t, device=dev, dtype=dtype)
        self.mask = m.masked_fill(
            torch.triu(torch.ones(t, t, device=dev), 1).bool(), -1e9)

    def ln(self, x, g, b):
        return torch.nn.functional.layer_norm(x, (x.shape[-1],), g, b,
                                              self.eps)

    def forward(self, idx):
        x = self.wte[idx] + self.pos
        t, d = x.shape
        h = self.h
        dh = d // h
        for (g1, b1, wq, bq, wk, bk, wv, bv, wo, bo, g2, b2, wf, bf, wp,
             bp) in self.layers:
            n = self.ln(x, g1, b1)
            q = (n @ wq + bq).view(t, h, dh).transpose(0, 1)
            k = (n @ wk + bk).view(t, h, dh).transpose(0, 1)
            v = (n @ wv + bv).view(t, h, dh).transpose(0, 1)
            s = q @ k.transpose(-1, -2) / math.sqrt(dh) + self.mask
            c = (s.softmax(-1) @ v).transpose(0, 1).reshape(t, d)
            x = x + (c @ wo + bo)
            n = self.ln(x, g2, b2)
            u = n @ wf + bf
            u = 0.5 * u * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) *
                                            (u + 0.044715 * u * u * u)))
            x = x + (u @ wp + bp)
        x = self.ln(x, *self.ln_f)
        return x @ self.wte.t()


def hf_model(cfg, dtype, dev):
    from transformers import GPT2LMHeadModel
    m = GPT2LMHeadModel.from_pretrained(cfg['source'])
    return m.to(dev, dtype).eval()


def main():
    mode = sys.argv[1]
    outdir = sys.argv[2]
    iters = int(sys.argv[3]) if len(sys.argv) > 3 else 20
    warmup = int(sys.argv[4]) if len(sys.argv) > 4 else 10
    dtype = getattr(torch, os.environ.get('RTENSOR_DTYPE', 'float32'))
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    cfg, flat = load(outdir)
    idx = torch.tensor(cfg['tokens'], device=dev, dtype=torch.long)
    if cfg['source'] == 'random':
        model = Model(cfg, flat, dtype, dev)
        fwd = model.forward
    else:
        hf = hf_model(cfg, dtype, dev)
        fwd = lambda i: hf(i.unsqueeze(0)).logits[0]
    if mode == 'compile':
        fwd = torch.compile(fwd)
    with torch.no_grad():
        for i in range(warmup):
            logits = fwd(idx)
        if dev == 'cuda':
            torch.cuda.synchronize()
        t0 = time.time()
        for i in range(iters):
            logits = fwd(idx)
        acc = logits.double().sum().item()
        steady_us = (time.time() - t0) / iters * 1e6
    argmax = logits.argmax(-1).tolist()
    ref = os.path.join(outdir, 'logits_pypy.bin')
    diff = ''
    if os.path.exists(ref):
        other = torch.frombuffer(open(ref, 'rb').read(),
                                 dtype=torch.float32).to(dev)
        other = other.view_as(logits.float())
        diff = ' maxabsdiff=%.6g' % (logits.float() - other).abs().max().item()
    print('gpt2 torch-%s layers=%d embd=%d heads=%d seq=%d vocab=%d dtype=%s '
          'iters=%d steady_us=%.1f checksum=%.6f' %
          (mode, cfg['n_layer'], cfg['n_embd'], cfg['n_head'], cfg['seq'],
           cfg['vocab'], os.environ.get('RTENSOR_DTYPE', 'float32'), iters,
           steady_us, acc))
    print('argmax %s%s' % (' '.join(str(a) for a in argmax), diff))


main()
