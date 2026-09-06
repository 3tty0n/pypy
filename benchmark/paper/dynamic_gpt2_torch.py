import array, json, math, os, sys, time

import torch
import torch._dynamo

LENGTHS = [32, 48, 64, 96, 128]


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
        d = cfg['n_embd']
        wpe_off = index['wpe'][0]
        self.wpe_flat = flat
        self.wpe_off = wpe_off
        self.d = d
        self.dev, self.dtype = dev, dtype

    def pos_and_mask(self, t):
        d = self.d
        pos = self.wpe_flat[self.wpe_off:self.wpe_off + t * d].view(
            t, d).to(self.dev, self.dtype)
        m = torch.zeros(t, t, device=self.dev, dtype=self.dtype)
        mask = m.masked_fill(
            torch.triu(torch.ones(t, t, device=self.dev), 1).bool(), -1e9)
        return pos, mask

    def ln(self, x, g, b):
        return torch.nn.functional.layer_norm(x, (x.shape[-1],), g, b,
                                              self.eps)

    def forward(self, idx, pos, mask):
        x = self.wte[idx] + pos
        t, d = x.shape
        h = self.h
        dh = d // h
        for (g1, b1, wq, bq, wk, bk, wv, bv, wo, bo, g2, b2, wf, bf, wp,
             bp) in self.layers:
            n = self.ln(x, g1, b1)
            q = (n @ wq + bq).view(t, h, dh).transpose(0, 1)
            k = (n @ wk + bk).view(t, h, dh).transpose(0, 1)
            v = (n @ wv + bv).view(t, h, dh).transpose(0, 1)
            s = q @ k.transpose(-1, -2) / math.sqrt(dh) + mask
            c = (s.softmax(-1) @ v).transpose(0, 1).reshape(t, d)
            x = x + (c @ wo + bo)
            n = self.ln(x, g2, b2)
            u = n @ wf + bf
            u = 0.5 * u * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) *
                                            (u + 0.044715 * u * u * u)))
            x = x + (u @ wp + bp)
        x = self.ln(x, *self.ln_f)
        return x @ self.wte.t()


def main():
    mode = sys.argv[1]
    outdir = sys.argv[2]
    iters = int(sys.argv[3]) if len(sys.argv) > 3 else 200
    dtype = getattr(torch, os.environ.get('RTENSOR_DTYPE', 'float32'))
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    cfg, flat = load(outdir)
    tokens = cfg['tokens']
    model = Model(cfg, flat, dtype, dev).to(dev, dtype).eval()

    dynamic = mode == 'compile-dynamic'
    fwd = model.forward
    if mode in ('compile', 'compile-dynamic'):
        torch._dynamo.reset()
        fwd = torch.compile(model.forward, dynamic=dynamic)

    def step(t):
        ids = [tokens[i % len(tokens)] for i in range(t)]
        idx = torch.tensor(ids, device=dev, dtype=torch.long)
        pos, mask = model.pos_and_mask(t)
        with torch.no_grad():
            out = fwd(idx, pos, mask)
        return out

    with torch.no_grad():
        for _ in range(5):
            step(LENGTHS[0])
        torch.cuda.synchronize() if dev == 'cuda' else None

        before = 0
        if mode in ('compile', 'compile-dynamic'):
            before = torch._dynamo.utils.counters['stats']['unique_graphs']

        print('length\tus')
        for i in range(iters):
            t = LENGTHS[i % len(LENGTHS)]
            t0 = time.time()
            step(t)
            if dev == 'cuda':
                torch.cuda.synchronize()
            us = (time.time() - t0) * 1e6
            print('%d\t%.1f' % (t, us))

        if mode in ('compile', 'compile-dynamic'):
            after = torch._dynamo.utils.counters['stats']['unique_graphs']
            print('recompiles\t%d' % (after - before))
        else:
            print('recompiles\t0')


main()
