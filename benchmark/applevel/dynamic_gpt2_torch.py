import json, os, sys, time

import torch
import torch._dynamo

import gpt2_torch
import torch_common

LENGTHS = [32, 48, 64, 96, 128]


class Model(gpt2_torch.Model):
    def __init__(self, cfg, flat, dtype, dev):
        super().__init__(cfg, flat, dtype, dev)
        self.wpe_flat = flat
        self.wpe_off = cfg['index']['wpe'][0]
        self.d = cfg['n_embd']
        self.dev, self.dtype = dev, dtype

    def pos_and_mask(self, t):
        d = self.d
        pos = self.wpe_flat[self.wpe_off:self.wpe_off + t * d].view(
            t, d).to(self.dev, self.dtype)
        m = torch.zeros(t, t, device=self.dev, dtype=self.dtype)
        mask = m.masked_fill(
            torch.triu(torch.ones(t, t, device=self.dev), 1).bool(), -1e9)
        return pos, mask

    def forward(self, idx, pos, mask):
        return gpt2_torch.blocks_forward(self, self.wte[idx] + pos, mask)


def main():
    mode = sys.argv[1]
    outdir = sys.argv[2]
    iters = int(sys.argv[3]) if len(sys.argv) > 3 else 200
    dtype = getattr(torch, os.environ.get('RTENSOR_DTYPE', 'float32'))
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    cfg = json.load(open(os.path.join(outdir, 'index.json')))
    tokens = cfg['tokens']
    flat = torch_common.flat_weights(outdir)
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


if __name__ == '__main__':
    main()
