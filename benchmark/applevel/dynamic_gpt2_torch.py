"""Dynamic sequence-length sweep, PyTorch side.

What the timed step contains - identical to dynamic_gpt2.py:

    build_us  token ids for this length, the position slice, the causal mask
    step_us   fwd(idx, pos, mask), then out.sum().item(), i.e. a reduction all
              the way back to a host scalar (which also synchronizes)
    total_us  build_us + step_us

Both boundaries are reported for both systems.  They are split because the
build is host-side input marshalling, not the runtime under test; see the
docstring of dynamic_gpt2.py for the numbers that motivate the split.

Per length window the driver reports the delta of torch._dynamo's unique-graph
counter: graphs is the running total at the end of the window, recompiles the
number of new graphs inside it, so a non-zero recompiles on a revisit is a
recompilation.  compile_ms is -1: torch.compile does not expose the compile
time separately from the first iteration that triggers it, so the cost is
inside that iteration's step_us instead.

The sweep order is explicit and is the same for every system: ORDER once per
pass, twice, so pass 2 shows what a revisit of an already-seen length costs.
"""

import json, os, sys, time

import torch
import torch._dynamo

import gpt2_torch
import torch_common

ORDER = [32, 48, 64, 96, 128]
PASSES = 2


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


def graphs_of(mode):
    if mode not in ('compile', 'compile-dynamic'):
        return 0
    return torch._dynamo.utils.counters['stats']['unique_graphs']


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

    fwd = model.forward
    if mode in ('compile', 'compile-dynamic'):
        torch._dynamo.reset()
        fwd = torch.compile(model.forward, dynamic=mode == 'compile-dynamic')

    def build(t):
        ids = [tokens[i % len(tokens)] for i in range(t)]
        idx = torch.tensor(ids, device=dev, dtype=torch.long)
        pos, mask = model.pos_and_mask(t)
        if dev == 'cuda':
            torch.cuda.synchronize()
        return idx, pos, mask

    with torch.no_grad():
        for _ in range(5):
            fwd(*build(ORDER[0])).sum().item()

        per_pass = max(1, iters // PASSES)
        print('pass\tlength\tstep_us\tbuild_us\tloops\tbridges\tkernels'
              '\tlaunches\tgraphs\trecompiles\tcompile_ms')
        for p in range(1, PASSES + 1):
            for i in range(per_pass):
                t = ORDER[i % len(ORDER)]
                g0 = graphs_of(mode)
                b0 = time.time()
                idx, pos, mask = build(t)
                b1 = time.time()
                out = fwd(idx, pos, mask)
                out.sum().item()
                s1 = time.time()
                g1 = graphs_of(mode)
                print('%d\t%d\t%.1f\t%.1f\t0\t0\t0\t0\t%d\t%d\t-1' % (
                    p, t, (s1 - b1) * 1e6, (b1 - b0) * 1e6, g1, g1 - g0))


if __name__ == '__main__':
    main()
