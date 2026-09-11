"""Dynamic sequence-length sweep, MetaTensor side.

What the timed step contains - identical to dynamic_gpt2_torch.py:

    build_us  token ids for this length, the position slice, the causal mask
    step_us   model(idx, pos) with that mask, then logits.sum().item(),
              i.e. a reduction all the way back to a host scalar
    total_us  build_us + step_us

Both boundaries are reported for both systems, so nothing depends on which one
a reader prefers.  They are split because the build is host-side input
marshalling, not the runtime under test: here it is a Python list build
(1.9 ms for the 12x128x128 mask, 1.1 ms for the 128x768 position slice on this
host), there it is a tensor slice plus an H2D copy.  Folding that into one
number would make the figure a measurement of list construction.  For the same
reason the causal mask is built once per length here and reused on a revisit
(a Python triple loop over h*t*t), while torch rebuilds it from three tensor
ops every step; that shows up in build_us, never in step_us.

Per length window the driver also reports the delta of the PyPy JIT counters
(loops compiled, bridges compiled, tracing+backend seconds) and of the
MetaTensor counters (kernels compiled, kernel launches).  Cache hits are
launches minus newly compiled kernels.

The sweep order is explicit and is the same for every system: ORDER once per
pass, twice, so pass 2 shows what a revisit of an already-seen length costs.
"""

import os, sys, time

import pypyjit

import _metatensor
import common
import gpt2

ORDER = [32, 48, 64, 96, 128]
PASSES = 2


def jit_counters():
    s = pypyjit.get_stats_snapshot()
    c = s.counters
    t = s.counter_times
    return (c['TOTAL_COMPILED_LOOPS'], c['TOTAL_COMPILED_BRIDGES'],
            t.get('TRACING', 0.0) + t.get('BACKEND', 0.0))


def build(cfg, buf, model, masks, t, dtype):
    tokens = cfg['tokens']
    ids = [tokens[i % len(tokens)] for i in range(t)]
    idx = common.tensor([float(tok) for tok in ids], [t], False, dtype)
    d = cfg['n_embd']
    wpe_off = cfg['index']['wpe'][0]
    pos = common.tensor(list(buf[wpe_off:wpe_off + t * d]), [t, d], False,
                        dtype)
    mask = masks.get(t)
    if mask is None:
        mask = masks[t] = common.causal_mask(cfg['n_head'], t, dtype)
    for blk in model.blocks:
        blk.attn.mask = mask
    return idx, pos


def main():
    outdir = sys.argv[1]
    iters = int(sys.argv[2]) if len(sys.argv) > 2 else 200
    dtype = os.environ.get('RTENSOR_DTYPE', 'float32')
    cfg, buf = common.load(outdir)
    model = gpt2.build_model(cfg, buf, dtype)
    masks = {}

    for _ in range(5):
        idx, pos = build(cfg, buf, model, masks, ORDER[0], dtype)
        model(idx, pos).sum().item()

    per_pass = max(1, iters // PASSES)
    print('pass\tlength\tstep_us\tbuild_us\tloops\tbridges\tkernels'
          '\tlaunches\tgraphs\trecompiles\tcompile_ms')
    for p in range(1, PASSES + 1):
        for i in range(per_pass):
            t = ORDER[i % len(ORDER)]
            loops0, bridges0, ct0 = jit_counters()
            k0 = _metatensor.kernel_count()
            l0 = _metatensor.launch_count()
            b0 = time.time()
            idx, pos = build(cfg, buf, model, masks, t, dtype)
            b1 = time.time()
            out = model(idx, pos)
            out.sum().item()
            s1 = time.time()
            loops1, bridges1, ct1 = jit_counters()
            print('%d\t%d\t%.1f\t%.1f\t%d\t%d\t%d\t%d\t0\t0\t%.3f' % (
                p, t, (s1 - b1) * 1e6, (b1 - b0) * 1e6,
                loops1 - loops0, bridges1 - bridges0,
                _metatensor.kernel_count() - k0,
                _metatensor.launch_count() - l0,
                (ct1 - ct0) * 1e3))


if __name__ == '__main__':
    main()
