"""Greedy autoregressive decode with a key/value cache, MetaTensor side.

    gpt2_decode.py WEIGHTS_DIR [NEW] [ROUNDS]

The prompt is the model's exported token sequence.  Its first seq-1 tokens
are prefilled - one forward that also writes every row's q|k|v into a
[seq+NEW, 3d] cache per block - and then NEW tokens are generated one at a
time, starting from the last prompt token.  A step is

    token id upload, embedding + position row, every block on one row
    against the cache, the language-model head, argmax back to a host int

so the next step's input exists only once the previous one has finished,
exactly as in a serving loop.  `token_us` is the wall time of that step.

ROUNDS generations of the same prompt run in one process.  Round 0 is cold:
the JIT traces the step, and every new cache length it meets is a new value
for the promoted sizes.  The later rounds revisit lengths it has seen, which
is steady-state serving; their per-token median is the headline and round 0
is reported beside it.

`prefill_ms` is the last round's prefill.  Prefill runs once per round, so
on the MetaTensor side it reaches the JIT only after a few rounds; it is not
the quantity this driver measures.

Correctness is checked twice: against the torch mirror's token stream by the
harness, and here, by one full causal forward over the prompt and the
generated tokens, whose argmax at every generated position must equal the
token decode produced there.
"""

import os
import sys
import time

import _metatensor
import common
import gpt2


def median(xs):
    s = sorted(xs)
    k = len(s)
    return 0.0 if not k else (s[k // 2] if k % 2
                              else 0.5 * (s[k // 2 - 1] + s[k // 2]))


def main():
    outdir = sys.argv[1]
    new = int(sys.argv[2]) if len(sys.argv) > 2 else 64
    rounds = int(sys.argv[3]) if len(sys.argv) > 3 else 5
    dtype = os.environ.get('RTENSOR_DTYPE', 'float32')
    cfg, buf = common.load(outdir)
    p = cfg['seq']
    d = cfg['n_embd']
    h = cfg['n_head']
    vocab = cfg['vocab']
    tokens = cfg['tokens']
    tmax = p + new

    model = gpt2.build_model(cfg, buf, dtype, common.causal_mask(h, p - 1,
                                                                 dtype))
    w = common.Weights(cfg, buf, dtype)
    wpe = w.get('wpe')
    wpe_off = cfg['index']['wpe'][0]
    idx0 = common.tensor([float(x) for x in tokens[:p - 1]], [p - 1], False,
                         dtype)
    pos0 = common.tensor(list(buf[wpe_off:wpe_off + (p - 1) * d]),
                         [p - 1, d], False, dtype)
    pos_ids = [common.tensor([float(t)], [1], False, dtype)
               for t in range(tmax)]
    caches = [_metatensor.zeros([tmax, 3 * d], False, dtype)
              for _ in model.blocks]

    def generate():
        t0 = time.time()
        model.prefill(idx0, pos0, caches).sum().item()
        prefill_ms = (time.time() - t0) * 1e3
        tok = tokens[p - 1]
        out = []
        us = []
        for t in range(p - 1, p - 1 + new):
            t0 = time.time()
            x = common.tensor([float(tok)], [1], False, dtype)
            tok = model.step(x, wpe.take(pos_ids[t]), caches, t).argmax()
            us.append((time.time() - t0) * 1e6)
            out.append(tok)
        return out, us, prefill_ms

    results = []
    l0 = _metatensor.launch_count()
    for r in range(rounds):
        if r == 1:
            l0 = _metatensor.launch_count()
        results.append(generate())
    steady = rounds - 1 if rounds > 1 else 1
    launches = (_metatensor.launch_count() - l0) / float(steady * new)
    retained = -1
    if hasattr(_metatensor, 'live_bytes'):
        retained = _metatensor.live_bytes()

    out = results[0][0]
    same = 1
    for o, _, _ in results:
        if o != out:
            same = 0

    # Teacher-forced check: position p-1+i of the full forward predicts
    # generated token i.
    seq = tokens[:p] + out[:-1]
    n = len(seq)
    full = gpt2.build_model(cfg, buf, dtype, common.causal_mask(h, n, dtype))
    pos = common.tensor(list(buf[wpe_off:wpe_off + n * d]), [n, d], False,
                        dtype)
    flat = full(common.tensor([float(x) for x in seq], [n], False, dtype),
                pos).tolist()
    forced = 1
    for i in range(new):
        row = flat[(p - 1 + i) * vocab:(p + i) * vocab]
        if row.index(max(row)) != out[i]:
            forced = 0
            sys.stderr.write('gpt2_decode: token %d: decode %d, full forward '
                             '%d\n' % (i, out[i], row.index(max(row))))
            break

    cold = results[0][1]
    warm = []
    for _, us, _ in results[1:]:
        warm.extend(us)
    if not warm:
        warm = cold
    warm_sorted = sorted(warm)
    print('decode pypy layers=%d embd=%d prompt=%d new=%d rounds=%d dtype=%s '
          'token_us=%.1f p90_us=%.1f cold_token_us=%.1f prefill_ms=%.2f '
          'launches_per_token=%.2f retained_bytes=%d rounds_agree=%d '
          'teacher_forced=%d tokens=%s'
          % (cfg['n_layer'], d, p, new, rounds, dtype, median(warm),
             warm_sorted[int(0.9 * (len(warm_sorted) - 1))], median(cold),
             results[-1][2],
             launches, retained, same, forced,
             ','.join([str(x) for x in out])))
    return 0 if (same and forced) else 1


if __name__ == '__main__':
    sys.exit(main())
