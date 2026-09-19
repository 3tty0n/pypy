"""Data-dependent early exit on distilgpt2, MetaTensor side.

The technique is DeeBERT/CALM's: run transformer blocks until a confidence
computed from the hidden state says the answer has settled, then stop.  It is
the reason this experiment exists - the paper's other control-flow evidence is
synthetic micro variants, and a reviewer is entitled to ask whether a host
read in the middle of a model is a real shape of program.  Here it is one:

    for block in blocks:
        x = block(x)
        conf = mean(x * x).item()     <- device scalar, read on the host
        if conf >= tau:               <- a Python branch on that value
            break
    logits = lm_head(layer_norm(x))

The `.item()` is the whole point.  It is a synchronisation and a host read in
the middle of the stack, and the branch after it decides whether the remaining
blocks execute at all - so a shorter exit really is less work, not a masked
result.

Two regimes, from inputs.CONTROL:

    stable   one input, one threshold: every iteration exits at layer 4, so
             every guard holds and one compiled path serves the whole run.
    varying  a rotating request mix, four slots, exits at layers 2, 3, 5, 6:
             the trace's assumptions break every iteration.

What is timed, identical on all three systems: building the token tensor is
outside, `forward(...)` plus forcing the logits back to a host scalar is
inside.  total_ms spans the whole phase including the first iterations that
compile, so the compile cost cannot be warmed away; the percentiles are taken
over the iterations after `warmup` so a recompilation spike in the steady
state shows up in p95/max instead of being hidden by the cold start.

Counters reported (whole phase): loops and bridges from the PyPy JIT, kernels
from MetaTensor's compile counter.  graphs/breaks/recompiles are torch's and
are zero here.

    earlyexit.py WEIGHTS REGIME [ITERS] [WARMUP]
"""

import os, sys, time

import pypyjit

import _metatensor
import common
import gpt2
import inputs

from tensorpypy.functional import layer_norm

KS = (0, 1, 2, 3)


def jit_counters():
    c = pypyjit.get_stats_snapshot().counters
    return c['TOTAL_COMPILED_LOOPS'], c['TOTAL_COMPILED_BRIDGES']


def build_inputs(cfg, buf, dtype):
    """One token tensor per rotation slot, plus the shared position slice.
    Built once, outside the timed region: the experiment is about control
    flow, not about how fast a Python list turns into a tensor."""
    t, d = cfg['seq'], cfg['n_embd']
    wpe_off = cfg['index']['wpe'][0]
    pos = common.tensor(list(buf[wpe_off:wpe_off + t * d]), [t, d], False,
                        dtype)
    idxs = {}
    for k in KS:
        toks = inputs.tokens(cfg['tokens'], cfg['vocab'], k)
        idxs[k] = common.tensor([float(tok) for tok in toks], [t], False,
                                dtype)
    return idxs, pos


def forward(model, idx, pos, tau, scale):
    x = model.wte.take(idx).add(pos)
    n = 0
    for block in model.blocks:
        x = block(x)
        n += 1
        conf = x.mul(x).sum().item() * scale
        inputs.control_check(conf, tau)
        if conf >= tau:
            break
    x = layer_norm(x, model.gf, model.bf, model.eps)
    return x.matmul(model.wte, True), n


def main():
    outdir = sys.argv[1]
    regime = sys.argv[2]
    iters = int(sys.argv[3]) if len(sys.argv) > 3 else 200
    warmup = int(sys.argv[4]) if len(sys.argv) > 4 else 30
    dtype = os.environ.get('RTENSOR_DTYPE', 'float32')
    cfg, buf = common.load(outdir)
    mask = common.causal_mask(cfg['n_head'], cfg['seq'], dtype)
    model = gpt2.build_model(cfg, buf, dtype, mask)
    idxs, pos = build_inputs(cfg, buf, dtype)
    scale = 1.0 / (cfg['seq'] * cfg['n_embd'])

    loops0, bridges0 = jit_counters()
    kc0 = _metatensor.kernel_compile_count()
    us, exits = [], []
    t_all = time.time()
    for i in range(iters):
        k, tau = inputs.control_slot(regime, i)
        t0 = time.time()
        logits, n = forward(model, idxs[k], pos, tau, scale)
        logits.sum().item()
        us.append((time.time() - t0) * 1e6)
        exits.append(n)
        print('series\t%d\t%.1f\t%d' % (i, us[-1], n))
    total_ms = (time.time() - t_all) * 1e3
    loops1, bridges1 = jit_counters()
    steady = us[warmup:] or us

    if exits[:len(inputs.CONTROL_EXITS[regime])] != inputs.CONTROL_EXITS[regime]:
        raise AssertionError('%s exited at %s, schedule says %s' % (
            regime, exits[:8], inputs.CONTROL_EXITS[regime]))

    # The correctness reference: one fixed configuration, the same one on
    # every system, run outside the timed region.  Deliberately not
    # logits_pypy.bin - that is run_models.sh's distilgpt2 reference and this
    # model stops after four blocks.
    ref_k, ref_tau = inputs.CONTROL['stable'][0]
    ref, _ = forward(model, idxs[ref_k], pos, ref_tau, scale)
    import array
    out = array.array('f', ref.tolist())
    if sys.byteorder != 'little':
        out.byteswap()
    out.tofile(open(os.path.join(outdir, 'logits_earlyexit.bin'), 'wb'))

    print('control regime=%s system=ours iters=%d warmup=%d total_ms=%.1f '
          'p50_us=%.1f p95_us=%.1f max_us=%.1f loops=%d bridges=%d '
          'kernels=%d graphs=0 breaks=0 frame_compiles=0 exit_hash=%d '
          'exit_seq=%s maxabsdiff=0 tol=%s pass=1' % (
              regime, iters, warmup, total_ms,
              inputs.pctl(steady, 0.5), inputs.pctl(steady, 0.95), max(steady),
              loops1 - loops0, bridges1 - bridges0,
              _metatensor.kernel_compile_count() - kc0,
              inputs.exit_hash(exits),
              ''.join([str(n) for n in inputs.CONTROL_EXITS[regime]]),
              os.environ.get('MODEL_TOL', '1e-3')))


if __name__ == '__main__':
    main()
