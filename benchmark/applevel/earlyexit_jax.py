"""Data-dependent early exit on distilgpt2, JAX/XLA side.

The same computation as earlyexit.py, on the same weights, with the same
schedule out of inputs.CONTROL.  Reuses jax_models.py's GPT-2 transcription
(Weights, layer_norm, gelu_tanh, block_params, gpt2_block), so the three
systems still compute the same function.

A staged jax.jit cannot branch on a traced value, so the technique has to be
expressed some other way, and what that costs is the thing to report.  Two
rows, both measured the same way:

  jax-perlayer  the literal transcription: one jit per block, returning the
                block output and its confidence, and the confidence read back
                to the host with float(...) - a device-to-host sync per layer,
                which is where the cost is.  The Python `if` is then a real
                host branch, exactly as in the other two systems, and a
                shorter exit really executes fewer blocks.

  jax-while     the staged rewrite: lax.while_loop over lax.switch across the
                six blocks, with the threshold as a traced argument.  No sync
                and one executable for every threshold - but the exit
                condition, the classifier and the stopping rule now have to be
                expressible inside XLA, and nothing about the decision is
                visible to the host until the whole loop has finished.

Counters: `graphs` is the number of traced (hence compiled) executables and
`recompiles` is how many of those traces happened after the warm-up window.

    earlyexit_jax.py MODE WEIGHTS REGIME [ITERS] [WARMUP]
    MODE: perlayer | while
"""

import os, sys, time

import numpy as np

import inputs
import jax_models

import jax
import jax.numpy as jnp
from jax import lax

KS = (0, 1, 2, 3)
TRACES = [0]


def traced(fn):
    """Counts compilations: a jit traces its body once per compiled variant."""
    def wrapper(*args, **kw):
        TRACES[0] += 1
        return fn(*args, **kw)
    return wrapper


def build(cfg, flat, dtype):
    w = jax_models.Weights(cfg, flat, dtype)
    t = cfg['seq']
    params = {'wte': w.get('wte'), 'pos': w.get('wpe')[:t],
              'gf': w.get('ln_f.g'), 'bf': w.get('ln_f.b'),
              'blocks': [jax_models.block_params(w, i)
                         for i in range(cfg['n_layer'])]}
    mask = jnp.where(jnp.triu(jnp.ones((t, t), bool), 1), -1e9, 0.0)
    params['mask'] = mask.astype(dtype)
    return params


def perlayer_runner(params, cfg, dtype):
    h, eps = cfg['n_head'], cfg['eps']

    @jax.jit
    @traced
    def embed(wte, pos, idx):
        return wte[idx] + pos

    # One trace serves all six blocks: the weights are arguments, and every
    # block has the same shapes.
    @jax.jit
    @traced
    def step(bp, mask, x):
        x = jax_models.gpt2_block(x, bp, h, mask, eps, jax_models.gelu_tanh)
        return x, (x * x).mean()

    @jax.jit
    @traced
    def head(gf, bf, wte, x):
        return jax_models.layer_norm(x, gf, bf, eps) @ wte.T

    def run(idx, tau):
        x = embed(params['wte'], params['pos'], idx)
        n = 0
        for bp in params['blocks']:
            x, c = step(bp, params['mask'], x)
            n += 1
            conf = float(c)          # the device-to-host sync
            inputs.control_check(conf, tau)
            if conf >= tau:
                break
        return head(params['gf'], params['bf'], params['wte'], x), n

    return run


def while_runner(params, cfg, dtype):
    h, eps, nl = cfg['n_head'], cfg['eps'], cfg['n_layer']

    @jax.jit
    @traced
    def run_jit(p, idx, tau):
        x = p['wte'][idx] + p['pos']
        branches = [(lambda x, bp=bp: jax_models.gpt2_block(
            x, bp, h, p['mask'], eps, jax_models.gelu_tanh))
            for bp in p['blocks']]

        def body(carry):
            i, x, _ = carry
            x = lax.switch(i, branches, x)
            return i + 1, x, (x * x).mean()

        def cond(carry):
            i, _, c = carry
            return jnp.logical_and(i < nl, c < tau)

        i, x, _ = lax.while_loop(
            cond, body, (0, x, jnp.asarray(0.0, x.dtype)))
        return jax_models.layer_norm(x, p['gf'], p['bf'], eps) @ p['wte'].T, i

    def run(idx, tau):
        logits, i = run_jit(params, idx, jnp.asarray(tau, dtype))
        return logits, i

    return run


def compare(outdir, logits):
    # MODEL_TOL is a fraction of the reference's largest absolute value, the
    # same rule the model sweep uses (config.sh's tolerance_for).
    ref = os.path.join(outdir, 'logits_earlyexit.bin')
    frac = float(os.environ.get('MODEL_TOL', '2e-5'))
    if not os.path.exists(ref):
        return -1.0, frac, 0
    mine = np.asarray(logits, dtype=np.float32).reshape(-1)
    other = np.fromfile(ref, dtype=np.float32)
    d = float(np.abs(mine - other).max())
    tol = frac * float(np.abs(other).max())
    return d, tol, int(d <= tol)


def main():
    mode, outdir, regime = sys.argv[1], sys.argv[2], sys.argv[3]
    iters = int(sys.argv[4]) if len(sys.argv) > 4 else 200
    warmup = int(sys.argv[5]) if len(sys.argv) > 5 else 30
    dtname = os.environ.get('RTENSOR_DTYPE', 'float32')
    dtype = jnp.dtype(dtname)
    cfg, flat = jax_models.load(outdir)
    params = build(cfg, flat, dtype)
    jax.block_until_ready(params)
    idxs = dict((k, jnp.asarray(inputs.tokens(cfg['tokens'], cfg['vocab'], k),
                                dtype=jnp.int32)) for k in KS)
    run = {'perlayer': perlayer_runner, 'while': while_runner}[mode](
        params, cfg, dtype)

    us, exits = [], []
    traces_at_warmup = [0]
    t_all = time.time()
    for i in range(iters):
        k, tau = inputs.control_slot(regime, i)
        t0 = time.time()
        logits, n = run(idxs[k], tau)
        float(logits.sum())
        us.append((time.time() - t0) * 1e6)
        exits.append(int(n))
        if i == warmup - 1:
            traces_at_warmup[0] = TRACES[0]
        print('series\t%d\t%.1f\t%d' % (i, us[-1], exits[-1]))
    total_ms = (time.time() - t_all) * 1e3
    steady = us[warmup:] or us

    if exits[:len(inputs.CONTROL_EXITS[regime])] != inputs.CONTROL_EXITS[regime]:
        raise AssertionError('%s exited at %s, schedule says %s' % (
            regime, exits[:8], inputs.CONTROL_EXITS[regime]))

    ref_k, ref_tau = inputs.CONTROL['stable'][0]
    ref, _ = run(idxs[ref_k], ref_tau)
    d, tol, ok = compare(outdir, jax.block_until_ready(ref))

    print('control regime=%s system=jax-%s iters=%d warmup=%d total_ms=%.1f '
          'p50_us=%.1f p95_us=%.1f max_us=%.1f loops=0 bridges=0 kernels=0 '
          'graphs=%d breaks=0 frame_compiles=%d exit_hash=%d exit_seq=%s '
          'maxabsdiff=%.6g tol=%.6g pass=%d' % (
              regime, mode, iters, warmup, total_ms,
              inputs.pctl(steady, 0.5), inputs.pctl(steady, 0.95), max(steady),
              TRACES[0], TRACES[0] - traces_at_warmup[0],
              inputs.exit_hash(exits),
              ''.join(str(n) for n in inputs.CONTROL_EXITS[regime]),
              d, tol, ok))


if __name__ == '__main__':
    main()
