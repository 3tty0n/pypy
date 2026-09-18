"""Data-dependent early exit on distilgpt2, PyTorch side.

The same computation as earlyexit.py, on the same weights, with the same
schedule out of inputs.CONTROL - see that file's docstring for the technique
and for what the timed region contains.

The `.item()` that reads the confidence sits inside the compiled function on
purpose.  Dynamo cannot trace a host read (capture_scalar_outputs is off by
default, and turning it on would only move the branch into the graph as a
data-dependent guard), so each confidence read is a graph break: the stack is
compiled as a chain of fragments joined by Python resume frames, and every
distinct exit depth is a distinct chain.  That is the cost this experiment is
about, so it is reported rather than worked around.

Counters: graphs and breaks come from torch._dynamo.explain, run once after
the timed loop.  recompiles is the delta of
torch._dynamo.utils.counters['frames']['total'] across the timed phase - one
frame compilation per compiled path, so in the varying regime it counts the
recompilations the rotating exit depth forces.

    earlyexit_torch.py MODE WEIGHTS REGIME [ITERS] [WARMUP]
    MODE: eager | compile | compile-ro | compile-mat
"""

import json, os, sys, time

import numpy as np
import torch
import torch._dynamo

import gpt2_torch
import inputs
import torch_common

KS = (0, 1, 2, 3)


def counters():
    c = torch._dynamo.utils.counters
    return c['frames']['total']


def make_forward(model, scale):
    def fwd(idx, tau):
        x = model.wte[idx] + model.pos
        n = 0
        for layer in model.layers:
            x = gpt2_torch.block(model, x, model.mask, layer)
            n += 1
            conf = (x * x).sum().item() * scale
            inputs.control_check(conf, tau)
            if conf >= tau:
                break
        return gpt2_torch.lm_head(model, x), n
    return fwd


def explain(fwd, args):
    """graphs/breaks for the compiled modes; eager has neither."""
    try:
        out = torch._dynamo.explain(fwd)(*args)
        return out.graph_count, out.graph_break_count
    except Exception as exc:
        sys.stderr.write('earlyexit_torch: explain failed: %s\n' % exc)
        return 0, 0


def compare(outdir, logits):
    ref = os.path.join(outdir, 'logits_earlyexit.bin')
    tol = float(os.environ.get('MODEL_TOL', '1e-3'))
    if not os.path.exists(ref):
        return -1.0, tol, 0
    other = np.fromfile(ref, dtype=np.float32)
    mine = logits.detach().float().cpu().numpy().reshape(-1)
    d = float(np.abs(mine - other.reshape(mine.shape)).max())
    return d, tol, int(d <= tol)


def main():
    mode, outdir, regime = sys.argv[1], sys.argv[2], sys.argv[3]
    iters = int(sys.argv[4]) if len(sys.argv) > 4 else 200
    warmup = int(sys.argv[5]) if len(sys.argv) > 5 else 30
    dtname = os.environ.get('RTENSOR_DTYPE', 'float32')
    dtype = getattr(torch, dtname)
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    cfg = json.load(open(os.path.join(outdir, 'index.json')))
    flat = torch_common.flat_weights(outdir)
    model = gpt2_torch.Model(cfg, flat, dtype, dev)
    scale = 1.0 / (cfg['seq'] * cfg['n_embd'])
    idxs = dict((k, torch.tensor(inputs.tokens(cfg['tokens'], cfg['vocab'], k),
                                 device=dev, dtype=torch.long)) for k in KS)

    a = torch_common.Args()
    a.mode, a.dtype = mode, dtype
    raw = make_forward(model, scale)
    torch._dynamo.reset()
    fwd = torch_common.compiled(raw, a)

    with torch.no_grad():
        frames0 = counters()
        us, exits = [], []
        t_all = time.time()
        for i in range(iters):
            k, tau = inputs.control_slot(regime, i)
            t0 = time.time()
            logits, n = fwd(idxs[k], tau)
            logits.sum().item()
            us.append((time.time() - t0) * 1e6)
            exits.append(n)
            print('series\t%d\t%.1f\t%d' % (i, us[-1], n))
        total_ms = (time.time() - t_all) * 1e3
        recompiles = counters() - frames0
        steady = us[warmup:] or us

        if exits[:len(inputs.CONTROL_EXITS[regime])] != inputs.CONTROL_EXITS[regime]:
            raise AssertionError('%s exited at %s, schedule says %s' % (
                regime, exits[:8], inputs.CONTROL_EXITS[regime]))

        ref_k, ref_tau = inputs.CONTROL['stable'][0]
        ref, _ = fwd(idxs[ref_k], ref_tau)
        # compile-ro/compile-mat replay into a static cudagraph output buffer;
        # explain() below runs more forwards, so take the copy first.
        d, tol, ok = compare(outdir, ref.clone())
        graphs, breaks = (0, 0)
        if mode.startswith('compile'):
            graphs, breaks = explain(raw, (idxs[ref_k], ref_tau))

    print('control regime=%s system=torch-%s iters=%d warmup=%d total_ms=%.1f '
          'p50_us=%.1f p95_us=%.1f max_us=%.1f loops=0 bridges=0 kernels=0 '
          'graphs=%d breaks=%d recompiles=%d exit_hash=%d exit_seq=%s '
          'maxabsdiff=%.6g tol=%.6g pass=%d' % (
              regime, mode, iters, warmup, total_ms,
              inputs.pctl(steady, 0.5), inputs.pctl(steady, 0.95), max(steady),
              graphs, breaks, recompiles, inputs.exit_hash(exits),
              ''.join(str(n) for n in inputs.CONTROL_EXITS[regime]),
              d, tol, ok))


if __name__ == '__main__':
    main()
