import array, json, os, sys, time

import torch


class Args(object):
    pass


def argv():
    a = Args()
    a.mode = sys.argv[1]
    a.outdir = sys.argv[2]
    a.iters = int(sys.argv[3]) if len(sys.argv) > 3 else 20
    a.warmup = int(sys.argv[4]) if len(sys.argv) > 4 else 10
    a.dtname = os.environ.get('RTENSOR_DTYPE', 'float32')
    a.dtype = getattr(torch, a.dtname)
    a.dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    a.cfg = json.load(open(os.path.join(a.outdir, 'index.json')))
    return a


def batch_argv():
    return int(sys.argv[5]) if len(sys.argv) > 5 else 1


def flat_weights(outdir):
    buf = array.array('f')
    path = os.path.join(outdir, 'weights.bin')
    buf.fromfile(open(path, 'rb'), os.path.getsize(path) // 4)
    if sys.byteorder != 'little':
        buf.byteswap()
    return torch.tensor(buf, dtype=torch.float32)


def image(a):
    off, shape = a.cfg['index']['image']
    n = shape[0] * shape[1] * shape[2]
    buf = array.array('f')
    path = os.path.join(a.outdir, 'weights.bin')
    buf.fromfile(open(path, 'rb'), os.path.getsize(path) // 4)
    return torch.tensor(buf[off:off + n]).view(1, *shape)


def timed(fwd, args, a):
    """Warm-up, sync, timed loop, sync.  The first forward is timed on its
    own into a.first_run_ms: for compile mode that is compile plus one run."""
    with torch.no_grad():
        t0 = time.time()
        logits = fwd(*args)
        if a.dev == 'cuda':
            torch.cuda.synchronize()
        a.first_run_ms = (time.time() - t0) * 1e3
        for i in range(a.warmup - 1):
            logits = fwd(*args)
        if a.dev == 'cuda':
            torch.cuda.synchronize()
        t0 = time.time()
        for i in range(a.iters):
            logits = fwd(*args)
        if a.dev == 'cuda':
            torch.cuda.synchronize()
        acc = logits.double().sum().item()
        steady_us = (time.time() - t0) / a.iters * 1e6
    return logits, acc, steady_us


def compare(outdir, logits, argmax=None):
    ref = os.path.join(outdir, 'logits_pypy.bin')
    if not os.path.exists(ref):
        return ''
    other = torch.frombuffer(open(ref, 'rb').read(),
                             dtype=torch.float32).to(logits.device)
    other = other.view_as(logits.float())
    diff = ' maxabsdiff=%.6g' % (logits.float() - other).abs().max().item()
    if argmax is not None:
        mine = other.argmax(-1).tolist()
        diff += ' argmax_match=%d/%d' % (
            sum(int(x == y) for x, y in zip(argmax, mine)), len(argmax))
    return diff


def report(line, order, diff, a=None):
    if a is not None and getattr(a, 'first_run_ms', None) is not None:
        line += ' compile_ms=-1 first_run_ms=%.1f' % a.first_run_ms
    print(line)
    print('argmax %s%s' % (' '.join(str(a) for a in order), diff))
