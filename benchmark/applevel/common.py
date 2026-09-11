import array, json, os, sys, time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                '..', '..', 'lib_pypy'))

import _metatensor


def load(outdir):
    cfg = json.load(open(os.path.join(outdir, 'index.json')))
    buf = array.array('f')
    path = os.path.join(outdir, 'weights.bin')
    buf.fromfile(open(path, 'rb'), os.path.getsize(path) // 4)
    if sys.byteorder != 'little':
        buf.byteswap()
    return cfg, buf


class Weights(object):
    def __init__(self, cfg, buf, dtype):
        self.index = cfg['index']
        self.buf = buf
        self.dtype = dtype

    def raw(self, name):
        off, shape = self.index[name]
        n = 1
        for d in shape:
            n *= d
        return list(self.buf[off:off + n]), shape

    def get(self, name):
        values, shape = self.raw(name)
        return _metatensor.tensor(values, shape, False, self.dtype)

    def cat(self, names):
        parts = [self.raw(n) for n in names]
        shape = parts[0][1]
        if len(shape) == 1:
            out = []
            for values, _ in parts:
                out.extend(values)
            return _metatensor.tensor(out, [len(out)], False, self.dtype)
        rows, cols = shape
        out = []
        for r in range(rows):
            for values, _ in parts:
                out.extend(values[r * cols:(r + 1) * cols])
        return _metatensor.tensor(out, [rows, cols * len(parts)], False,
                                  self.dtype)


def argv():
    outdir = sys.argv[1]
    iters = int(sys.argv[2]) if len(sys.argv) > 2 else 20
    warmup = int(sys.argv[3]) if len(sys.argv) > 3 else 10
    dtype = os.environ.get('RTENSOR_DTYPE', 'float32')
    return outdir, iters, warmup, dtype


def batch_argv():
    return int(sys.argv[4]) if len(sys.argv) > 4 else 1


FIRST_RUN_MS = [None]


def timed(model, args, iters, warmup):
    # The first forward carries the tracing and kernel compilation; it is
    # timed on its own so the compile cost can sit next to the steady state.
    t0 = time.time()
    logits = model(*args)
    logits.sum().item()
    FIRST_RUN_MS[0] = (time.time() - t0) * 1e3
    for i in range(warmup - 1):
        logits = model(*args)
    logits.sum().item()
    t0 = time.time()
    for i in range(iters):
        logits = model(*args)
    acc = logits.sum().item()
    steady_us = (time.time() - t0) / iters * 1e6
    return logits, acc, steady_us


def causal_mask(h, t, dtype):
    mask = [0.0] * (h * t * t)
    for head in range(h):
        for i in range(t):
            for j in range(i + 1, t):
                mask[(head * t + i) * t + j] = -1e9
    return tensor(mask, [h * t, t], False, dtype)


def token_argmax(flat, t, v):
    argmax = []
    for i in range(t):
        best, bestj = flat[i * v], 0
        for j in range(1, v):
            if flat[i * v + j] > best:
                best, bestj = flat[i * v + j], j
        argmax.append(bestj)
    return argmax


def top5(flat):
    return sorted(range(len(flat)), key=lambda j: -flat[j])[:5]


def reference_name(dtype=None):
    """logits_pypy.bin is the float32 reference every system is checked
    against; other dtypes get their own file so a float16 ablation run
    cannot overwrite it."""
    dtype = dtype or os.environ.get('RTENSOR_DTYPE', 'float32')
    if dtype == 'float32':
        return 'logits_pypy.bin'
    return 'logits_pypy_%s.bin' % dtype


def dump(outdir, flat):
    out = array.array('f', flat)
    if sys.byteorder != 'little':
        out.byteswap()
    out.tofile(open(os.path.join(outdir, reference_name()), 'wb'))


def report(line, order):
    if FIRST_RUN_MS[0] is not None:
        line += ' compile_ms=-1 first_run_ms=%.1f' % FIRST_RUN_MS[0]
    print(line)
    print('argmax %s' % ' '.join([str(a) for a in order]))


tensor = _metatensor.tensor
