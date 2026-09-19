import array, json, os, sys, time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                '..', '..', 'lib_pypy'))

import _metatensor
import inputs
import warmup_common


def load(outdir):
    cfg = json.load(open(os.path.join(outdir, 'index.json')))
    buf = array.array('f')
    path = os.path.join(outdir, 'weights.bin')
    buf.fromfile(open(path, 'rb'), os.path.getsize(path) // 4)
    if sys.byteorder != 'little':
        buf.byteswap()
    # INPUT_SEED=k>0 asks for a derived input (see inputs.py); the model code
    # below reads the tokens and the image from these two places, so this is
    # the only place any of it has to know.
    k = inputs.seed()
    if k:
        inputs.perturb_cfg(cfg, k)
        span = inputs.image_span(cfg)
        if span:
            off, n = span
            for i in range(n):
                buf[off + i] = buf[off + i] + inputs.noise(i, k)
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

    def tiled(self, name, batch):
        values, shape = self.raw(name)
        if batch == 1:
            return _metatensor.tensor(values, shape, False, self.dtype)
        return _metatensor.tensor(values * batch,
                                  [shape[0] * batch] + shape[1:], False,
                                  self.dtype)

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
# Dead-code guard: kernel launches the timed loop issued, per iteration.  The
# loop only observes its result once, so without a counter nothing proves the
# other iterations executed at all; a launch count that tracks the iteration
# count does.
LAUNCHES_PER_ITER = [None]
# RTENSOR_STATS=1: the kernel-name counter either side of the run, so a
# caller can tell which rtensor_k<n>.ttir files this model's fusion regions
# produced (everything below the first value is the start-up single-op set).
KERNEL_WINDOW = [None, None]
# RTENSOR_LAZY_STATS=1: collector time over the timed loop and the peak heap,
# plus the deferred-execution counters.  Where the operation DAG is built
# shows up here: the fusion pass builds it once when a trace is optimized, a
# deferred library builds it again on every iteration, and the objects that
# takes are what the collector then has to walk.
GC_MS = [None]
PEAK_BYTES = [None]


JIT_COUNTERS = [None, None]


_LAZY = [None]


def mark_step():
    """Deferred execution needs an explicit per-iteration materialization
    point, the way PyTorch/XLA needs mark_step; without one the operation
    graph grows across iterations instead of being executed.  With the fusion
    pass on this is a no-op: the trace boundary already is one."""
    if _LAZY[0] is None:
        _LAZY[0] = (hasattr(_metatensor, 'mark_step') and
                    _metatensor.lazy_enabled())
    if _LAZY[0]:
        _metatensor.mark_step()


def _jit_counters():
    """Compiled loops and bridges so far.  The deferred arm runs with the
    tensor pass off, so its guards are the interpreter's own; counting them
    on both arms is what says whether the difference is guard work or host
    work."""
    try:
        import pypyjit
        c = pypyjit.get_stats_snapshot().counters
        return c['TOTAL_COMPILED_LOOPS'], c['TOTAL_COMPILED_BRIDGES']
    except (ImportError, KeyError, AttributeError):
        return 0, 0


def _gc_stats():
    """PyPy formats the memory fields of gc.get_stats() as strings; the raw
    integers are on the stats object underneath.  total_gc_time is
    milliseconds spent collecting."""
    try:
        import gc
        s = gc.get_stats()._s
        return s.total_gc_time, s.peak_memory
    except (ImportError, AttributeError):
        return 0, 0


def timed(model, args, iters, warmup):
    n = warmup_common.trace_n()
    if n:
        has_counters = hasattr(_metatensor, 'kernel_compile_count')
        us, compiled = [], []
        prev_kc = _metatensor.kernel_compile_count() if has_counters else 0
        prev_lc = _metatensor.launch_count() if has_counters else 0
        for i in range(n):
            t0 = time.time()
            logits = model(*args)
            logits.sum().item()
            mark_step()
            dt = (time.time() - t0) * 1e6
            us.append(dt)
            if has_counters:
                kc, lc = _metatensor.kernel_compile_count(), _metatensor.launch_count()
                c, l = kc - prev_kc, lc - prev_lc
                prev_kc, prev_lc = kc, lc
                compiled.append(c)
                print('iter=%d us=%.1f compiled=%d launches=%d' % (i, dt, c, l))
            else:
                print('iter=%d us=%.1f' % (i, dt))
        warmup_common.report_trace(us, compiled if has_counters else None)
        sys.exit(0)
    KERNEL_WINDOW[0] = _metatensor.kernel_count()
    # The first forward carries the tracing and kernel compilation; it is
    # timed on its own so the compile cost can sit next to the steady state.
    t0 = time.time()
    logits = model(*args)
    logits.sum().item()
    FIRST_RUN_MS[0] = (time.time() - t0) * 1e3
    for i in range(warmup - 1):
        logits = model(*args)
        mark_step()
    logits.sum().item()
    launches0 = _metatensor.launch_count()
    gc0 = _gc_stats() if os.environ.get('RTENSOR_LAZY_STATS') else (0, 0)
    jit0 = _jit_counters() if os.environ.get('RTENSOR_LAZY_STATS') else (0, 0)
    t0 = time.time()
    for i in range(iters):
        logits = model(*args)
        mark_step()
    acc = logits.sum().item()
    steady_us = (time.time() - t0) / iters * 1e6
    if os.environ.get('RTENSOR_LAZY_STATS'):
        gc1 = _gc_stats()
        GC_MS[0] = gc1[0] - gc0[0]
        PEAK_BYTES[0] = gc1[1]
        jit1 = _jit_counters()
        JIT_COUNTERS[0] = jit1[0] - jit0[0]
        JIT_COUNTERS[1] = jit1[1] - jit0[1]
    # Read after acc: the graph is lazy, so the launches happen when the
    # result is forced, not when the expression is built.
    LAUNCHES_PER_ITER[0] = float(_metatensor.launch_count() - launches0) / iters
    KERNEL_WINDOW[1] = _metatensor.kernel_count()
    return logits, acc, steady_us


def causal_mask(h, t, dtype):
    mask = [0.0] * (h * t * t)
    for head in range(h):
        for i in range(t):
            for j in range(i + 1, t):
                mask[(head * t + i) * t + j] = -1e9
    return tensor(mask, [h * t, t], False, dtype)


def row_block(logits, seq, b, dtype):
    """Rows [b*seq, (b+1)*seq) of a [batch*seq, cols] tensor."""
    idx = tensor([float(i) for i in range(b * seq, (b + 1) * seq)], [seq],
                 False, dtype)
    return logits.take(idx)


def batch_rows_identical(logits, seq, batch, dtype):
    """The correctness check for the batching itself: every sequence in the
    batch is the same token ids, so every block of `seq` output rows must come
    out bit-identical to the first.  Compared as a sum of squared differences
    so the whole check stays on the device - materialising B*seq*vocab floats
    as a Python list is gigabytes at batch 32."""
    if batch <= 1:
        return 1
    base = row_block(logits, seq, 0, dtype)
    for b in range(1, batch):
        d = row_block(logits, seq, b, dtype).sub(base)
        if d.mul(d).sum().item() != 0.0:
            return 0
    return 1


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
    cannot overwrite it, and so does every INPUT_SEED>0 input."""
    dtype = dtype or os.environ.get('RTENSOR_DTYPE', 'float32')
    return inputs.reference_name(dtype, inputs.seed())


def dump(outdir, flat):
    out = array.array('f', flat)
    if sys.byteorder != 'little':
        out.byteswap()
    out.tofile(open(os.path.join(outdir, reference_name()), 'wb'))


def report(line, order):
    if LAUNCHES_PER_ITER[0] is not None:
        line += ' launches_per_iter=%.1f' % LAUNCHES_PER_ITER[0]
    if os.environ.get('RTENSOR_STATS') and KERNEL_WINDOW[1] is not None:
        line += ' kernel_count_begin=%d kernel_count_end=%d' % (
            KERNEL_WINDOW[0], KERNEL_WINDOW[1])
    if os.environ.get('RTENSOR_LAZY_STATS'):
        if GC_MS[0] is not None:
            line += ' gc_ms=%d peak_bytes=%d' % (GC_MS[0], PEAK_BYTES[0])
        if hasattr(_metatensor, 'lazy_stats'):
            f, n, b, sc, fb = _metatensor.lazy_stats()
            line += (' deferred=%d lazy_forces=%d lazy_nodes=%d'
                     ' lazy_barriers=%d lazy_pruned=%d lazy_fallbacks=%d'
                     % (1 if _metatensor.lazy_enabled() else 0, f, n, b, sc, fb))
        if JIT_COUNTERS[0] is not None:
            line += ' loops=%d bridges=%d' % (JIT_COUNTERS[0], JIT_COUNTERS[1])
        if hasattr(_metatensor, 'kernel_compile_count'):
            line += ' kernels=%d compiles=%d' % (
                _metatensor.kernel_count(), _metatensor.kernel_compile_count())
    if FIRST_RUN_MS[0] is not None:
        line += ' compile_ms=-1 first_run_ms=%.1f' % FIRST_RUN_MS[0]
    print(line)
    print('argmax %s' % ' '.join([str(a) for a in order]))


tensor = _metatensor.tensor
