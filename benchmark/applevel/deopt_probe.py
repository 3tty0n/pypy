"""Cost of deoptimization: what a guard failure costs, against the steady state.

The loop body is micro.py's variant 3 ("value guard"), the data-dependent
branch that forces the fused chain back to a host scalar:

    h = (x * b + b).relu() * s
    if h.sum().item() > 0.0: h = h + b
    else:                    h = h * b

`s` is a promoted scalar (interp_tensor.scalar promotes its value), so the
schedule of `s` decides what the warm trace's guards see:

  never      s = +1 always; the branch never flips, nothing ever fails.
  alternate  s = +1 through warm-up, then flips every iteration; the first
             timed iteration that goes the other way is the first guard
             failure and compiles a bridge.
  both-hot   s flips through warm-up as well, so both paths are compiled
             before the timed loop starts.
  fresh      s is a different value on every iteration, so the promoted-value
             guard fails every time and no bridge can ever become the hot
             path (watch `bridges`: the JIT stops compiling them).

Every iteration is timed, warm-up included, so the reported numbers are

  steady_us       median of the last quarter of the timed iterations
  first_fail_us   the iteration where the guard first fails (known from the
                  schedule; -1 when the pattern has no such point)
  after_fail_us   median of the five iterations after that one
  peak_us/peak_i  the most expensive timed iteration and its index.  With
                  trace_eagerness=2 the bridge is not traced on the first
                  failure but a few failures later, so this - not
                  first_fail_us - is where the bridge compile lands.
  cold_us         iteration 0, the cold trace plus the first kernel compile

The same file runs the torch side (mode eager/compile/compile-ro) on the same
schedules: `.item()` is a graph break on every iteration there, so the three
numbers are expected to coincide.

    deopt_probe.py PATTERN [ITERS] [WARMUP] [--mode M] [--trace]
"""

import os
import sys
import time

N = 65536
PATTERNS = ('never', 'alternate', 'both-hot', 'fresh')


def schedule(pattern, warm, iters):
    """(per-iteration multiplier, index of the first guard failure or -1)."""
    total = warm + iters
    if pattern == 'never':
        return [1.0] * total, -1
    if pattern == 'alternate':
        tail = [1.0 if i % 2 == 0 else -1.0 for i in range(iters)]
        return [1.0] * warm + tail, warm + 1
    if pattern == 'both-hot':
        return [1.0 if i % 2 == 0 else -1.0 for i in range(total)], -1
    if pattern == 'fresh':
        return [float(i + 1) for i in range(total)], warm
    raise SystemExit("deopt_probe.py: unknown pattern %r (have %s)"
                     % (pattern, ", ".join(PATTERNS)))


def median(xs):
    xs = sorted(xs)
    n = len(xs)
    if not n:
        return -1.0
    if n % 2:
        return xs[n // 2]
    return (xs[n // 2 - 1] + xs[n // 2]) / 2.0


def summarize(times, warm, fail_at):
    timed = times[warm:]
    tail = timed[len(timed) * 3 // 4:] or timed
    first = times[fail_at] if 0 <= fail_at < len(times) else -1.0
    after = -1.0
    if 0 <= fail_at < len(times) - 1:
        after = median(times[fail_at + 1:fail_at + 6])
    peak = max(timed)
    return median(tail), first, after, peak, warm + timed.index(peak)


# --- MetaTensor ------------------------------------------------------------

def run_ours(pattern, iters, warm, trace):
    import _metatensor
    try:
        import pypyjit
    except ImportError:
        pypyjit = None

    def counters():
        loops = bridges = 0
        if pypyjit is not None:
            c = pypyjit.get_stats_snapshot().counters
            loops = c['TOTAL_COMPILED_LOOPS']
            bridges = c['TOTAL_COMPILED_BRIDGES']
        return loops, bridges

    mult, fail_at = schedule(pattern, warm, iters)
    x = _metatensor.tensor([float((i % 7) - 3) for i in range(N)])
    b = _metatensor.tensor([0.5] * N)
    times = [0.0] * len(mult)
    acc = 0.0
    l0 = b0 = k0 = lch0 = 0
    for i in range(len(mult)):
        if i == warm:
            l0, b0 = counters()
            k0 = _metatensor.kernel_count()
            lch0 = _metatensor.launch_count()
        t0 = time.time()
        s = _metatensor.scalar(mult[i], 'float64')
        h = (x * b + b).relu() * s
        if h.sum().item() > 0.0:
            h = h + b
        else:
            h = h * b
        acc += h.sum().item()
        times[i] = (time.time() - t0) * 1e6
    loops, bridges = counters()
    stats = {'loops': loops - l0, 'bridges': bridges - b0,
             'kernels': _metatensor.kernel_count() - k0,
             'launches_per_iter':
                 float(_metatensor.launch_count() - lch0) / iters}
    return times, fail_at, acc, stats


# --- torch -----------------------------------------------------------------

def run_torch(mode, pattern, iters, warm, trace):
    import torch
    dev = 'cuda'
    dt = {'float64': torch.float64, 'float32': torch.float32,
          'float16': torch.float16}[os.environ.get('TORCH_DTYPE', 'float64')]

    def step(h, b, s):
        h = torch.relu(h * b + b) * s
        if h.sum().item() > 0.0:
            return h + b
        return h * b

    if mode == 'compile':
        step = torch.compile(step, dynamic=False)
    elif mode == 'compile-ro':
        step = torch.compile(step, dynamic=False, mode='reduce-overhead')

    mult, fail_at = schedule(pattern, warm, iters)
    x = torch.tensor([float((i % 7) - 3) for i in range(N)], dtype=dt,
                     device=dev)
    b = torch.full((N,), 0.5, dtype=dt, device=dev)
    times = [0.0] * len(mult)
    acc = 0.0
    # torch has no bridge; what corresponds to a bridge compile here is a
    # dynamo recompilation of the frame, so `loops` counts the graphs compiled
    # inside the timed region and `bridges` stays -1 (no such thing).
    from torch._dynamo.utils import counters as dyn
    g0 = 0
    for i in range(len(mult)):
        if i == warm:
            g0 = dyn['frames'].get('ok', 0)
        t0 = time.time()
        if mode == 'compile-ro':
            torch.compiler.cudagraph_mark_step_begin()
        h = step(x, b, mult[i])
        acc += h.double().sum().item()
        times[i] = (time.time() - t0) * 1e6
    stats = {'loops': dyn['frames'].get('ok', 0) - g0, 'bridges': -1,
             'kernels': -1, 'launches_per_iter': -1.0}
    return times, fail_at, acc, stats


def selftest():
    """The schedules and the windowing, without a GPU: deopt_probe.py --selftest"""
    m, f = schedule('alternate', 4, 6)
    assert m == [1.0, 1.0, 1.0, 1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0], m
    assert f == 5 and m[f] < 0
    assert schedule('never', 2, 2) == ([1.0] * 4, -1)
    assert schedule('both-hot', 0, 3)[0] == [1.0, -1.0, 1.0]
    assert schedule('fresh', 0, 3) == ([1.0, 2.0, 3.0], 0)
    times = [99.0, 1.0, 1.0, 50.0, 9.0, 9.0, 9.0, 2.0, 2.0, 2.0, 2.0]
    steady, first, after, peak, peak_i = summarize(times, 3, 3)
    assert (first, after, peak, peak_i) == (50.0, 9.0, 50.0, 3), \
        (first, after, peak, peak_i)
    assert steady == 2.0, steady
    assert median([3.0, 1.0]) == 2.0 and median([]) == -1.0
    print("selftest ok")
    return 0


def main(argv):
    if '--selftest' in argv:
        return selftest()
    args = [a for a in argv[1:] if not a.startswith('--')]
    flags = [a for a in argv[1:] if a.startswith('--')]
    if not args:
        print(__doc__)
        return 1
    pattern = args[0]
    iters = int(args[1]) if len(args) > 1 else 200
    warm = int(args[2]) if len(args) > 2 else 30
    mode = 'ours'
    for f in flags:
        if f.startswith('--mode'):
            mode = f.split('=', 1)[1] if '=' in f else argv[argv.index(f) + 1]
    trace = '--trace' in flags

    if mode == 'ours':
        times, fail_at, acc, stats = run_ours(pattern, iters, warm, trace)
    else:
        times, fail_at, acc, stats = run_torch(mode, pattern, iters, warm,
                                               trace)
    steady, first, after, peak, peak_i = summarize(times, warm, fail_at)
    if trace:
        for i in range(len(times)):
            sys.stderr.write("trace %d %d %.1f\n"
                             % (i, 1 if i >= warm else 0, times[i]))
    print("deopt mode=%s pattern=%s iters=%d warmup=%d steady_us=%.1f "
          "first_fail_us=%.1f after_fail_us=%.1f peak_us=%.1f peak_i=%d "
          "cold_us=%.1f launches_per_iter=%.3f "
          "loops=%d bridges=%d kernels=%d acc=%.6g"
          % (mode, pattern, iters, warm, steady, first, after, peak, peak_i,
             times[0], stats['launches_per_iter'], stats['loops'],
             stats['bridges'], stats['kernels'], acc))
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv))
