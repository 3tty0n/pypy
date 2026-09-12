"""App-level semantic probe for the MetaTensor recovery contract.

One program, warm JIT, five cases.  Each case runs a long enough loop to be
traced (threshold=3 as in benchmark/paper/config.sh), and each one checks the
observable result against a pure-Python reference computed in the same
process:

  a shared-branch  a live fused intermediate used on BOTH sides of a
                   data-dependent branch whose predicate really takes both
                   outcomes after warm-up (the count of each outcome is part
                   of the checked result).
  b exception      a ValueError raised by surrounding Python while the fused
                   chain is live and unforced, caught by try/except; the
                   partial chain is read afterwards and must be correct.
  c effects        a Python list append plus a counter, interleaved with a
                   tensor-value-dependent branch: order and count are the
                   result.
  d1 mut-alias     in-place `Tensor.add_` (-> ops.assign -> tensor_assign)
                   with an alias obtained from `Tensor.detach()`, read after
                   the write.
  d2 mut-hazard    a deferred expression built from a tensor that is then
                   mutated in place before the expression is forced.
  e promotion      the shape and a Python int change halfway through the loop,
                   which fails the guard on the warm trace and compiles a
                   bridge.

Usage:
    pypy-c [--jit ...] recovery_probe.py [--save REF | --ref REF] [--nostats]
                                        [--times]

`--times` runs only the two cases whose guards really fail and times every
iteration of the two cases whose guards
really fail at a known point - (a), whose branch takes both outcomes, and (e),
whose shape and int change at the halfway iteration - and prints a
`deopt-probe` line per case with the cost of the failing iteration, of the
iterations right after it (where the bridge is traced and compiled) and of a
later revisit.

`--save` writes the per-case observed values and effect logs; `--ref` compares
against that file, which is how the interpreter-only (--jit off) run becomes
the reference for the two JIT configurations.  Exit status is non-zero if any
case disagrees with its in-process expectation or with the reference file.
"""

import sys
import time

import _metatensor
import tensorpypy as tp

try:
    import pypyjit
except ImportError:
    pypyjit = None

# get_stats_snapshot() segfaults in a `--jit off` binary (the JIT portal is
# never set up), so that run has to be told not to ask.
if '--nostats' in sys.argv:
    pypyjit = None

N = 64
ITERS = 300
HALF = ITERS // 2


# --times sets this to a list; the two timed cases then stamp every iteration.
# It stays None otherwise so the traced loops are exactly what they were.
TIMES = None


def tick():
    if TIMES is not None:
        TIMES.append(time.time())


def g(v):
    return "%.12g" % v


def base(n):
    return [float((i % 7) - 3) for i in range(n)]


def counters():
    loops = bridges = 0
    if pypyjit is not None:
        try:
            c = pypyjit.get_stats_snapshot().counters
            loops = c['TOTAL_COMPILED_LOOPS']
            bridges = c['TOTAL_COMPILED_BRIDGES']
        except Exception:
            pass
    return (loops, bridges, _metatensor.launch_count(),
            _metatensor.kernel_count())


# --- (a) shared live intermediate, both sides of a data-dependent branch ----

def case_shared_branch():
    x = tp.asarray(base(N))
    b = tp.asarray([0.5] * N)
    total = 0.0
    taken = [0, 0]
    for i in range(ITERS):
        tick()
        s = _metatensor.scalar(float(1 + (i % 4)), 'float64')
        h = (x * b + b).relu() * s          # live, unforced, used twice below
        v = h.sum().item()
        if v > 100.0:                       # crosses in both directions
            out = h + b
            taken[0] += 1
        else:
            out = h * b
            taken[1] += 1
        total += out.sum().item()
    tick()
    return g(total), "true=%d false=%d" % (taken[0], taken[1])


def ref_shared_branch():
    x = base(N)
    total = 0.0
    taken = [0, 0]
    for i in range(ITERS):
        s = float(1 + (i % 4))
        h = [max(0.0, xi * 0.5 + 0.5) * s for xi in x]
        v = sum(h)
        if v > 100.0:
            out = [hi + 0.5 for hi in h]
            taken[0] += 1
        else:
            out = [hi * 0.5 for hi in h]
            taken[1] += 1
        total += sum(out)
    return g(total), "true=%d false=%d" % (taken[0], taken[1])


# --- (b) exception from surrounding Python inside the fused region ---------

class ProbeError(Exception):
    pass


def maybe_raise(i):
    if i >= HALF and i % 17 == 0:
        raise ProbeError(i)
    return i


def case_exception():
    x = tp.asarray(base(N))
    b = tp.asarray([0.5] * N)
    total = 0.0
    caught = 0
    for i in range(ITERS):
        h = (x * b + b).relu()              # live and unforced across the raise
        try:
            maybe_raise(i)
            h = h + b
        except ProbeError:
            caught += 1
        total += h.sum().item()             # partial chain must still be right
    return g(total), "caught=%d" % caught


def ref_exception():
    x = base(N)
    total = 0.0
    caught = 0
    for i in range(ITERS):
        h = [max(0.0, xi * 0.5 + 0.5) for xi in x]
        try:
            maybe_raise(i)
            h = [hi + 0.5 for hi in h]
        except ProbeError:
            caught += 1
        total += sum(h)
    return g(total), "caught=%d" % caught


# --- (c) observable side effects: order and count --------------------------

def case_effects():
    x = tp.asarray(base(N))
    b = tp.asarray([0.5] * N)
    log = []
    n = [0]
    for i in range(ITERS):
        h = (x * b + b).relu() * _metatensor.scalar(float(1 + (i % 2)),
                                                    'float64')
        if i % 3 == 0:
            log.append(i)
            n[0] += 1
        if h.sum().item() > 63.0:
            log.append(-i)
            n[0] += 1
    return "n=%d" % n[0], ",".join([str(e) for e in log])


def ref_effects():
    x = base(N)
    log = []
    n = [0]
    for i in range(ITERS):
        h = [max(0.0, xi * 0.5 + 0.5) * float(1 + (i % 2)) for xi in x]
        if i % 3 == 0:
            log.append(i)
            n[0] += 1
        if sum(h) > 63.0:
            log.append(-i)
            n[0] += 1
    return "n=%d" % n[0], ",".join([str(e) for e in log])


# --- (d) in-place mutation read through an alias ---------------------------

def case_mutation():
    a = tp.asarray([1.0] * N)
    alias = a.detach()                      # shares the same underlying tensor
    b = tp.asarray([0.5] * N)
    reads = []
    for i in range(ITERS):
        a.add_(b)                           # ops.assign -> tensor_assign
        reads.append(alias.sum().item())    # alias read after the write
    return g(reads[-1]), "first=%s mid=%s" % (g(reads[0]), g(reads[HALF]))


def ref_mutation():
    a = [1.0] * N
    reads = []
    for i in range(ITERS):
        a = [ai + 0.5 for ai in a]
        reads.append(sum(a))
    return g(reads[-1]), "first=%s mid=%s" % (g(reads[0]), g(reads[HALF]))


def case_mut_hazard():
    a = tp.asarray([1.0] * N)
    b = tp.asarray([0.5] * N)
    reads = []
    for i in range(ITERS):
        y = a * b                           # deferred, not forced yet
        a.add_(b)                           # in-place write to y's input
        reads.append(y.sum().item())        # must see a as it was above
    return g(sum(reads)), "first=%s last=%s" % (g(reads[0]), g(reads[-1]))


def ref_mut_hazard():
    a = [1.0] * N
    reads = []
    for i in range(ITERS):
        y = [ai * 0.5 for ai in a]
        a = [ai + 0.5 for ai in a]
        reads.append(sum(y))
    return g(sum(reads)), "first=%s last=%s" % (g(reads[0]), g(reads[-1]))


# --- (e) int/shape promotion change forcing a bridge -----------------------

def case_promotion():
    xs = {N: tp.asarray(base(N)), N + 32: tp.asarray(base(N + 32))}
    bs = {N: tp.asarray([0.5] * N), N + 32: tp.asarray([0.5] * (N + 32))}
    total = 0.0
    for i in range(ITERS):
        tick()
        n = N if i < HALF else N + 32       # shape changes halfway
        k = 1 if i < HALF else 2            # Python int changes halfway
        h = (xs[n] * bs[n] + bs[n]).relu()
        total += h.sum().item() * k
    tick()
    return g(total), "shapes=%d,%d" % (N, N + 32)


def ref_promotion():
    total = 0.0
    for i in range(ITERS):
        n = N if i < HALF else N + 32
        k = 1 if i < HALF else 2
        h = [max(0.0, xi * 0.5 + 0.5) for xi in base(n)]
        total += sum(h) * k
    return g(total), "shapes=%d,%d" % (N, N + 32)


CASES = [
    ('a shared-branch', case_shared_branch, ref_shared_branch),
    ('b exception', case_exception, ref_exception),
    ('c effects', case_effects, ref_effects),
    ('d1 mut-alias', case_mutation, ref_mutation),
    ('d2 mut-hazard', case_mut_hazard, ref_mut_hazard),
    ('e promotion', case_promotion, ref_promotion),
]


def _median(xs):
    xs = sorted(xs)
    n = len(xs)
    if not n:
        return -1.0
    if n % 2:
        return xs[n // 2]
    return (xs[n // 2 - 1] + xs[n // 2]) / 2.0


def timed_cases():
    """One `deopt-probe` line per case whose guards really fail.

    (e) fails at a known iteration (HALF), so its three numbers are exact.
    (a) flips its branch every few iterations from the start, so there is no
    single first failure to point at; peak_i is where the JIT paid for the
    other side and steady/revisit is everything after it."""
    global TIMES
    out = []
    for name, fn, fail_at in (('a', case_shared_branch, -1),
                              ('e', case_promotion, HALF)):
        TIMES = []
        c0 = counters()
        fn()
        c1 = counters()
        stamps = TIMES
        TIMES = None
        us = [(stamps[i + 1] - stamps[i]) * 1e6
              for i in range(len(stamps) - 1)]
        # iteration 0 is the cold trace; the peak after it is the compile.
        peak = max(us[1:])
        peak_i = us.index(peak)
        first = us[fail_at] if fail_at >= 0 else -1.0
        after = _median(us[fail_at + 1:fail_at + 6]) if fail_at >= 0 else \
            _median(us[peak_i + 1:peak_i + 6])
        out.append("deopt-probe case=%s steady_us=%.1f first_fail_us=%.1f "
                   "after_fail_us=%.1f revisit_us=%.1f peak_us=%.1f "
                   "peak_i=%d cold_us=%.1f launches_per_iter=%.3f "
                   "loops=%d bridges=%d"
                   % (name, _median(us[len(us) * 3 // 4:]), first, after,
                      _median(us[len(us) * 3 // 4:]), peak, peak_i, us[0],
                      float(c1[2] - c0[2]) / ITERS, c1[0] - c0[0],
                      c1[1] - c0[1]))
    return out


def main(argv):
    argv = [a for a in argv if a != '--nostats']
    times = '--times' in argv
    argv = [a for a in argv if a != '--times']
    save = ref = None
    if len(argv) >= 3 and argv[1] == '--save':
        save = argv[2]
    elif len(argv) >= 3 and argv[1] == '--ref':
        ref = argv[2]

    # The timing pass has to be the first thing that runs these two loops:
    # the compile it is measuring happens once per process.
    if times:
        for line in timed_cases():
            print(line)
        return 0

    reference = {}
    if ref is not None:
        for line in open(ref):
            parts = line.rstrip("\n").split("\t")
            if len(parts) == 3:
                reference[parts[0]] = (parts[1], parts[2])

    rows = []
    saved = []
    bad = []
    for name, run, refrun in CASES:
        expected = refrun()
        c0 = counters()
        observed = run()
        c1 = counters()
        loops = c1[0] - c0[0]
        bridges = c1[1] - c0[1]
        launches = float(c1[2] - c0[2]) / ITERS
        kernels = c1[3] - c0[3]
        ok = observed == expected
        if not ok:
            bad.append("%s: expected %r observed %r" % (name, expected,
                                                        observed))
        refok = '-'
        if ref is not None:
            want = reference.get(name)
            if want is None:
                refok = 'missing'
                bad.append("%s: no reference row" % name)
            elif want == observed:
                refok = 'yes'
            else:
                refok = 'NO'
                bad.append("%s: reference %r observed %r" % (name, want,
                                                             observed))
        saved.append("%s\t%s\t%s" % (name, observed[0], observed[1]))
        shown = observed[1]
        if len(shown) > 30:
            shown = "%s... (%d chars)" % (shown[:30], len(observed[1]))
        rows.append((name, ok, refok, expected[0], observed[0], shown,
                     launches, kernels, loops, bridges))

    if save is not None:
        f = open(save, 'w')
        f.write("\n".join(saved) + "\n")
        f.close()

    print("%-16s %-5s %-5s %-14s %-14s %8s %7s %5s %7s  %s" % (
        'case', 'ok', 'vs-ref', 'expected', 'observed', 'launch/it',
        'kernels', 'loops', 'bridges', 'effects'))
    for r in rows:
        print("%-16s %-5s %-5s %-14s %-14s %8.3f %7d %5d %7d  %s" % (
            r[0], 'yes' if r[1] else 'NO', r[2], r[3], r[4], r[6], r[7],
            r[8], r[9], r[5]))
    if bad:
        print("")
        for line in bad:
            print("FAIL %s" % line)
        return 1
    print("\nall cases match")
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv))
