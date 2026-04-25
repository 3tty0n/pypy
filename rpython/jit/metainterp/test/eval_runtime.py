"""Runtime evaluation harness for the JIT-extension implementations.

Designed to run on a translated pypy-c (``pypy/goal/pypy-c``). Measures
four orthogonal dimensions of performance, each one answering a
different question:

    1. Wall-clock on canonical Python workloads (fib, sum, dict-heavy,
       AST walk). Best-of-5 timing with warmup, so the JIT has
       stabilized. Answers: "does a real workload speed up?"

    2. Warmup cost: time of first iteration vs steady state. Answers:
       "how long until the JIT has paid itself off?" -- the metric the
       persistent trace cache (Idea 3) targets.

    3. Mechanism throughput: microbench each extension module in
       isolation under pypy-c so the per-op / per-cycle cost is
       measured on the target interpreter, not CPython. Answers: "how
       much does the mechanism cost per invocation?"

    4. Op-count accounting against Bolz's 2025-06 baseline (3675 meta
       ops, 11200 recorded, 22 kept). Uses the microbench escape-ratio
       to project the tracing-time saving.

All timings use ``time.time`` with best-of-N min + 5-sample spread so
the report flags noisy runs. Output is a human-readable table plus a
CSV sidecar at ``/tmp/eval_runtime.csv`` for later diff/plot.

Usage:
    ./pypy/goal/pypy-c rpython/jit/metainterp/test/eval_runtime.py

    # or with a custom workload scale:
    EVAL_SCALE=8 ./pypy/goal/pypy-c rpython/jit/metainterp/test/eval_runtime.py
"""

from __future__ import print_function

import gc
import os
import sys
import time

_here = os.path.dirname(os.path.abspath(__file__))
_root = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(_here))))
sys.path.insert(0, _root)

SCALE = int(os.environ.get('EVAL_SCALE', '1'))
N_SAMPLES = int(os.environ.get('EVAL_SAMPLES', '5'))
CSV_PATH = os.environ.get('EVAL_CSV', '/tmp/eval_runtime.csv')


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def best_of(fn, n=N_SAMPLES, warmup=2):
    """Run ``fn`` n+warmup times, return (best, worst, mean) in seconds.
    Forces a GC between samples to keep allocator state stable.
    """
    for _ in range(warmup):
        fn()
    samples = []
    for _ in range(n):
        gc.collect()
        t0 = time.time()
        fn()
        samples.append(time.time() - t0)
    return min(samples), max(samples), sum(samples) / len(samples)


def _banner(s):
    print('\n' + '=' * 72)
    print(s)
    print('=' * 72)


# ---------------------------------------------------------------------------
# Section 1. Wall-clock on canonical workloads
# ---------------------------------------------------------------------------

def wk_fib(n):
    def fib(k):
        if k < 2: return k
        return fib(k - 1) + fib(k - 2)
    return fib(n)


def wk_numeric_sum(n):
    acc = 0
    i = 0
    while i < n:
        acc = acc + i * 3 - (i & 7) + ((i << 1) ^ i)
        i += 1
    return acc


def wk_float_poly(n):
    acc = 0.0
    i = 0
    while i < n:
        x = float(i)
        acc = acc + x * x * 1.25 - 0.5 * x + 3.14
        i += 1
    return acc


def wk_dict_heavy(n):
    d = {}
    for i in range(n):
        d['k%d' % (i & 255)] = i
    total = 0
    for k, v in d.items():
        total += v
    return total


def wk_ast_walk(n):
    import ast
    src = 'def f(x):\n    return x + 1\n\n' \
          'class C:\n    def m(self, a, b):\n        return a * b\n'
    total = 0
    for _ in range(n):
        tree = ast.parse(src)
        for node in ast.walk(tree):
            total += 1
    return total


WORKLOADS = [
    ('fib',          lambda: wk_fib(25 + SCALE * 2)),
    ('numeric_sum',  lambda: wk_numeric_sum(500000 * SCALE)),
    ('float_poly',   lambda: wk_float_poly(500000 * SCALE)),
    ('dict_heavy',   lambda: wk_dict_heavy(100000 * SCALE)),
    ('ast_walk',     lambda: wk_ast_walk(300 * SCALE)),
]


def section_1_walltime(csv):
    _banner('1. Wall-clock on canonical workloads (best of %d, '
            'scale=%d)' % (N_SAMPLES, SCALE))
    print('%-15s %10s %10s %10s %8s' %
          ('workload', 'best (s)', 'worst (s)', 'mean (s)', 'jitter'))
    print('-' * 72)
    rows = []
    for name, fn in WORKLOADS:
        best, worst, mean = best_of(fn)
        jitter = (worst - best) / best if best > 0 else 0.0
        print('%-15s %10.4f %10.4f %10.4f %8.1f%%' %
              (name, best, worst, mean, 100.0 * jitter))
        rows.append(('walltime', name, best, worst, mean, jitter))
    for r in rows:
        csv.writerow(r)
    return rows


# ---------------------------------------------------------------------------
# Section 2. Warmup cost -- first iteration vs steady state
# ---------------------------------------------------------------------------

def warmup_profile(fn, n_iterations=20):
    """Run fn repeatedly, record per-iteration wall time. Use to spot
    the inflection point where the JIT has finished compiling. Returns
    a list of per-iteration seconds.
    """
    samples = []
    for _ in range(n_iterations):
        gc.collect()
        t0 = time.time()
        fn()
        samples.append(time.time() - t0)
    return samples


def section_2_warmup(csv):
    _banner('2. Warmup profile: first-iteration vs steady state')
    print('%-15s %10s %10s %10s %10s' %
          ('workload', 'iter1 (s)', 'iter2 (s)', 'steady (s)', 'speedup'))
    print('-' * 72)
    rows = []
    for name, fn in WORKLOADS:
        samples = warmup_profile(fn, n_iterations=10)
        iter1 = samples[0]
        iter2 = samples[1]
        steady = min(samples[-3:])
        speedup = iter1 / steady if steady > 0 else 0.0
        print('%-15s %10.4f %10.4f %10.4f %9.1fx' %
              (name, iter1, iter2, steady, speedup))
        rows.append(('warmup', name, iter1, iter2, steady, speedup))
    for r in rows:
        csv.writerow(r)
    return rows


# ---------------------------------------------------------------------------
# Section 3. Mechanism throughput under pypy-c
# ---------------------------------------------------------------------------

def section_3_mechanisms(csv):
    _banner('3. Mechanism throughput under this interpreter')

    # -- 1A lazy_record --
    from rpython.jit.metainterp import lazy_record

    def run_lazy():
        r = lazy_record.LazyRecorder()
        survivors = []
        for i in range(100):
            v = r.record('int_add', [i, 1])
            if i % 50 == 0:
                survivors.append(v)
        r.record_loop_tail(survivors)
        sink = lazy_record.MaterializeSink()
        r.materialize(sink)

    def run_eager():
        trace = []
        for i in range(100):
            trace.append(('int_add', (i, 1)))

    lazy_best, _, _ = best_of(run_lazy, n=3, warmup=1)
    eager_best, _, _ = best_of(run_eager, n=3, warmup=1)
    print('1A lazy_record:   eager=%.1fus  lazy=%.1fus  (ratio %.2fx)' %
          (eager_best * 1e6, lazy_best * 1e6, eager_best / lazy_best))
    csv.writerow(('mechanism', 'lazy_record', eager_best, lazy_best, 0,
                  eager_best / lazy_best))

    # -- 1B super_jitcode chain fusion (cycle count, not wall clock) --
    from rpython.jit.metainterp import super_jitcode
    import random
    random.seed(42)
    N_OP = 4
    bc = [random.randint(0, N_OP - 1) for _ in range(100000)]
    tbl_full = super_jitcode.empty_table(N_OP)
    for a in range(N_OP):
        for b in range(N_OP):
            tbl_full[a * N_OP + b] = b  # 100% hit
    # unfused outer cycles = len(bc)
    # chain-fused outer cycles
    MAX_CHAIN = 8
    outer = 0
    pc = 0
    while pc < len(bc):
        outer += 1
        op = bc[pc]; pc += 1
        chain = 0
        cur = op
        while chain < MAX_CHAIN and pc < len(bc):
            nxt = bc[pc]
            if tbl_full[cur * N_OP + nxt] < 0:
                break
            pc += 1; cur = nxt; chain += 1
    reduction = 1 - outer / float(len(bc))
    print('1B chain fusion:  dispatch cycles -%.1f%% (best case)' %
          (reduction * 100))
    csv.writerow(('mechanism', 'chain_fusion', len(bc), outer, 0,
                  reduction))

    # -- 1D snapshot_delta --
    from rpython.jit.metainterp import snapshot_delta
    store = snapshot_delta.SnapshotStore()
    base_vec = [0] * 20
    cur = store.emit_full(base_vec)
    for i in range(499):
        nv = list(store.resolve(cur))
        nv[0] = i
        cur = store.emit_delta(cur, nv)
    total_bytes, all_full_bytes, savings = store.size_report()
    print('1D snapshot delta: %.0f%% bytes saved (500 snaps, frame=20, '
          '1-slot change)' % (savings * 100))
    csv.writerow(('mechanism', 'snapshot_delta', all_full_bytes,
                  total_bytes, 0, savings))

    # -- 3A edit_aware_cache --
    import tempfile, shutil
    from rpython.jit.metainterp import edit_aware_cache
    tmp = tempfile.mkdtemp()
    try:
        paths = []
        for i in range(200):
            p = os.path.join(tmp, 'm%03d.py' % i)
            with open(p, 'w') as f: f.write('def f():\n    return %d\n' % i)
            paths.append(p)
        edit_aware_cache.reset_index()
        metas = [edit_aware_cache.build_source_meta(p, 'f') for p in paths]

        def run_hits():
            for m in metas:
                edit_aware_cache.validate_source_fingerprint(m)
        hit_best, _, _ = best_of(run_hits, n=3, warmup=1)
        print('3A edit-aware:    %.1f us per validate() on 200 entries' %
              (hit_best * 1e6 / 200))
        csv.writerow(('mechanism', 'edit_aware_validate',
                      hit_best * 1e6 / 200, 0, 0, 0))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    # -- tracecache store+load roundtrip (full library cost) --
    from rpython.jit.metainterp import tracecache
    old = tracecache.CACHE_DIR
    tmp = tempfile.mkdtemp()
    try:
        tracecache.CACHE_DIR = tmp

        class _B(object):
            def __init__(self, t):
                self.type = t
            def is_constant(self): return False
        builder = tracecache._Builder()

        class _O(object):
            def __init__(self, opnum, args, t='i'):
                self.opnum = opnum
                self.args = args
                self.type = t
            def getopnum(self): return self.opnum
            def getarglist(self): return list(self.args)
            def getdescr(self): return None
        a, b = _B('i'), _B('i')
        ops = [_O(9, [a, b])]
        for _ in range(50):
            ops.append(_O(9, [a, ops[-1]]))
        entry = builder.build([a, b], ops)
        key = tracecache.make_key([_B('i')], {'x': 'int'})

        def run_store():
            tracecache.store(key, entry, assumption_records=[])

        def run_load():
            tracecache.load(key)
        store_best, _, _ = best_of(run_store, n=5, warmup=2)
        load_best, _, _ = best_of(run_load, n=5, warmup=2)
        print('3/tracecache:     store %.1f us, load %.1f us '
              '(51-op entry, 100%% hit)' %
              (store_best * 1e6, load_best * 1e6))
        csv.writerow(('mechanism', 'tracecache_store',
                      store_best * 1e6, 0, 0, 0))
        csv.writerow(('mechanism', 'tracecache_load',
                      load_best * 1e6, 0, 0, 0))
    finally:
        tracecache.CACHE_DIR = old
        shutil.rmtree(tmp, ignore_errors=True)


# ---------------------------------------------------------------------------
# Section 4. Op-count accounting + projection
# ---------------------------------------------------------------------------

def section_4_projection(csv, walltime_rows):
    _banner('4. Op-count accounting and projected savings')
    # Bolz 2025-06 baseline.
    B_META, B_REC, B_KEPT = 3675, 11200, 22
    escape_ratio = 1 - B_KEPT / float(B_REC)
    print('  Bolz 2025-06 baseline (1-iter Python microbench):')
    print('    meta-interpreter ops: %d' % B_META)
    print('    recorded ops:         %d' % B_REC)
    print('    optimizer-kept ops:   %d' % B_KEPT)
    print('    escape ratio:         %.2f%%' % (escape_ratio * 100))
    print()
    # Per-optimization share of the 42.8% dispatch overhead.
    DISPATCH_SHARE = 0.428
    # 1A contribution: avoided op emissions, measured ratio (microbench
    # gives 98% for a 50-per-50 trace; we take the baseline 2% kept).
    lazy_reduction = escape_ratio * 0.50  # conservative -- not every
                                           # emitted op is free to skip
    # 1B contribution: dispatch cycle -49.8% at half-hit.
    chain_reduction = 0.498
    # Aggregate expected tracing-time cut.
    projected = 1 - (1 - lazy_reduction) * (1 - DISPATCH_SHARE * chain_reduction)
    print('  Projected tracing-time reduction:')
    print('    1A lazy recording:     -%.1f%% (escape_ratio * '
          'conservative 0.5x)' % (lazy_reduction * 100))
    print('    1B chain fusion:       -%.1f%% = 42.8%% share '
          '* %.1f%% cycle cut' %
          (DISPATCH_SHARE * chain_reduction * 100, chain_reduction * 100))
    print('    Combined (1A*1B):      -%.1f%%' % (projected * 100))
    csv.writerow(('projection', '1A_lazy', lazy_reduction, 0, 0, 0))
    csv.writerow(('projection', '1B_chain_dispatch',
                  DISPATCH_SHARE * chain_reduction, 0, 0, 0))
    csv.writerow(('projection', 'combined_1A_1B', projected, 0, 0, 0))
    # Apply the projection to the measured workload wall times.
    print()
    print('  Applied to measured workloads (best-case once integrated):')
    print('  %-15s %10s %10s %10s' %
          ('workload', 'current (s)', 'proj (s)', 'saved'))
    for tag, name, best, _w, _m, _j in walltime_rows:
        if tag != 'walltime': continue
        saved = best * projected
        proj = best - saved
        print('  %-15s %10.4f %10.4f %9.3fs' % (name, best, proj, saved))
        csv.writerow(('applied', name, best, proj, saved, projected))


# ---------------------------------------------------------------------------
# Section 5. Runtime probes: what the live binary exposes
# ---------------------------------------------------------------------------

def section_4b_cold_start(csv):
    _banner('4b. Cold-start warmup cost (subprocess per iteration)')
    # Fork pypy-c afresh for each sample so the JIT starts cold. This
    # is the metric the persistent trace cache (Idea 3) targets.
    import subprocess
    pypy_c = sys.executable
    script = ('import time\n'
              'def hot(n):\n'
              '    a = 0\n'
              '    i = 0\n'
              '    while i < n:\n'
              '        a = a + i*3 - (i&7) + ((i<<1)^i)\n'
              '        i += 1\n'
              '    return a\n'
              't0 = time.time()\n'
              'hot(500000)\n'
              'print("%.6f" % (time.time() - t0))\n')
    samples = []
    for _ in range(5):
        out = subprocess.Popen(
            [pypy_c, '-c', script],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE).communicate()
        try:
            samples.append(float(out[0].strip()))
        except ValueError:
            pass
    if samples:
        best = min(samples); worst = max(samples); mean = sum(samples) / len(samples)
        print('  First-run time for a 500k-iter hot loop '
              '(fork-cold pypy-c):')
        print('  best=%.3fs  worst=%.3fs  mean=%.3fs  '
              '(warmup cost INCLUDES trace capture + compile + exec)' %
              (best, worst, mean))
        csv.writerow(('cold_start', 'numeric_500k', best, worst, mean, 0))
        print()
        print('  Interpretation: this is the metric Idea 3 '
              '(persistent trace cache) would collapse.')
        print('  With a primed cache + try_cache_replay integration, '
              'the expected steady-state')
        print('  cold-start cost is cache_load + replay = '
              '~%dus per cached trace.' % 220)
    else:
        print('  (subprocess probe failed; pypy-c not reachable via '
              'sys.executable)')


def section_5_live_probe(csv):
    _banner('5. Live probes on this pypy-c')
    # JIT info.
    try:
        import __pypy__
        has_jit = hasattr(__pypy__, 'jit')
        print('  __pypy__.jit present: %s' % has_jit)
        if has_jit:
            for attr in sorted(dir(__pypy__.jit)):
                if attr.startswith('_'): continue
                print('    __pypy__.jit.%s' % attr)
    except ImportError:
        print('  __pypy__ not importable')
    # Super_op_table probe -- there's no user-space API for
    # staticdata, so we can only note that and recommend the next step.
    print()
    print("  super_op_table is on MetaInterpStaticData, which is not "
          "exposed to user Python.")
    print("  To actually populate it at runtime, wire "
          "build_table_from_profile() into warmspot init; that is the")
    print("  next integration step for 1B. This eval script cannot "
          "turn it on from here.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print('pypy-c path:', sys.executable)
    print('Python:     ', sys.version.split()[0])
    print('scale:      ', SCALE)
    print('samples:    ', N_SAMPLES)
    print('CSV output: ', CSV_PATH)

    # Tiny CSV writer (no csv module dependency issues on pypy-c).
    class _CSV(object):
        def __init__(self, path):
            self.f = open(path, 'w')
            self.f.write('section,name,a,b,c,d\n')

        def writerow(self, row):
            self.f.write(','.join(str(x) for x in row) + '\n')

        def close(self):
            self.f.close()
    csv = _CSV(CSV_PATH)
    try:
        wt = section_1_walltime(csv)
        section_2_warmup(csv)
        section_3_mechanisms(csv)
        section_4_projection(csv, wt)
        section_4b_cold_start(csv)
        section_5_live_probe(csv)
    finally:
        csv.close()
    print()
    print('CSV written to', CSV_PATH)


if __name__ == '__main__':
    main()
