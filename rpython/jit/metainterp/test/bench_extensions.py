"""Combined microbenchmark for Ideas 1A/1B/1D/3A.

Usage:
    pypy rpython/jit/metainterp/test/bench_extensions.py [N]

Each block is a microbench of one optimization in isolation. Numbers
are *meta-level* proxies, not end-to-end JIT wall-clock. They let us
answer "how much does the mechanism itself cost or save" without a
full translate.

Baseline anchor (from Bolz 2025-06 blog, 1-iter Python bench):
    meta-interpreter ops         3675
    trace ops recorded           11200
    ops after optimization         22
    dispatch-overhead share      42.8%
We reuse this as the mental model for projecting real savings.
"""

from __future__ import print_function

import os
import sys
import time


here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(here)))))


from rpython.jit.metainterp import lazy_record
from rpython.jit.metainterp import snapshot_delta
from rpython.jit.metainterp import dispatch_profile
from rpython.jit.metainterp import edit_aware_cache
from rpython.jit.metainterp import super_jitcode
from rpython.jit.metainterp import bta


N = int(sys.argv[1]) if len(sys.argv) > 1 else 2000


# ---------------------------------------------------------------------------
# 1A. Lazy op recording vs eager
# ---------------------------------------------------------------------------

class _ResOperationStub(object):
    """Represents the work a real ``history.record`` does per op:
    allocate a ResOperation, populate its fields, append to trace, and
    update the heapcache / opcode counters. We proxy that with a fixed
    amount of attribute assignment so the microbench reflects the
    realistic ratio between "create IR node" (expensive) and "decide
    if we need to create it" (cheap).
    """
    __slots__ = ('opname', 'args', 'result', 'pc',
                 'flag_effect', 'flag_pure', 'next_')

    def __init__(self, opname, args):
        self.opname = opname
        self.args = list(args)
        self.result = None
        self.pc = -1
        self.flag_effect = False
        self.flag_pure = True
        self.next_ = None


def bench_lazy_vs_eager(n_traces=200, ops_per_trace=100,
                        survivor_every=50):
    """Simulate ops_per_trace ops per trace; about `ops_per_trace /
    survivor_every` actually escape. Compare:

      * eager: every op produces a ResOperation-shaped record
      * lazy : record into VirtualOp, materialize the escaped subset
               into ResOperation-shaped records at trace end

    The stub record is heavier than a tuple -- that's the point: a
    real ResOperation is not a two-tuple, and the gap matters.
    """

    def eager_emit(opname, args):
        return _ResOperationStub(opname, args)

    # Eager
    t0 = time.time()
    for _ in range(n_traces):
        trace = []
        for i in range(ops_per_trace):
            op = eager_emit('int_add', [i, 1])
            trace.append(op)
    eager_t = time.time() - t0

    # Lazy
    t0 = time.time()
    total_kept = 0
    for _ in range(n_traces):
        r = lazy_record.LazyRecorder()
        survivors = []
        for i in range(ops_per_trace):
            v = r.record('int_add', [i, 1])
            if i % survivor_every == 0:
                survivors.append(v)
        r.record_loop_tail(survivors)
        # Emit real stubs for the escaped subset.
        real = []

        def materialize_emit(opname, args):
            op = _ResOperationStub(opname, args)
            real.append(op)
            return op
        r.materialize(materialize_emit)
        total_kept += len(real)
    lazy_t = time.time() - t0

    total_ops = n_traces * ops_per_trace
    avg_kept = total_kept / float(n_traces)
    return {
        'n_traces': n_traces,
        'ops_per_trace': ops_per_trace,
        'total_ops': total_ops,
        'eager_ms': eager_t * 1000,
        'lazy_ms':  lazy_t * 1000,
        'speedup_x': eager_t / lazy_t if lazy_t else float('inf'),
        'avg_kept_per_trace': avg_kept,
        'kept_ratio': avg_kept / ops_per_trace,
    }


# ---------------------------------------------------------------------------
# 1B. Chain fusion (pair vs triple vs unfused)
# ---------------------------------------------------------------------------

def count_chain_fusion(n_ops=100000, hit_ratio=1.0):
    """Count outer-loop iterations for each mode. In translated PyPy,
    each outer iteration maps to a branch-target transition in the
    dispatcher (opcode_implementations[op] is an indirect call through
    a computed address, the #1 source of branch-predictor misses).
    Inner fused steps are straight-line. So outer iterations == the
    real dispatch-cycle metric; wall-clock in interpreted Python is
    dominated by Python overhead and misleads.

    hit_ratio = fraction of positions where a fuseable pair is available.
    1.0 = every pair fuses (best case); 0.5 = half fuse; 0.0 = none.
    """
    import random
    random.seed(42)
    N_OP = 4
    bc = []
    for _ in range(n_ops):
        bc.append(random.randint(0, N_OP - 1))

    def build_table(pair_fn):
        t = super_jitcode.empty_table(N_OP)
        for a in range(N_OP):
            for b in range(N_OP):
                if pair_fn(a, b):
                    t[a * N_OP + b] = b
        return t

    def always(a, b): return True
    def never(a, b): return False
    def half(a, b): return (a + b) & 1 == 0

    if hit_ratio >= 0.99:
        pair_fn = always
    elif hit_ratio <= 0.01:
        pair_fn = never
    else:
        pair_fn = half
    table = build_table(pair_fn)

    # Unfused outer loop count is just n_ops.
    unfused_outer = n_ops

    # Pair-fused: every successful fusion shaves one outer iteration.
    pair_outer = 0
    pc = 0
    while pc < n_ops:
        pair_outer += 1
        op = bc[pc]; pc += 1
        if pc < n_ops:
            nxt = bc[pc]
            if table[op * N_OP + nxt] >= 0:
                pc += 1  # fused

    # Chain-fused: iterative fusion up to SUPER_MAX_CHAIN per outer iter.
    chain_outer = 0
    pc = 0
    MAX_CHAIN = 8
    while pc < n_ops:
        chain_outer += 1
        op = bc[pc]; pc += 1
        chain = 0
        cur_op = op
        while chain < MAX_CHAIN and pc < n_ops:
            nxt = bc[pc]
            if table[cur_op * N_OP + nxt] < 0:
                break
            pc += 1
            cur_op = nxt
            chain += 1
    return {
        'n_ops': n_ops,
        'hit_ratio_target': hit_ratio,
        'unfused_outer': unfused_outer,
        'pair_outer': pair_outer,
        'chain_outer': chain_outer,
        'pair_reduction_pct':  100.0 * (1 - pair_outer  / float(unfused_outer)),
        'chain_reduction_pct': 100.0 * (1 - chain_outer / float(unfused_outer)),
    }


def bench_chain_fusion(n_ops=100000, impl_work=16):
    """Simulate a tight dispatch loop. Two distinct ops (0, 1) cycling.
    Each op impl does a small amount of work (a tight arithmetic loop
    of ``impl_work`` iterations) to model that in the real metainterp
    each opcode impl *does something*, so dispatch overhead is
    measured relative to actual work, not relative to zero.

    Compares:
      unfused    - outer-loop work every iteration (op_live/op_goto
                   compare pair, counter update, bytecode re-read)
      pair-fused - pair (0,1) fuses one step, outer loop still runs
                   per-pair
      chain-fused - chain of length up to SUPER_MAX_CHAIN fuses greedily
                    per outer iteration
    """
    bc = [i & 1 for i in range(n_ops)]
    N_OP = 2

    def mk_impl(tag):
        def impl(pc):
            # Simulate real per-op work (heapcache update etc).
            acc = 0
            for _ in range(impl_work):
                acc += 1
            return pc + 1
        impl.tag = tag
        return impl
    impls = [mk_impl('op0'), mk_impl('op1')]

    # Simulate outer-loop plumbing (op_live / op_goto compares, counter
    # update, bytecode[pc] re-read). In the translated build these are
    # all inlined, so the microbench puts them in Python to keep the
    # *ratio* meaningful on CPython/PyPy2.
    counters = [0, 0]
    OP_LIVE, OP_GOTO = -1, -2  # won't match in our synthetic bc

    # -- unfused baseline --
    t0 = time.time()
    pc = 0
    while pc < n_ops:
        op = bc[pc]
        if op == OP_LIVE:
            pc += 1; continue
        elif op == OP_GOTO:
            pc += 1; continue
        counters[op] += 1
        impls[op](pc)
        pc += 1
    unfused_t = time.time() - t0

    # -- pair-fused --
    pair_table = super_jitcode.empty_table(N_OP)
    pair_table[0 * N_OP + 1] = 1
    pair_table[1 * N_OP + 0] = 0
    counters = [0, 0]
    t0 = time.time()
    pc = 0
    while pc < n_ops:
        op = bc[pc]
        if op == OP_LIVE:
            pc += 1; continue
        elif op == OP_GOTO:
            pc += 1; continue
        counters[op] += 1
        impls[op](pc)
        pc += 1
        if pc < n_ops:
            nxt = bc[pc]
            slot = pair_table[op * N_OP + nxt]
            if slot >= 0:
                counters[slot] += 1
                impls[slot](pc)
                pc += 1
    pair_t = time.time() - t0

    # -- chain-fused --
    chain_table = super_jitcode.empty_table(N_OP)
    chain_table[0 * N_OP + 1] = 1
    chain_table[1 * N_OP + 0] = 0
    MAX_CHAIN = 8
    counters = [0, 0]
    t0 = time.time()
    pc = 0
    while pc < n_ops:
        op = bc[pc]
        if op == OP_LIVE:
            pc += 1; continue
        elif op == OP_GOTO:
            pc += 1; continue
        counters[op] += 1
        impls[op](pc)
        pc += 1
        chain = 0
        cur_op = op
        while chain < MAX_CHAIN and pc < n_ops:
            nxt = bc[pc]
            slot = chain_table[cur_op * N_OP + nxt]
            if slot < 0:
                break
            counters[slot] += 1
            impls[slot](pc)
            pc += 1
            cur_op = nxt
            chain += 1
    chain_t = time.time() - t0

    return {
        'n_ops': n_ops,
        'impl_work': impl_work,
        'unfused_ms': unfused_t * 1000,
        'pair_ms':    pair_t * 1000,
        'chain_ms':   chain_t * 1000,
        'pair_speedup_x':  unfused_t / pair_t  if pair_t  else float('inf'),
        'chain_speedup_x': unfused_t / chain_t if chain_t else float('inf'),
    }


# ---------------------------------------------------------------------------
# 1D. Snapshot delta vs full-snapshot storage cost
# ---------------------------------------------------------------------------

def bench_snapshot_delta(n_snapshots=500, frame_size=20,
                         changed_per_step=1):
    """Emit n snapshots; each differs from the previous in
    `changed_per_step` slots. Compare per-snapshot byte cost of the
    delta-aware store vs an all-full store.
    """
    # Delta store
    store = snapshot_delta.SnapshotStore()
    base_vec = list(range(frame_size))
    prev = store.emit_full(base_vec)
    t0 = time.time()
    for i in range(n_snapshots - 1):
        nv = list(prev.values) if prev.kind == snapshot_delta.Snapshot.KIND_FULL else store.resolve(prev)
        for k in range(changed_per_step):
            nv[k] = i * frame_size + k
        prev = store.emit_delta(prev, nv)
    delta_t = time.time() - t0
    total_delta, all_full, savings = store.size_report()

    # All-full baseline
    store2 = snapshot_delta.SnapshotStore()
    t0 = time.time()
    for i in range(n_snapshots):
        store2.emit_full(base_vec)
    full_t = time.time() - t0
    total_full, _, _ = store2.size_report()

    return {
        'n_snapshots': n_snapshots,
        'frame_size': frame_size,
        'changed_per_step': changed_per_step,
        'delta_ms': delta_t * 1000,
        'full_ms':  full_t * 1000,
        'delta_bytes': total_delta,
        'full_bytes':  total_full,
        'bytes_saved_pct': 100.0 * (1 - total_delta / float(all_full))
                             if all_full else 0.0,
    }


# ---------------------------------------------------------------------------
# 3A. Edit-aware cache validation cost (hit / miss)
# ---------------------------------------------------------------------------

def bench_edit_aware(n_entries=500):
    """Measure the cost of validate_source_fingerprint for a cache of
    n_entries against (a) unchanged source (all hits) and (b) edited
    source (all misses). Dominated by one os.stat + hash-table lookup.
    """
    import tempfile
    tmp = tempfile.mkdtemp()
    paths = []
    for i in range(n_entries):
        p = os.path.join(tmp, 'm%03d.py' % i)
        with open(p, 'w') as f:
            f.write('def f():\n    return %d\n' % i)
        paths.append(p)
    edit_aware_cache.reset_index()
    metas = [edit_aware_cache.build_source_meta(p, 'f') for p in paths]

    t0 = time.time()
    for m in metas:
        assert edit_aware_cache.validate_source_fingerprint(m)
    hit_t = time.time() - t0

    # Edit each file so validation should miss.
    for p in paths:
        with open(p, 'a') as f:
            f.write('# edited\n')
        os.utime(p, (os.stat(p).st_mtime + 1, os.stat(p).st_mtime + 1))
    edit_aware_cache.reset_index()

    t0 = time.time()
    misses = 0
    for m in metas:
        if not edit_aware_cache.validate_source_fingerprint(m):
            misses += 1
    miss_t = time.time() - t0

    # cleanup
    import shutil
    shutil.rmtree(tmp, ignore_errors=True)

    return {
        'n_entries': n_entries,
        'hit_ms': hit_t * 1000,
        'miss_ms': miss_t * 1000,
        'hit_us_per_entry':  hit_t  * 1e6 / n_entries,
        'miss_us_per_entry': miss_t * 1e6 / n_entries,
        'miss_rate': misses / float(n_entries),
    }


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def _hdr(s):
    print('\n' + '=' * 70)
    print(s)
    print('=' * 70)


# ---------------------------------------------------------------------------
# BTA. Binding-time analysis speed + static-ratio
# ---------------------------------------------------------------------------

class _FakePyCode(object):
    def __init__(self, code, nlocals=8, argcount=0, name='test'):
        self.co_code = code
        self.co_nlocals = nlocals
        self.co_argcount = argcount
        self.co_name = name


def _make_bench_bytecode(n_ops=100):
    """Build a bytecode sequence: repeated LOAD_CONST + BINARY_ADD."""
    parts = []
    parts.append(chr(bta.LOAD_CONST) + '\x01\x00')  # const 1
    for _ in range(n_ops):
        parts.append(chr(bta.LOAD_CONST) + '\x02\x00')  # const 2
        parts.append(chr(bta.BINARY_ADD) + '\x00\x00')
    parts.append(chr(bta.RETURN_VALUE))
    return ''.join(parts)


def bench_bta(n_codes=1000, ops_per_code=100):
    code = _make_bench_bytecode(ops_per_code)
    pycode = _FakePyCode(code, nlocals=0)
    t0 = time.time()
    for _ in range(n_codes):
        info = bta.analyse_pycode(pycode)
    elapsed = time.time() - t0
    info = bta.analyse_pycode(pycode)
    static_count = sum(1 for bt in info.offset_bt if bt == bta.BT_STATIC)
    return {
        'n_codes': n_codes,
        'ops_per_code': ops_per_code,
        'total_ms': elapsed * 1000,
        'static_count': static_count,
        'total_offsets': len(info.offset_bt),
        'static_pct': 100.0 * static_count / len(info.offset_bt) if info.offset_bt else 0.0,
    }


def main():
    _hdr('1A. Lazy op recording vs eager (%d traces)' % N)
    r = bench_lazy_vs_eager(n_traces=N, ops_per_trace=100,
                            survivor_every=50)
    print('total ops processed: %d' % r['total_ops'])
    print('eager recording:   %7.1f ms' % r['eager_ms'])
    print('lazy recording:    %7.1f ms  (speedup %.2fx)'
          % (r['lazy_ms'], r['speedup_x']))
    print('avg kept per trace: %.1f / %d (%.1f%%)'
          % (r['avg_kept_per_trace'], r['ops_per_trace'],
             100 * r['kept_ratio']))

    _hdr('BTA. Binding-time analysis (Python-bytecode level)')
    r = bench_bta(n_codes=500, ops_per_code=100)
    print('codes analysed: %d  (ops/code=%d)' % (r['n_codes'], r['ops_per_code']))
    print('analysis time:  %7.1f ms  (%.1f us/code)' %
          (r['total_ms'], r['total_ms'] * 1000 / r['n_codes']))
    print('static slots:   %d / %d offsets (%.1f%%)' %
          (r['static_count'], r['total_offsets'], r['static_pct']))

    _hdr('1B. Chain fusion (dispatch-cycle count -- primary metric)')
    print('Outer-loop iterations per 100k ops (== branch-target '
          'transitions in translated PyPy):')
    print('%-14s %12s %12s %12s' %
          ('hit_ratio', 'unfused', 'pair', 'chain'))
    for hr in (1.0, 0.5, 0.0):
        r = count_chain_fusion(n_ops=100000, hit_ratio=hr)
        print('%-14.2f %12d %12d %12d    pair -%.1f%%  chain -%.1f%%'
              % (hr, r['unfused_outer'], r['pair_outer'], r['chain_outer'],
                 r['pair_reduction_pct'], r['chain_reduction_pct']))
    print()
    print('Secondary metric (wall-clock, impl_work=16):')
    r = bench_chain_fusion(n_ops=200000, impl_work=16)
    print('unfused:     %7.1f ms' % r['unfused_ms'])
    print('pair-fused:  %7.1f ms  (ratio %.2fx)'
          % (r['pair_ms'], r['pair_speedup_x']))
    print('chain-fused: %7.1f ms  (ratio %.2fx)'
          % (r['chain_ms'], r['chain_speedup_x']))
    print('(ratio < 1 is expected under CPython/PyPy2-python: the extra '
          'Python-level overhead of the fused-path loop outweighs the '
          'dispatch-cycle savings until we are actually translated.)')

    _hdr('1D. Snapshot delta vs full')
    for changed in (1, 2, 4):
        r = bench_snapshot_delta(n_snapshots=500, frame_size=20,
                                 changed_per_step=changed)
        print('frame=20, changed=%d:' % changed)
        print('  delta emit: %6.1f ms  %d bytes' %
              (r['delta_ms'], r['delta_bytes']))
        print('  full  emit: %6.1f ms  %d bytes' %
              (r['full_ms'], r['full_bytes']))
        print('  bytes saved vs all-full: %.1f%%' % r['bytes_saved_pct'])

    _hdr('3A. Edit-aware cache validation')
    r = bench_edit_aware(n_entries=500)
    print('hits:  %6.1f ms total, %.1f us/entry' %
          (r['hit_ms'], r['hit_us_per_entry']))
    print('miss:  %6.1f ms total, %.1f us/entry (miss rate %.0f%%)' %
          (r['miss_ms'], r['miss_us_per_entry'], 100 * r['miss_rate']))

    _hdr('Aggregate projection (against Bolz 2025-06 baseline)')
    # 1A -- fraction of recorded ops we avoid emitting. Apply measured
    # escape ratio to the 11200 recorded / 22 kept baseline. The wall-
    # clock speedup in the microbench depends on how expensive the
    # eager emit actually is; we report both.
    r1a = bench_lazy_vs_eager(n_traces=500, ops_per_trace=100,
                              survivor_every=50)
    avoid_ratio = 1 - r1a['kept_ratio']
    saved_ops = 11200 * avoid_ratio
    print('1A  avoided op emissions: %.0f / 11200 (%.0f%% of recorded ops)'
          % (saved_ops, avoid_ratio * 100))
    print('    wall-clock ratio (stubbed ResOp cost): %.2fx  '
          '(eager %.1fms vs lazy %.1fms)'
          % (r1a['speedup_x'], r1a['eager_ms'], r1a['lazy_ms']))
    # 1B -- dispatch cycle count. Use the translation-independent proxy
    # (outer-loop iterations) at representative hit rates; 50% is a
    # rough Pareto estimate for the realistic workload.
    r1b_full = count_chain_fusion(n_ops=100000, hit_ratio=1.0)
    r1b_half = count_chain_fusion(n_ops=100000, hit_ratio=0.5)
    print('1B  pair-fusion   dispatch cycles -%.1f%% (best)  -%.1f%% (half-hit)'
          % (r1b_full['pair_reduction_pct'], r1b_half['pair_reduction_pct']))
    print('    chain-fusion  dispatch cycles -%.1f%% (best)  -%.1f%% (half-hit)'
          % (r1b_full['chain_reduction_pct'], r1b_half['chain_reduction_pct']))
    # Pick the middling hit rate for the projection; conservative.
    cut = r1b_half['chain_reduction_pct'] / 100.0
    print('    -> projected tracing-time cut = 42.8%% share * %.1f%% '
          '= %.1f%% (half-hit estimate)'
          % (cut * 100, 0.428 * cut * 100))
    # BTA -- Python-bytecode static ratio.
    rbta = bench_bta(n_codes=500, ops_per_code=100)
    print('BTA static slots: %.1f%% of bytecode offsets (analysis %.1f us/code)'
          % (rbta['static_pct'], rbta['total_ms'] * 1000 / rbta['n_codes']))
    # 1D -- resume data size impact (wire bytes, not wall clock).
    r1d = bench_snapshot_delta(n_snapshots=500, frame_size=20,
                               changed_per_step=1)
    print('1D  resume data size saved: %.0f%% on typical 1-slot-change pattern'
          % r1d['bytes_saved_pct'])
    # 3A -- validation cost per entry.
    r3a = bench_edit_aware(n_entries=500)
    print('3A  validation cost: %.1f us/hit, %.1f us/miss '
          '(hit = cached os.stat + dict lookup; miss = re-parse file)'
          % (r3a['hit_us_per_entry'], r3a['miss_us_per_entry']))


if __name__ == '__main__':
    main()
