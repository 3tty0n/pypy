"""Microbenchmark for Idea 3 (persistent trace cache).

Simulates the cost of recomputing a trace's fingerprint + storing/loading
against a mocked optimizer pipeline. Not a replacement for a real trace
workload, but it reports the per-entry overhead so we can sanity-check
the claim that cache revalidation is cheap relative to tracing+optimizing.

Run:
    PYTHONPATH=. pypy rpython/jit/metainterp/test/bench_tracecache.py

Env knobs:
    BENCH_N       number of distinct keys (default 500)
    BENCH_OPS     operations per synthetic entry (default 100)
    BENCH_DIR     cache directory (default: tmp dir, deleted at exit)
"""

from __future__ import print_function

import os
import shutil
import sys
import tempfile
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))))

from rpython.jit.metainterp import tracecache


class FakeConst(object):
    def __init__(self, type_, value):
        self.type = type_
        self.value = value


class FakeBox(object):
    def __init__(self, type_='i', is_const=False, value=0):
        self.type = type_
        self._is_const = is_const
        self._value = value

    def is_constant(self):
        return self._is_const

    def getint(self):
        return int(self._value)


class FakeOp(object):
    def __init__(self, opnum, args, type_='i'):
        self.opnum = opnum
        self.args = args
        self.type = type_

    def getopnum(self):   return self.opnum
    def getarglist(self): return list(self.args)
    def getdescr(self):   return None


def _make_entry(n_ops):
    inputs = [FakeBox('i') for _ in range(2)]
    builder = tracecache._Builder()
    ops = []
    prev = inputs[0]
    for i in range(n_ops):
        c = FakeBox('i', is_const=True, value=i)
        o = FakeOp(7, [prev, c], type_='i')
        ops.append(o)
        prev = o
    entry = builder.build(inputs, ops)
    entry.meta = {'num_ops': n_ops}
    return entry


def bench(n_keys, n_ops, cache_dir):
    tracecache.CACHE_DIR = cache_dir
    entries = []
    keys = []
    for i in range(n_keys):
        gk = [FakeConst('i', i), FakeConst('i', i * 2)]
        tp = {'x': 'int', 'y': 'float', 'seq': i}
        keys.append(tracecache.make_key(gk, tp))
        entries.append(_make_entry(n_ops))

    # Store pass
    t0 = time.time()
    for k, e in zip(keys, entries):
        tracecache.store(k, e, assumption_records=[])
    store_t = time.time() - t0

    # Warm reload (hit every entry once)
    t0 = time.time()
    hits = 0
    for k in keys:
        if tracecache.load(k) is not None:
            hits += 1
    load_t = time.time() - t0

    # Miss path (bogus keys)
    t0 = time.time()
    misses = 0
    for i in range(n_keys):
        if tracecache.load('deadbeef%032d' % i) is None:
            misses += 1
    miss_t = time.time() - t0

    print('N=%d ops_per_entry=%d' % (n_keys, n_ops))
    print('store: %.2fms total, %.1fus/entry' %
          (store_t * 1000, store_t * 1e6 / n_keys))
    print('load:  %.2fms total, %.1fus/entry (hits=%d)' %
          (load_t * 1000, load_t * 1e6 / n_keys, hits))
    print('miss:  %.2fms total, %.1fus/entry (misses=%d)' %
          (miss_t * 1000, miss_t * 1e6 / n_keys, misses))


def main():
    n_keys = int(os.environ.get('BENCH_N', '500'))
    n_ops = int(os.environ.get('BENCH_OPS', '100'))
    cache_dir = os.environ.get('BENCH_DIR') or tempfile.mkdtemp(
        prefix='pypy_tc_bench_')
    try:
        bench(n_keys, n_ops, cache_dir)
    finally:
        if not os.environ.get('BENCH_DIR'):
            shutil.rmtree(cache_dir, ignore_errors=True)


if __name__ == '__main__':
    main()
