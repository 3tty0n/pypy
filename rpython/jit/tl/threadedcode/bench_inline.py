#!/usr/bin/env python2
"""Compare TLA runtime across JIT configurations (e.g. threaded vs inlined)."""

import os
import subprocess
import sys
import json

THIS = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(THIS, '..', '..', '..', '..'))
LANG = os.path.join(THIS, 'lang')
CACHE = '/tmp/tla_bench'

DEFAULT_BIN = os.path.join(THIS, 'targettla-c')

CONFIGS = [
    ('tier0',     ['--tier', '0']),
    ('threaded',  ['--tier', '1', '--jit', 'inlining=0']),
    ('inline1',   ['--tier', '1', '--jit', 'inlining=1']),
]

PROGRAMS = [
    ('sum',           10,   5),
    ('sum-tail',      10,   5),
    ('sum-callasm',   10,   5),
    ('fib',           10,   5),
    ('fib-tail',      10,   5),
    ('fact',          10,   5),
    ('loop',          100,  5),
    ('loopabit',      100,  5),
    ('ack',           3,    5),
    ('ack-callasm',   3,    5),
    ('gcd',           1024, 5),
    ('sieve',         100,  5),
    ('square',        10,   5),
    ('ary',           100,  5),
    ('prefix_sum',    50,   5),
    ('tak',           7,    3),
    ('tarai',         7,    3),
]


def ensure_path():
    if ROOT not in sys.path:
        sys.path.insert(0, ROOT)


def assemble(name):
    if not os.path.isdir(CACHE):
        os.makedirs(CACHE)
    src = os.path.join(LANG, name + '.tla.py')
    dst = os.path.join(CACHE, name + '.tlc')
    if not os.path.exists(src):
        return None
    if os.path.exists(dst) and os.path.getmtime(dst) >= os.path.getmtime(src):
        return dst
    ensure_path()
    from rpython.jit.tl.threadedcode.bytecode import assemble as do_asm
    ns = {}
    execfile(src, ns)
    with open(dst, 'wb') as f:
        f.write(do_asm(ns['code']))
    return dst


def ground_truth(binary, prog, x):
    "Use the translated binary in --tier 0 as the authoritative reference."
    result, _times, raw = run_once(binary, ['--tier', '0'], prog, x, 1)
    if result is None:
        return '<gt-err:no-result>'
    return result


def run_once(binary, flags, prog, x, n, timeout=120):
    cmd = [binary] + flags + [prog, str(x), str(n)]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT,
                            env=dict(os.environ, PYPYLOG=''))
    out, _ = proc.communicate()
    lines = [l.strip() for l in out.splitlines() if l.strip()]
    times = []
    for s in lines[:n]:
        try:
            times.append(float(s))
        except ValueError:
            break
    result = lines[-1] if len(lines) > len(times) else None
    return result, times, out


def median(xs):
    if not xs:
        return float('nan')
    s = sorted(xs)
    m = len(s) // 2
    if len(s) % 2:
        return s[m]
    return 0.5 * (s[m - 1] + s[m])


def fmt(v):
    if v != v:
        return '       nan'
    if v == 0:
        return '         0'
    return '%10.6f' % v


def main(argv):
    binary = os.environ.get('TLA_BIN', DEFAULT_BIN)
    csv_path = None
    if '--csv' in argv:
        i = argv.index('--csv')
        csv_path = argv[i + 1]
    if not os.path.exists(binary):
        sys.stderr.write('missing binary: %s\n' % binary)
        return 1

    rows = []
    print '%-14s %-9s %-10s %-10s %-32s %-12s %s' % (
        'program', 'config', 'median', 'min', 'result', 'gt', 'speedup')
    for name, x, n in PROGRAMS:
        prog = assemble(name)
        if prog is None:
            continue
        gt = ground_truth(binary, prog, x)
        baseline_median = None
        for label, flags in CONFIGS:
            result, times, raw = run_once(binary, flags, prog, x, n)
            m = median(times)
            mn = min(times) if times else float('nan')
            status = 'ok' if result == gt else 'FAIL'
            if label == CONFIGS[0][0]:
                baseline_median = m
            speedup = (baseline_median / m) if (baseline_median and m) else float('nan')
            disp = str(result) if result and len(result) <= 32 else (
                '<crash>' if result and result != gt else str(result))
            ok = 'ok' if result == gt else 'FAIL'
            print '%-14s %-9s %s %s %-32s %-12s x%.2f %s' % (
                name, label, fmt(m), fmt(mn), disp, gt, speedup, ok)
            rows.append({
                'program': name, 'x': x, 'iters': n,
                'config': label, 'flags': flags,
                'result': result, 'gt': gt, 'status': status,
                'median': m, 'min': mn,
                'times': times,
                'speedup_vs_baseline': speedup,
            })
        print
    if csv_path:
        with open(csv_path, 'w') as f:
            json.dump(rows, f, indent=2)
        sys.stderr.write('wrote %s\n' % csv_path)
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
