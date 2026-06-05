#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""Comprehensive cross-tier performance evaluation for the TLA threaded-code VM.

This is the heavier successor to ``eval_perf.py``.  It keeps the same four
headline metrics but adds statistical robustness (repeated runs + variance),
aggregate speedups (geometric mean), an abort/guard breakdown, CSV output and
matplotlib PDF plots.

Tiers (see tla.run / targettla.py entry_point):

  0 interp    pure interpreter, JIT off            -- the speedup baseline
  1 threaded  threaded-code generation             (Frame.interp, tier1driver)
  2 inliner   residual-arithmetic inliner          (JitFrame._interp, virtualizable, _t2_*)
  3 tracing   conventional tracing JIT             (JitFrame3._interp, virtualizable, *_inline)
  4 hybrid    selective inliner                    (JitFrame3 hybrid=True: per-site
                                                     inline if monomorphic, residual if poly)

Metrics, per (benchmark, tier):

  (1) tracing + compilation time   PYPYLOG=jit-summary  (Tracing time / Backend time)
  (2) trace count & length         jit-summary (#loops, #bridges, opt ops, aborts)
                                    + PYPYLOG=jit-log-opt (per-trace op counts)
  (3) memory footprint             /usr/bin/time -v  (maximum resident set size)
  (4) running-time improvement     warm steady-state time + speedup over interp,
                                    measured over N repeats -> median / min / CV

Programs are compiled straight from the high-level lang/<name>.tla definition
(parser.compile_source + bytecode.assemble), cached as .tlc under /tmp.

Two interpreters are used on purpose: the orchestration (compiling .tla, driving
targettla-c, collecting numbers) runs under python2 (pypy2.7); the plotting runs
under python3 (which has matplotlib).  The script re-execs itself with
``--plot-only`` under python3 when it is time to draw, so a single file does
both.  The body is written to parse under both interpreters.

Usage:
  python2 eval_comprehensive.py [options] [NAME ...]

Options:
  --full              evaluate every lang/*.tla (default: the curated subset)
  --shootout          evaluate the ported shootout suite (dot/matmul/heapsort/...)
  --repeats N         timing repeats per (benchmark, tier)        [default 5]
  --iters N           override the per-program iteration count
  --timeout SEC       per-child-process timeout                   [default 120]
  --tiers a,b,..      restrict to a subset of tiers, e.g. 0,1     [default 0,1,2,3]
  --csv PATH          write the flat per-(benchmark,tier) CSV      [default: <out>.csv]
  --json PATH         write the full structured results JSON       [default: <out>.json]
  --pdf PATH          write the matplotlib report PDF              [default: <out>.pdf]
  --out PREFIX        basename prefix for the default csv/json/pdf [default eval_report]
  --no-plot           skip the PDF (do not invoke python3/matplotlib)
  --quick             shorthand for --repeats 2
  -h, --help          show this help

  --plot-only JSON PDF   (internal) draw the PDF from a results JSON; run by the
                         python3 re-exec, but usable by hand too.
"""
from __future__ import print_function, division

import os
import re
import sys
import json
import glob
import math
import subprocess

THIS = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(THIS, '..', '..', '..', '..'))
LANG = os.path.join(THIS, 'lang')
CACHE = '/tmp/tla_eval_cache'

BIN = os.environ.get('TLA_BIN', os.path.join(THIS, 'targettla-c'))

# label, --tier value
ALL_TIERS = [('interp', '0'), ('threaded', '1'), ('inliner', '2'),
             ('tracing', '3'), ('hybrid', '4')]
JIT_LABELS = ('threaded', 'inliner', 'tracing', 'hybrid')

# (x, iters) per program.  x is the workload size, iters the number of times the
# program is re-run inside one process (the first is cold, the rest are warm).
# Tail-recursive loops take a big x; non-tail recursion must stay under the
# interpreter's MAX_INTERP_DEPTH (50), hence the small x there.
SIZES = {
    # tail-recursive loops (the JIT sweet spot)
    'mb_loop': (1000000, 12), 'mb_count': (1000000, 12), 'mb_sum': (1000000, 12),
    'mb_inc': (1000000, 12), 'mb_pass': (1000000, 12),
    'gcd': (1000000, 12), 'sum-tail': (1000000, 12), 'fib-tail': (1000000, 12),
    'sh_countdown': (1000000, 12), 'sh_sumtail': (1000000, 12),
    'sh_gcd': (1000000, 12), 'loop': (1000000, 12),
    # recursion / call-assembler workloads
    'sh_fib': (30, 12), 'sh_binarytrees': (18, 12), 'sh_collatz': (100000, 12),
    'sh_primes': (30000, 12), 'sh_tak': (24, 12), 'tak': (24, 12),
    'sh_ack': (9, 12), 'ack': (10, 12), 'tarai': (5, 12),
    # shallow non-tail recursion (depth-bounded)
    'fib': (25, 12), 'sum': (40, 12), 'fact': (10, 12), 'sh_fact': (10, 12),
    'square': (10, 12),
    # ---- shootout ports: generic kernels reused across int+float (poly) vs
    #      monomorphic (int/flt) vs sequential-poly vs monomorphic controls ----
    'dot_int': (30, 12), 'dot_flt': (30, 12), 'dot_mix': (30, 12),
    'matmul_int': (20, 12), 'matmul_flt': (20, 12), 'matmul_poly': (20, 12),
    'heapsort_int': (600, 12), 'heapsort_flt': (600, 12), 'heapsort_poly': (600, 12),
    'nsieve': (2000, 12), 'mandelbrot': (35, 12),
}
DEFAULT_SIZE = (10, 12)

# Shootout suite: pass with `--shootout` (or name them explicitly).
SHOOTOUT = ['dot_int', 'dot_flt', 'dot_mix',
            'matmul_int', 'matmul_flt', 'matmul_poly',
            'heapsort_int', 'heapsort_flt', 'heapsort_poly',
            'nsieve', 'mandelbrot']

# The curated default set: representative of every workload shape, sized so the
# JIT tiers do real warm work.  --full runs everything in lang/.  (gcd/sh_gcd are
# left out of the curated set: Euclid converges in O(log) steps, so they finish
# below the timer resolution and carry no perf signal -- use --full for them.)
CURATED = ['mb_loop', 'mb_count', 'mb_sum', 'mb_inc', 'sum-tail',
           'sh_fib', 'sh_collatz', 'sh_primes', 'sh_binarytrees',
           'sh_tak', 'tak', 'sh_ack', 'ack', 'tarai']


# ===========================================================================
# program compilation (.tla -> cached .tlc)  [python2 only]
# ===========================================================================
def compile_tla_to_tlc(name):
    """Compile lang/<name>.tla to a cached .tlc; return the path or None."""
    src = os.path.join(LANG, name + '.tla')
    if not os.path.exists(src):
        return None
    if not os.path.isdir(CACHE):
        os.makedirs(CACHE)
    dst = os.path.join(CACHE, name + '.tlc')
    if os.path.exists(dst) and os.path.getmtime(dst) >= os.path.getmtime(src):
        return dst
    if ROOT not in sys.path:
        sys.path.insert(0, ROOT)
    from rpython.jit.tl.threadedcode import parser
    from rpython.jit.tl.threadedcode import bytecode as bc
    with open(src) as f:
        code = parser.compile_source(f.read())
    with open(dst, 'wb') as f:
        f.write(bc.assemble(code))
    return dst


# ===========================================================================
# process running / parsing  [python2]
# ===========================================================================
def _run(tier, prog, x, iters, pypylog='', with_time=False, timeout=120):
    """Run one targettla-c process. Returns (stdout, stderr, rc)."""
    cmd = ['timeout', str(timeout)]
    if with_time:
        cmd += ['/usr/bin/time', '-v']
    cmd += [BIN, '--tier', tier, prog, str(x), str(iters)]
    env = dict(os.environ)
    env['PYPYLOG'] = pypylog
    try:
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                             stderr=subprocess.PIPE, env=env)
        out, err = p.communicate()
        if not isinstance(out, str):
            out = out.decode('utf-8', 'replace')
            err = err.decode('utf-8', 'replace')
        return out, err, p.returncode
    except OSError as e:
        return '', str(e), -1


def _parse_times(out):
    """The binary prints `iters` timing lines (floats, seconds) then the result
    on the LAST line.  The result is an integer (also parses as float), so split
    by position: result = last line, times = the lines before it."""
    lines = [l.strip() for l in out.splitlines() if l.strip()]
    if not lines:
        return [], None
    result = lines[-1]
    times = []
    for s in lines[:-1]:
        try:
            times.append(float(s))
        except ValueError:
            pass
    return times, result


def _max_rss_kb(err):
    m = re.search(r'Maximum resident set size \(kbytes\):\s*(\d+)', err)
    return int(m.group(1)) if m else None


def _parse_summary(text):
    d = {}
    for key, f in (('Tracing', 'trace'), ('Backend', 'backend')):
        m = re.search(r'%s:\s*(\d+)\s+([\d.]+)' % key, text)
        if m:
            d[f + '_n'] = int(m.group(1))
            d[f + '_t'] = float(m.group(2))
    for key, f in (('recorded ops', 'recorded'), ('opt ops', 'optops'),
                   ('guards', 'guards'),
                   ('Total # of loops', 'loops'),
                   ('Total # of bridges', 'bridges')):
        m = re.search(r'(?m)^\s*%s:\s*(\d+)' % re.escape(key), text)
        if m:
            d[f] = int(m.group(1))
    # aborts: the summary has "abort: N" lines (one per reason) and/or a count.
    aborts = re.findall(r'(?mi)^\s*abort[a-z ]*:\s*(\d+)', text)
    if aborts:
        d['aborts'] = sum(int(a) for a in aborts)
    return d


def _read_safe(path):
    try:
        f = open(path)
        try:
            return f.read()
        finally:
            f.close()
    except (IOError, OSError):
        return ''


# ===========================================================================
# statistics  [python2]
# ===========================================================================
def _median(xs):
    s = sorted(xs)
    n = len(s)
    if n == 0:
        return None
    m = n // 2
    return s[m] if n % 2 else 0.5 * (s[m - 1] + s[m])


def _mean(xs):
    return sum(xs) / len(xs) if xs else None


def _stddev(xs):
    if len(xs) < 2:
        return 0.0
    mu = _mean(xs)
    var = sum((v - mu) ** 2 for v in xs) / (len(xs) - 1)
    return math.sqrt(var)


def _cv(xs):
    """Coefficient of variation (stddev/mean), a unit-free noise measure."""
    mu = _mean(xs)
    if not mu:
        return None
    return _stddev(xs) / mu


def _warm_of(times):
    """Steady-state time of one run: the min over the warm iterations (the first
    iteration pays tracing+compilation)."""
    if not times:
        return None
    if len(times) == 1:
        return times[0]
    return min(times[1:])


def _geomean(xs):
    xs = [v for v in xs if v and v > 0]
    if not xs:
        return None
    return math.exp(sum(math.log(v) for v in xs) / len(xs))


# ===========================================================================
# evaluation  [python2]
# ===========================================================================
def evaluate(name, x, iters, repeats, timeout, tiers):
    prog = compile_tla_to_tlc(name)
    if prog is None:
        sys.stderr.write('skip %s: no lang/%s.tla\n' % (name, name))
        return None

    rows = {}
    interp_warm = None
    for label, tier in tiers:
        row = {'label': label, 'tier': tier, 'status': 'ok', 'result': None,
               'cold_list': [], 'warm_list': [], 'rss_list': []}

        # --- (3)+(4) repeated timing runs (each wrapped in /usr/bin/time -v;
        #     the per-iteration numbers come from time() *inside* the program so
        #     the wrapper does not perturb them, but it gives us RSS for free).
        for r in range(repeats):
            out, err, rc = _run(tier, prog, x, iters, pypylog='',
                                 with_time=True, timeout=timeout)
            if rc == 124:
                row['status'] = 'timeout'
                break
            if ('Fatal' in err) or ('Fatal' in out) or (rc not in (0, None)):
                row['status'] = 'crash'
                break
            times, result = _parse_times(out)
            if not times or result is None:
                row['status'] = 'noresult'
                break
            row['result'] = result
            row['cold_list'].append(times[0])
            row['warm_list'].append(_warm_of(times))
            rss = _max_rss_kb(err)
            if rss is not None:
                row['rss_list'].append(rss)

        # aggregate the timing stats
        if row['warm_list']:
            row['warm'] = _median(row['warm_list'])
            row['warm_min'] = min(row['warm_list'])
            row['warm_cv'] = _cv(row['warm_list'])
            row['cold'] = _median(row['cold_list'])
        else:
            row['warm'] = row['warm_min'] = row['warm_cv'] = row['cold'] = None
        row['rss_kb'] = _median(row['rss_list']) if row['rss_list'] else None

        # --- (1)+(2) JIT instrumentation (tiers 1/2/3 only; tier 0 has JIT off),
        #     a single run each; skip when the timing runs already failed.
        if tier != '0' and row['status'] == 'ok':
            js = os.path.join(CACHE, '_js_%s_%s.log' % (name, tier))
            _run(tier, prog, x, iters, pypylog='jit-summary:' + js,
                 timeout=timeout)
            row.update(_parse_summary(_read_safe(js)))
            jlo = os.path.join(CACHE, '_jlo_%s_%s.log' % (name, tier))
            _run(tier, prog, x, iters, pypylog='jit-log-opt:' + jlo,
                 timeout=timeout)
            ops = [int(n) for n in
                   re.findall(r'with (\d+) ops', _read_safe(jlo))]
            row['trace_ops'] = ops
            row['n_traces'] = len(ops)
            if ops:
                row['trace_len_min'] = min(ops)
                row['trace_len_max'] = max(ops)
                row['trace_len_mean'] = _mean(ops)

        rows[label] = row
        if label == 'interp':
            interp_warm = row['warm']
        sys.stderr.write('  %-9s tier%s  result=%s  warm=%s  status=%s\n' % (
            label, tier, row['result'],
            ('%.3fms' % (row['warm'] * 1000)) if row['warm'] else '-',
            row['status']))

    # speedups vs interp warm
    for label, _ in tiers:
        w = rows[label]['warm']
        rows[label]['speedup'] = (interp_warm / w) if (interp_warm and w) else None

    # correctness vs interp (tier 0) result
    base = rows['interp']['result'] if 'interp' in rows else None
    for label, _ in tiers:
        rr = rows[label]
        if rr['status'] != 'ok':
            rr['correct'] = False
        elif base is None:
            rr['correct'] = None
        else:
            rr['correct'] = (rr['result'] == base)

    return {'name': name, 'x': x, 'iters': iters, 'repeats': repeats,
            'rows': rows}


# ===========================================================================
# console report  [python2]
# ===========================================================================
def _ms(v):
    return ('%.3f' % (v * 1000.0)) if isinstance(v, float) else '-'


def _fnum(v, fmt='%d'):
    return (fmt % v) if v is not None else '-'


def _pct(v):
    return ('%.1f%%' % (v * 100)) if isinstance(v, float) else '-'


def _spd(v):
    return ('%.2fx' % v) if isinstance(v, float) else '-'


def report(results, tiers):
    labels = [l for l, _ in tiers]
    jitl = [l for l in JIT_LABELS if l in labels]
    out = []
    w = out.append
    line = '=' * (20 + 11 * len(labels))
    w(line)
    w('TIERS:  ' + ' | '.join('%s %s' % (t, l) for (l, t) in tiers))
    w(line)

    # correctness
    w('\n## Correctness (result per tier; must all match interp)')
    w('%-16s ' % 'benchmark' + ' '.join('%-12s' % l for l in labels) + '  status')
    for r in results:
        ro = r['rows']
        cells, ok = [], True
        for l in labels:
            rr = ro[l]
            v = rr['result'] if rr['status'] == 'ok' else ('<%s>' % rr['status'])
            cells.append('%-12s' % (v if v is not None else '-'))
            if rr.get('correct') is False:
                ok = False
        w('%-16s ' % r['name'] + ' '.join(cells) + '  ' +
          ('OK' if ok else 'MISMATCH'))

    # (4) running time + speedup
    w('\n## (4) Warm steady-state time (ms, median of repeats); spd=speedup vs interp')
    w('%-15s' % 'benchmark' + ''.join('%9s' % l[:8] for l in labels) +
      '  |' + ''.join('%8s' % ('spd:' + l[:4]) for l in jitl))
    for r in results:
        ro = r['rows']
        w('%-15s' % r['name'] +
          ''.join('%9s' % (_ms(ro[l]['warm']) if l in ro else '-') for l in labels) +
          '  |' + ''.join('%8s' % (_spd(ro[l]['speedup']) if l in ro else '-')
                          for l in jitl))
    w('%-15s' % 'GEOMEAN' + ''.join('%9s' % '' for _ in labels) + '  |' +
      ''.join('%8s' % _spd(_geomean([r['rows'].get(l, {}).get('speedup')
                                     for r in results])) for l in jitl))

    # measurement noise
    w('\n## Measurement noise (coefficient of variation of warm time)')
    w('%-15s' % 'benchmark' + ''.join('%11s' % l[:10] for l in labels))
    for r in results:
        ro = r['rows']
        w('%-15s' % r['name'] +
          ''.join('%11s' % _pct(ro.get(l, {}).get('warm_cv')) for l in labels))

    # (1) tracing + compilation time
    w('\n## (1) Tracing + compilation time (ms): trace / comp(backend)')
    w('%-15s' % 'benchmark' + ''.join('%17s' % l for l in jitl))
    w('%-15s' % '' + ''.join('%8s %8s' % ('trace', 'comp') for _ in jitl))
    for r in results:
        ro = r['rows']
        w('%-15s' % r['name'] + ''.join(
            '%8s %8s' % (_ms(ro.get(l, {}).get('trace_t')),
                         _ms(ro.get(l, {}).get('backend_t'))) for l in jitl))

    # (2) traces: number, length, aborts
    w('\n## (2) Traces -- #=loops+bridges, ops=opt ops total, avg=ops/#, ab=aborts')
    w('%-15s' % 'benchmark' + ''.join('%23s' % l for l in jitl))
    w('%-15s' % '' + ''.join('%5s%7s%5s%5s' % ('#', 'ops', 'avg', 'ab')
                             for _ in jitl))
    for r in results:
        ro = r['rows']
        cells = []
        for l in jitl:
            d = ro.get(l, {})
            ntr = (d.get('loops', 0) or 0) + (d.get('bridges', 0) or 0)
            ops = d.get('optops')
            avg = ('%d' % (ops // ntr)) if (ops and ntr) else '-'
            cells.append('%5s%7s%5s%5s' % (ntr or '-', _fnum(ops), avg,
                                           _fnum(d.get('aborts'))))
        w('%-15s' % r['name'] + ''.join(cells))

    # (3) memory footprint
    w('\n## (3) Memory footprint -- maximum resident set size (MB, median)')
    w('%-15s' % 'benchmark' + ''.join('%11s' % l[:10] for l in labels))
    for r in results:
        ro = r['rows']
        def mb(l):
            v = ro.get(l, {}).get('rss_kb')
            return ('%.1f' % (v / 1024.0)) if v else '-'
        w('%-15s' % r['name'] + ''.join('%11s' % mb(l) for l in labels))

    return '\n'.join(out)


# ===========================================================================
# CSV output  [python2]
# ===========================================================================
def write_csv(results, path, tiers):
    cols = ['benchmark', 'x', 'iters', 'tier', 'label', 'result', 'correct',
            'status', 'warm_s', 'warm_min_s', 'warm_cv', 'cold_s', 'speedup',
            'trace_time_s', 'backend_time_s', 'loops', 'bridges', 'aborts',
            'recorded_ops', 'opt_ops', 'guards', 'n_traces', 'trace_len_mean',
            'rss_kb']
    f = open(path, 'w')
    try:
        f.write(','.join(cols) + '\n')
        for r in results:
            for label, tier in tiers:
                d = r['rows'].get(label, {})
                vals = [r['name'], r['x'], r['iters'], tier, label,
                        d.get('result'), d.get('correct'), d.get('status'),
                        d.get('warm'), d.get('warm_min'), d.get('warm_cv'),
                        d.get('cold'), d.get('speedup'),
                        d.get('trace_t'), d.get('backend_t'), d.get('loops'),
                        d.get('bridges'), d.get('aborts'), d.get('recorded'),
                        d.get('optops'), d.get('guards'), d.get('n_traces'),
                        d.get('trace_len_mean'), d.get('rss_kb')]
                f.write(','.join('' if v is None else str(v) for v in vals) + '\n')
    finally:
        f.close()
    sys.stderr.write('wrote %s\n' % path)


# ===========================================================================
# plotting  [python3 re-exec: --plot-only]
# ===========================================================================
def make_plots(results, pdf_path):
    """Draw the report PDF from `results`.  Imported lazily so the python2
    orchestration never needs matplotlib."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    names = [r['name'] for r in results]
    jit = list(JIT_LABELS)
    colors = {'interp': '#888888', 'threaded': '#1f77b4',
              'inliner': '#2ca02c', 'tracing': '#d62728', 'hybrid': '#9467bd'}

    def col(label, key, scale=1.0):
        out = []
        for r in results:
            v = r['rows'].get(label, {}).get(key)
            out.append(v * scale if isinstance(v, (int, float)) else float('nan'))
        return out

    import numpy as np
    x = np.arange(len(names))

    with PdfPages(pdf_path) as pdf:
        # --- Fig 1: warm speedup vs interp (threaded/inliner/tracing) ---------
        fig, ax = plt.subplots(figsize=(max(8, len(names) * 0.9), 5))
        nj = len(jit); wbar = 0.8 / nj
        for i, l in enumerate(jit):
            ax.bar(x + (i - (nj - 1) / 2.0) * wbar, col(l, 'speedup'), wbar,
                   label=l, color=colors[l])
        ax.axhline(1.0, color='k', lw=0.8, ls='--')
        ax.set_yscale('log')
        ax.set_ylabel('speedup vs interp (log)')
        ax.set_title('(4) Warm steady-state speedup over interp (higher is better)')
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=40, ha='right')
        ax.legend()
        ax.grid(axis='y', which='both', ls=':', alpha=0.4)
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

        # --- Fig 2: warm runtime (ms), all tiers ------------------------------
        fig, ax = plt.subplots(figsize=(max(8, len(names) * 0.9), 5))
        nt = len(ALL_TIERS); wbar = 0.8 / nt
        for i, (l, _) in enumerate(ALL_TIERS):
            ax.bar(x + (i - (nt - 1) / 2.0) * wbar, col(l, 'warm', 1000.0), wbar,
                   label=l, color=colors[l])
        ax.set_yscale('log')
        ax.set_ylabel('warm time per iteration (ms, log)')
        ax.set_title('(4) Warm steady-state running time (lower is better)')
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=40, ha='right')
        ax.legend()
        ax.grid(axis='y', which='both', ls=':', alpha=0.4)
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

        # --- Fig 3: compile cost = tracing + backend time, stacked, per tier --
        fig, axes = plt.subplots(1, len(jit), figsize=(4.0 * len(jit), 4.5),
                                 sharey=True)
        for ax, l in zip(axes, jit):
            tr = col(l, 'trace_t', 1000.0)
            be = col(l, 'backend_t', 1000.0)
            ax.bar(x, tr, 0.6, label='tracing', color='#ff7f0e')
            ax.bar(x, be, 0.6, bottom=[0 if t != t else t for t in tr],
                   label='backend', color='#9467bd')
            ax.set_title(l)
            ax.set_xticks(x)
            ax.set_xticklabels(names, rotation=60, ha='right', fontsize=7)
        axes[0].set_ylabel('compile time (ms)')
        axes[0].legend()
        fig.suptitle('(1) Tracing + backend (compilation) time, stacked')
        fig.tight_layout(rect=(0, 0, 1, 0.96))
        pdf.savefig(fig)
        plt.close(fig)

        # --- Fig 4: memory footprint (MB), all tiers --------------------------
        fig, ax = plt.subplots(figsize=(max(8, len(names) * 0.9), 5))
        nt = len(ALL_TIERS); wbar = 0.8 / nt
        for i, (l, _) in enumerate(ALL_TIERS):
            ax.bar(x + (i - (nt - 1) / 2.0) * wbar, col(l, 'rss_kb', 1.0 / 1024.0),
                   wbar, label=l, color=colors[l])
        ax.set_ylabel('max RSS (MB)')
        ax.set_title('(3) Memory footprint (maximum resident set size)')
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=40, ha='right')
        ax.legend()
        ax.grid(axis='y', ls=':', alpha=0.4)
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

        # --- Fig 5: trace count and avg length, per tier ----------------------
        fig, (a1, a2) = plt.subplots(1, 2, figsize=(14, 4.5))
        nj = len(jit); wbar = 0.8 / nj
        for i, l in enumerate(jit):
            ntr = []
            avg = []
            for r in results:
                d = r['rows'].get(l, {})
                n = (d.get('loops', 0) or 0) + (d.get('bridges', 0) or 0)
                ops = d.get('optops')
                ntr.append(n if n else float('nan'))
                avg.append((ops / n) if (ops and n) else float('nan'))
            a1.bar(x + (i - (nj - 1) / 2.0) * wbar, ntr, wbar, label=l, color=colors[l])
            a2.bar(x + (i - (nj - 1) / 2.0) * wbar, avg, wbar, label=l, color=colors[l])
        a1.set_title('(2) #traces (loops + bridges)')
        a2.set_title('(2) avg trace length (opt ops / trace)')
        for ax in (a1, a2):
            ax.set_xticks(x)
            ax.set_xticklabels(names, rotation=60, ha='right', fontsize=7)
            ax.grid(axis='y', ls=':', alpha=0.4)
            ax.legend()
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

    sys.stderr.write('wrote %s\n' % pdf_path)


# ===========================================================================
# main  [python2 orchestration]
# ===========================================================================
def _usage():
    sys.stdout.write(__doc__)


def main(argv):
    # internal python3 plotting entry point
    if argv and argv[0] == '--plot-only':
        if len(argv) != 3:
            sys.stderr.write('usage: --plot-only RESULTS.json OUT.pdf\n')
            return 2
        with open(argv[1]) as f:
            results = json.load(f)
        make_plots(results, argv[2])
        return 0

    full = no_plot = shootout = False
    repeats = 5
    iters_override = None
    timeout = 120
    tiers = list(ALL_TIERS)
    out_prefix = 'eval_report'
    csv_path = json_path = pdf_path = None
    names = []

    i = 0
    while i < len(argv):
        a = argv[i]
        if a in ('-h', '--help'):
            _usage(); return 0
        elif a == '--full':
            full = True
        elif a == '--shootout':
            shootout = True
        elif a == '--no-plot':
            no_plot = True
        elif a == '--quick':
            repeats = 2
        elif a == '--repeats':
            repeats = int(argv[i + 1]); i += 1
        elif a == '--iters':
            iters_override = int(argv[i + 1]); i += 1
        elif a == '--timeout':
            timeout = int(argv[i + 1]); i += 1
        elif a == '--tiers':
            sel = set(argv[i + 1].split(',')); i += 1
            tiers = [(l, t) for (l, t) in ALL_TIERS if t in sel]
        elif a == '--out':
            out_prefix = argv[i + 1]; i += 1
        elif a == '--csv':
            csv_path = argv[i + 1]; i += 1
        elif a == '--json':
            json_path = argv[i + 1]; i += 1
        elif a == '--pdf':
            pdf_path = argv[i + 1]; i += 1
        else:
            names.append(a)
        i += 1

    if not os.path.exists(BIN):
        sys.stderr.write('missing binary: %s (set TLA_BIN)\n' % BIN)
        return 1
    if 'interp' not in [l for l, _ in tiers]:
        # interp is the speedup/correctness baseline; always include it.
        tiers = [('interp', '0')] + tiers

    # build the work list
    if names:
        worklist = names
    elif shootout:
        worklist = list(SHOOTOUT)
    elif full:
        worklist = sorted(os.path.splitext(os.path.basename(p))[0]
                          for p in glob.glob(os.path.join(LANG, '*.tla')))
    else:
        worklist = list(CURATED)

    csv_path = csv_path or (out_prefix + '.csv')
    json_path = json_path or (out_prefix + '.json')
    pdf_path = pdf_path or (out_prefix + '.pdf')

    results = []
    for name in worklist:
        x, it = SIZES.get(name, DEFAULT_SIZE)
        if iters_override:
            it = iters_override
        sys.stderr.write('[%s x=%s iters=%s repeats=%s]\n' % (name, x, it, repeats))
        r = evaluate(name, x, it, repeats, timeout, tiers)
        if r is not None:
            results.append(r)

    if not results:
        sys.stderr.write('no benchmarks evaluated\n')
        return 1

    text = report(results, tiers)
    print(text)

    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    sys.stderr.write('wrote %s\n' % json_path)
    write_csv(results, csv_path, tiers)

    if not no_plot:
        # plotting needs matplotlib, which lives under python3 here.
        try:
            import matplotlib  # noqa: F401
            make_plots(results, pdf_path)
        except ImportError:
            rc = subprocess.call(['python3', os.path.abspath(__file__),
                                  '--plot-only', json_path, pdf_path])
            if rc != 0:
                sys.stderr.write('plotting via python3 failed (rc=%s); '
                                 'CSV/JSON still written.\n' % rc)

    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
