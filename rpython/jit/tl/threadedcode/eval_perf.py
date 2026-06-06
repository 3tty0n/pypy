#!/usr/bin/env python2
"""Cross-tier performance evaluation for the TLA threaded-code VM.

Tiers (see tla.run / targettla.py entry_point):

  0 interp    pure interpreter, JIT off            -- the speedup baseline
  1 threaded  threaded-code generation             (Frame.interp, tier1driver)
  2 inliner   stack-manipulation inliner           (JitFrame._interp, virtualizable)
  3 tracing   conventional tracing JIT             (Frame._interp, non-virtualizable)

For each (benchmark, tier) it reports the four requested metrics:

  (1) tracing + compilation time   PYPYLOG=jit-summary  (Tracing time / Backend time)
  (2) trace length & #traces       jit-summary (#loops, #bridges, opt ops)
                                    + PYPYLOG=jit-log-opt (per-trace op counts)
  (3) memory footprint             /usr/bin/time -v  (maximum resident set size)
  (4) running-time improvement     warm wall time + speedup over tier 0 (interp)

Programs are compiled straight from the high-level lang/<name>.tla definition
(parser.compile_source + bytecode.assemble) and cached as .tlc under /tmp -- this
tool is self-contained and has no dependency on any other eval/bench module.

A child that times out / crashes (e.g. a benchmark that still hangs on a tier) is
reported as ``crashed`` and skipped for the JIT-instrumentation passes -- the rest
of the matrix still prints.

Usage:
  python2 eval_perf.py [--json OUT.json] [--csv OUT.csv]
                       [--iters N] [--timeout SEC] [--tiers a,b,..] [NAME ...]
"""
import os
import re
import sys
import json
import subprocess

THIS = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(THIS, '..', '..', '..', '..'))
for _p in (THIS, ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

LANG = os.path.join(THIS, 'lang')
CACHE = '/tmp/tla_eval_perf'
BIN = os.environ.get('TLA_BIN', os.path.join(THIS, 'targettla-c'))
TIMEOUT = 120                       # seconds, per child process (override: --timeout)

# label, --tier value
ALL_TIERS = [('interp', '0'), ('threaded', '1'), ('inliner', '2'), ('tracing', '3')]

# (name, x, iters).  Sized so the JIT tiers trace/compile real work while the
# non-tail-recursive programs stay within the interpreter's MAX_INTERP_DEPTH (50).
# Default set = programs that compile/run on every tier today (tail loops + tree
# recursion).  Deep FRAME_RESET-tail recursion (ack/tak/tarai) still hangs
# tier 1 -- pass those as NAME args to measure them (they'll show as crashed).
BENCHMARKS = [
    ('sumtail',        1000000, 12),
    ('loop',           1000000, 12),
    ('inc',            1000000, 12),
    ('gcd',            1000000, 12),
    ('fib',            30,      12),
    ('binarytrees',    18,      12),
    ('primes',         30000,   12),
    ('collatz',        100000,  12),
]
DEFAULT_SIZE = (10, 12)             # for a NAME arg not in BENCHMARKS


# ----------------------------------------------------------------------------
# program compilation (.tla -> cached .tlc)
# ----------------------------------------------------------------------------
def compile_program(name):
    """Compile lang/<name>.tla to a cached .tlc; return the path or None."""
    src = os.path.join(LANG, name + '.tla')
    if not os.path.exists(src):
        return None
    if not os.path.isdir(CACHE):
        os.makedirs(CACHE)
    dst = os.path.join(CACHE, name + '.tlc')
    if os.path.exists(dst) and os.path.getmtime(dst) >= os.path.getmtime(src):
        return dst
    from rpython.jit.tl.threadedcode import parser
    from rpython.jit.tl.threadedcode import bytecode as bc
    with open(src) as f:
        code = parser.compile_source(f.read())
    with open(dst, 'wb') as f:
        f.write(bc.assemble(code))
    return dst


# ----------------------------------------------------------------------------
# process running / parsing
# ----------------------------------------------------------------------------
def _run(flags, prog, x, iters, pypylog='', with_time=False):
    cmd = ['timeout', str(TIMEOUT)]
    if with_time:
        cmd += ['/usr/bin/time', '-v']
    cmd += [BIN] + flags + [prog, str(x), str(iters)]
    env = dict(os.environ)
    env['PYPYLOG'] = pypylog
    try:
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                             stderr=subprocess.PIPE, env=env)
        out, err = p.communicate()
        return out, err, p.returncode
    except OSError as e:
        return '', str(e), -1


def _parse_times(out):
    # The binary prints `iters` timing lines (floats, seconds) then the result on
    # the LAST line.  The result is an integer (which also parses as a float), so
    # split by position: result = last line, times = the lines before it.
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


def _warm(times):
    "Steady-state time: the min over warm iterations (the first one compiles)."
    if not times:
        return None
    if len(times) == 1:
        return times[0]
    return min(times[1:])


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
                   ('Total # of loops', 'loops'),
                   ('Total # of bridges', 'bridges')):
        m = re.search(r'(?m)^%s:\s*(\d+)' % re.escape(key), text)
        if m:
            d[f] = int(m.group(1))
    return d


def _read_safe(path):
    try:
        return open(path).read()
    except IOError:
        return ''


# ----------------------------------------------------------------------------
# evaluation
# ----------------------------------------------------------------------------
def evaluate(name, x, iters, tiers):
    prog = compile_program(name)
    if prog is None:
        sys.stderr.write('skip %s: no lang/%s.tla\n' % (name, name))
        return None
    rows = {}
    interp_warm = None
    for label, tier in tiers:
        flags = ['--tier', tier]
        # Run 1: clean (no JIT logging) under /usr/bin/time -v -> timing + RSS.
        out, err, rc = _run(flags, prog, x, iters, pypylog='', with_time=True)
        times, result = _parse_times(out)
        timed_out = (rc == 124)
        crashed = (timed_out or ('Fatal' in err) or ('Fatal' in out) or
                   (rc not in (0, None)) or not times)
        row = {
            'label': label, 'tier': tier,
            'result': (None if timed_out else result),
            'crashed': crashed, 'timed_out': timed_out,
            'cold': (times[0] if times else None), 'warm': _warm(times),
            'rss_kb': _max_rss_kb(err),
        }
        # Runs 2 & 3: JIT instrumentation (tiers 1/2/3 only; tier 0 has JIT off).
        if tier != '0' and not crashed:
            js = os.path.join(CACHE, '_js_%s_%s.log' % (name, tier))
            _run(flags, prog, x, iters, pypylog='jit-summary:' + js)
            row.update(_parse_summary(_read_safe(js)))
            jlo = os.path.join(CACHE, '_jlo_%s_%s.log' % (name, tier))
            _run(flags, prog, x, iters, pypylog='jit-log-opt:' + jlo)
            row['trace_ops'] = [int(n) for n in
                                re.findall(r'with (\d+) ops', _read_safe(jlo))]
        rows[label] = row
        if label == 'interp':
            interp_warm = row['warm']
        sys.stderr.write('  %-8s tier%s  result=%s warm=%s%s\n' % (
            label, tier, row['result'], row['warm'],
            '  CRASHED' if crashed else ''))
    for label, _ in tiers:
        w = rows[label]['warm']
        rows[label]['speedup'] = (interp_warm / w) if (interp_warm and w) else None
    return {'name': name, 'x': x, 'iters': iters, 'rows': rows}


# ----------------------------------------------------------------------------
# reporting
# ----------------------------------------------------------------------------
def _ms(v):
    return '%.3f' % (v * 1000.0) if isinstance(v, float) else '-'


def _fnum(v, fmt='%d'):
    return (fmt % v) if v is not None else '-'


def _cell(row):
    "result cell: the value, or <crash>/<timeout>."
    if row['timed_out']:
        return '<timeout>'
    if row['crashed']:
        return '<crash>'
    return row['result']


def report(results, tiers):
    labels = [l for l, _ in tiers]
    jit_labels = [l for l in ('threaded', 'inliner', 'tracing') if l in labels]
    out = []
    w = out.append
    w('=' * 92)
    w('TIERS:  ' + ' | '.join('%s %s' % (t, l) for (l, t) in tiers))
    w('=' * 92)

    # correctness
    w('\n## Correctness (result per tier; must all match interp)')
    w('%-16s ' % 'benchmark' + ' '.join('%-12s' % l for l in labels) + '  status')
    for r in results:
        ro = r['rows']
        base = ro['interp']['result'] if 'interp' in ro else None
        cells = [_cell(ro[l]) for l in labels]
        ok = all((not ro[l]['crashed']) and (ro[l]['result'] == base)
                 for l in labels)
        w('%-16s ' % r['name'] +
          ' '.join('%-12s' % (c if c is not None else '-') for c in cells) +
          '  ' + ('OK' if ok else 'MISMATCH'))

    # (4) running time + speedup
    w('\n## (4) Running time -- warm steady-state (ms) and speedup vs interp')
    w('%-16s ' % 'benchmark' + ' '.join('%10s' % l for l in labels) +
      ' | ' + ' '.join('%8s' % l[:3] for l in jit_labels))
    for r in results:
        ro = r['rows']
        warm = ' '.join('%10s' % _ms(ro[l]['warm']) for l in labels)
        spd = ' '.join('%7sx' % _fnum(ro[l]['speedup'], '%.2f') for l in jit_labels)
        w('%-16s %s | %s' % (r['name'], warm, spd))

    # (1) tracing + compilation time
    w('\n## (1) Tracing + compilation time (ms): trace=tracing, comp=backend')
    w('%-16s | %s' % ('benchmark',
                      ' | '.join('%-16s' % l for l in jit_labels)))
    w('%-16s | %s' % ('',
                      ' | '.join('%7s %7s' % ('trace', 'comp') for _ in jit_labels)))
    for r in results:
        ro = r['rows']
        cells = ['%7s %7s' % (_ms(ro[l].get('trace_t')), _ms(ro[l].get('backend_t')))
                 for l in jit_labels]
        w('%-16s | %s' % (r['name'], ' | '.join(cells)))

    # (2) traces: number and length
    w('\n## (2) Traces -- #=loops+bridges, ops=optimised ops (total), avg=ops/#')
    w('%-16s | %s' % ('benchmark', ' | '.join('%-18s' % l for l in jit_labels)))
    w('%-16s | %s' % ('',
                      ' | '.join('%4s %6s %5s' % ('#', 'ops', 'avg')
                                 for _ in jit_labels)))
    for r in results:
        ro = r['rows']
        cells = []
        for l in jit_labels:
            d = ro[l]
            ntr = (d.get('loops', 0) or 0) + (d.get('bridges', 0) or 0)
            ops = d.get('optops')
            avg = ('%d' % (ops // ntr)) if (ops and ntr) else '-'
            cells.append('%4s %6s %5s' % (ntr or '-', _fnum(ops), avg))
        w('%-16s | %s' % (r['name'], ' | '.join(cells)))

    # (3) memory footprint
    w('\n## (3) Memory footprint -- maximum resident set size (MB)')
    w('%-16s ' % 'benchmark' + ' '.join('%10s' % l for l in labels))
    for r in results:
        ro = r['rows']
        def mb(l):
            v = ro[l]['rss_kb']
            return '%.1f' % (v / 1024.0) if v else '-'
        w('%-16s %s' % (r['name'], ' '.join('%10s' % mb(l) for l in labels)))
    return '\n'.join(out)


def main(argv):
    global TIMEOUT
    out_json = out_csv = None
    iters_override = None
    tiers = list(ALL_TIERS)
    names = []
    i = 0
    while i < len(argv):
        a = argv[i]
        if a == '--json':
            out_json = argv[i + 1]; i += 2; continue
        if a == '--csv':
            out_csv = argv[i + 1]; i += 2; continue
        if a == '--iters':
            iters_override = int(argv[i + 1]); i += 2; continue
        if a == '--timeout':
            TIMEOUT = int(argv[i + 1]); i += 2; continue
        if a == '--tiers':
            sel = set(argv[i + 1].split(',')); i += 2
            tiers = [(l, t) for (l, t) in ALL_TIERS if t in sel]
            continue
        if a in ('-h', '--help'):
            sys.stdout.write(__doc__); return 0
        names.append(a); i += 1

    if not os.path.exists(BIN):
        sys.stderr.write('missing binary: %s (set TLA_BIN)\n' % BIN)
        return 1
    if 'interp' not in [l for l, _ in tiers]:
        tiers = [('interp', '0')] + tiers      # interp is the speedup baseline

    benches = BENCHMARKS
    if names:
        bymap = dict((n, (n, x, it)) for n, x, it in BENCHMARKS)
        benches = [bymap.get(n, (n, DEFAULT_SIZE[0], DEFAULT_SIZE[1]))
                   for n in names]
    if iters_override:
        benches = [(n, x, iters_override) for n, x, _ in benches]

    results = []
    for name, x, iters in benches:
        sys.stderr.write('[%s x=%s iters=%s]\n' % (name, x, iters))
        r = evaluate(name, x, iters, tiers)
        if r is not None:
            results.append(r)

    if not results:
        sys.stderr.write('no benchmarks evaluated\n')
        return 1

    print report(results, tiers)
    if out_json:
        with open(out_json, 'w') as f:
            json.dump(results, f, indent=2)
        sys.stderr.write('wrote %s\n' % out_json)
    if out_csv:
        with open(out_csv, 'w') as f:
            f.write('benchmark,x,tier,result,warm_s,cold_s,speedup,'
                    'trace_time_s,backend_time_s,loops,bridges,recorded_ops,'
                    'opt_ops,rss_kb,crashed\n')
            for r in results:
                for label, _ in tiers:
                    d = r['rows'][label]
                    f.write(','.join(str(v) for v in [
                        r['name'], r['x'], label, d.get('result'),
                        d.get('warm'), d.get('cold'), d.get('speedup'),
                        d.get('trace_t'), d.get('backend_t'),
                        d.get('loops'), d.get('bridges'), d.get('recorded'),
                        d.get('optops'), d.get('rss_kb'), d.get('crashed')]) + '\n')
        sys.stderr.write('wrote %s\n' % out_csv)
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
