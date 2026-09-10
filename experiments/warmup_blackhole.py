import argparse
import os
import re
import statistics
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BENCHMARKS = os.environ.get('BENCHMARKS', os.path.join(ROOT, 'benchmarks'))
OWN = os.path.join(BENCHMARKS, 'own')
UNLADEN = os.path.join(BENCHMARKS, 'unladen_swallow', 'performance')
DEFAULT_BENCHES = ['eparse', 'go', 'pyxl_bench', 'deltablue',
                   'raytrace-simple', 'bm_spambayes']


def pythonpath():
    parts = [os.path.join(BENCHMARKS, 'lib', 'sqlalchemy', 'lib')]
    for lib in ('lib', 'unladen_swallow/lib'):
        d = os.path.join(BENCHMARKS, lib)
        if os.path.isdir(d):
            parts += [os.path.join(d, x) for x in sorted(os.listdir(d))]
    return ':'.join(parts)


def summary_value(text, name):
    m = re.search(r'^%s:\s+(\d+)\s+([\d.]+)' % re.escape(name), text, re.M)
    return (int(m.group(1)), float(m.group(2))) if m else (0, 0.0)


def run(binary, bench, n, eagerness, outdir):
    script = bench + '.py'
    cwd = OWN if os.path.exists(os.path.join(OWN, script)) else UNLADEN
    log = os.path.join(outdir, '%s-e%d.log' % (bench, eagerness))
    env = dict(os.environ, PYTHONPATH=pythonpath(), PYPY_GC_NURSERY='1G',
               PYPYLOG='jit-summary:' + log)
    cmd = [binary, '--jit', 'trace_eagerness=%d' % eagerness, script,
           '-n', str(n)]
    out = subprocess.run(cmd, cwd=cwd, env=env, capture_output=True,
                         text=True).stdout
    times = [float(x) for x in out.split() if re.match(r'^[\d.]+$', x)]
    if len(times) < 4:
        return None
    steady = statistics.median(times[len(times) // 2:])
    excess = sum(max(t - steady, 0.0) for t in times)
    text = open(log).read()
    compile_s = sum(summary_value(text, k)[1]
                    for k in ('Tracing', 'Optimizing', 'Backend'))
    resumes, bh = summary_value(text, 'Blackhole')
    m = re.search(r'^Total # of bridges:\s+(\d+)', text, re.M)
    return dict(excess=excess, steady=steady, compile=compile_s,
                blackhole=bh, resumes=resumes,
                bridges=int(m.group(1)) if m else 0, total=sum(times))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('binary')
    ap.add_argument('-n', type=int, default=50)
    ap.add_argument('-e', default='1,200,0', help='trace_eagerness values')
    ap.add_argument('--bench', default=','.join(DEFAULT_BENCHES))
    ap.add_argument('-o', default='warmup-blackhole')
    a = ap.parse_args()
    os.makedirs(a.o, exist_ok=True)
    eager = [int(x) for x in a.e.split(',')]
    print('%-16s %5s %7s %7s %8s %8s %7s %7s' % (
        'bench', 'eager', 'total', 'excess', 'compile', 'blackh',
        'resumes', 'bridges'))
    for bench in a.bench.split(','):
        for e in eager:
            r = run(a.binary, bench, a.n, e, a.o)
            if r is None:
                print('%-16s %5d  failed' % (bench, e))
                continue
            print('%-16s %5d %7.2f %7.2f %8.3f %8.3f %7d %7d' % (
                bench, e, r['total'], r['excess'], r['compile'],
                r['blackhole'], r['resumes'], r['bridges']))
        sys.stdout.flush()


if __name__ == '__main__':
    main()
