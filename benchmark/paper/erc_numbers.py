#!/usr/bin/env python3
"""The five MetaTensor numbers in the LIFELINE proposal, recomputed from the
committed result files, each with where it comes from.

    erc_numbers.py [RESULTS_ROOT]

For every number: the stage and command that produced it, the result file,
the rows it reads, the formula, the binary those rows carry, the source
commit that binary was built from, and the last commit that wrote the file
read (so the content this script reads is the content that commit holds).
A number that does not come out as the proposal states it is printed as a
MISMATCH and the exit status is non-zero, so the proposal cannot drift from
the data without this saying so.
"""
import csv
import math
import os
import statistics
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, '..', 'results', 'luchkylilac-rtx3090')

# The deployed-checkpoint population (run_models.sh POPULATION).
POPULATION = ["distilgpt2", "gpt2", "gpt2-medium", "smollm2-135m",
              "smollm2-360m", "smollm2-1.7b", "qwen2.5-0.5b", "bert-mini",
              "bert-base", "resnet18-b1", "resnet18-b8", "mixer_b16",
              "deit-tiny", "vit-base"]

# Binary digest -> source commit, and where that pairing is recorded.
SOURCES = {
    'ddc8008cdeaf': ('1bdc7e63fa', 'results commit 7bf624f1df states it'),
    'transition': ('3581d3a8a9', 'commit 3581d3a8a9 built the drain knob; '
                   'the phases are recorded in c8d02109bf'),
}


def rows(run, name):
    return list(csv.DictReader(open(os.path.join(ROOT, run, name)),
                               delimiter='\t'))


def med(xs):
    return statistics.median(xs)


def recorded_in(run, name):
    path = os.path.join('benchmark', 'results', 'luchkylilac-rtx3090', run,
                        name)
    for p in (path, os.path.join(os.path.dirname(path), 'data.tar.gz')):
        out = subprocess.run(['git', 'log', '--format=%h', '--diff-filter=AM',
                              '--', p], capture_output=True, text=True,
                             cwd=os.path.join(HERE, '..', '..')).stdout.split()
        if out:
            return out[0] + ('' if p == path else ' (in data.tar.gz)')
    return 'untracked'


def by(rs, *keys):
    g = {}
    for r in rs:
        g.setdefault(tuple(r[k] for k in keys), []).append(r)
    return g


def n_deferred():
    rs = rows('paper-2026-09-21', 'lazy.tsv')
    g = by(rs, 'model', 'system')
    d = [float(r['steady_us']) for r in g[('bert-base', 'deferred')]]
    v = [float(r['steady_us']) for r in g[('bert-base', 'virtual')]]
    ld = med([float(r['launches_per_iter'])
              for r in g[('bert-base', 'deferred')]])
    lv = med([float(r['launches_per_iter'])
              for r in g[('bert-base', 'virtual')]])
    return (med(d) / med(v), 1.63,
            'bench.sh lazy, bert-base: median steady_us deferred / virtual '
            '(%d rounds each); launches %.1f vs %.1f' % (len(d), ld, lv),
            'paper-2026-09-21', 'lazy.tsv',
            sorted(set(r['binary'] for r in rs if r['model'] == 'bert-base')))


def n_early_exit():
    rs = rows('paper-2026-09-19', 'control.tsv')
    g = by(rs, 'system', 'regime')
    tc = med([float(r['p50_us']) for r in g[('torch-compile', 'varying')]])
    ours = med([float(r['p50_us']) for r in g[('ours', 'varying')]])
    return (tc / ours, 1.42,
            'bench.sh control, distilgpt2 early exit, regime varying: median '
            'p50 per-forward torch-compile / ours', 'paper-2026-09-19',
            'control.tsv', sorted(set(r['binary'] for r in rs
                                      if r['system'] == 'ours')))


def n_tail():
    rs = rows('paper-2026-09-21', 'control.tsv')
    g = by(rs, 'system', 'regime')
    ours = med([float(r['p95_us']) for r in g[('ours', 'stable')]])
    tc = med([float(r['p95_us']) for r in g[('torch-compile', 'stable')]])
    return (ours / tc, 3.0,
            'bench.sh control, distilgpt2 early exit, regime stable: median '
            'p95 ours / torch-compile ("3x worse tail")', 'paper-2026-09-21',
            'control.tsv', sorted(set(r['binary'] for r in rs
                                      if r['system'] == 'ours')))


def n_guard():
    rs = rows('paper-2026-09-22', 'transition_phases.tsv')
    g = {(r['arm'], r['interior_reads']): float(r['launches']) for r in rs}
    before = g[('drain', 'none')] / g[('direct', 'none')]
    after = g[('drain', 'half')] / g[('direct', 'half')]
    return ((before, after), (3.86, 1.48),
            'bench.sh transition phases, 400 iterations: launches drain / '
            'direct with no interior consumer (%d/%d), then with the '
            'consumer arriving after a guard at 200 (%d/%d)'
            % (g[('drain', 'none')], g[('direct', 'none')],
               g[('drain', 'half')], g[('direct', 'half')]),
            'paper-2026-09-22', 'transition_phases.tsv', ['transition'])


def n_xla():
    rs = rows('paper-2026-09-21', 'models.tsv')
    g = by(rs, 'model', 'system')
    ratios = []
    for m in POPULATION:
        if (m, 'ours') in g and (m, 'jax') in g:
            o = med([float(r['steady_us']) for r in g[(m, 'ours')]])
            j = med([float(r['steady_us']) for r in g[(m, 'jax')]])
            ratios.append(j / o)
    gm = math.exp(sum(math.log(x) for x in ratios) / len(ratios))
    return (gm, 0.82,
            'bench.sh models, the %d-model population with a JAX row: '
            'geometric mean of median steady_us jax / ours (<1: XLA faster)'
            % len(ratios), 'paper-2026-09-21', 'models.tsv',
            sorted(set(r['binary'] for r in rs if r['system'] == 'ours')))


def main(argv):
    global ROOT
    if len(argv) > 1:
        ROOT = argv[1]
    bad = 0
    for fn in (n_deferred, n_early_exit, n_tail, n_guard, n_xla):
        got, stated, what, run, name, binaries = fn()
        gots = got if isinstance(got, tuple) else (got,)
        states = stated if isinstance(stated, tuple) else (stated,)
        # ok: the proposal's figure is the recomputed value rounded to two
        # places (for "3x", to a whole number).  TRUNC: it is the value cut,
        # not rounded - the data supports it but the rounding should be
        # fixed.  Anything else is a MISMATCH.
        status = 'ok'
        for a, b in zip(gots, states):
            if b == 3.0:
                good = round(a) == 3
            else:
                good = abs(round(a, 2) - b) < 1e-9
            if not good:
                if abs(math.floor(a * 100) / 100.0 - b) < 1e-9:
                    status = 'TRUNC'
                else:
                    status = 'MISMATCH'
        bad += status == 'MISMATCH'
        print('%-8s  %s  (proposal: %s)' % (
            status,
            ' -> '.join('%.3fx' % a for a in gots),
            ' -> '.join('%.2fx' % b for b in states)))
        print('    what:    %s' % what)
        print('    file:    benchmark/results/luchkylilac-rtx3090/%s/%s '
              '(recorded in %s)' % (run, name, recorded_in(run, name)))
        for b in binaries:
            src, how = SOURCES.get(b, ('?', 'not recorded'))
            print('    binary:  %s  source %s  (%s)' % (b, src, how))
        print('    branch:  pypytensor (origin, tensorpypy)\n')
    return 1 if bad else 0


if __name__ == '__main__':
    sys.exit(main(sys.argv))
