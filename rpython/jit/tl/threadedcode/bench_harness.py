#!/usr/bin/env python
"""Run TLA micro-benchmarks (lang/bench/*.tla.py) with warmup and timing."""
import os
import sys
import py

py.path.local(__file__)

from rpython.rlib import jit
from rpython.rlib.jit import set_param, set_param_to_default
from rpython.rlib.rtime import time
from rpython.jit.tl.threadedcode import tla
from rpython.jit.tl.threadedcode.bytecode import assemble, Bytecode

BENCH_DIR = os.path.join(os.path.dirname(__file__), 'lang', 'bench')

EXEC_MODES = ('interpreter', 'tier1', 'tier2')


def configure_jit_exec_mode(mode):
    """Select JIT drivers for interpreter-only vs tier-1 vs tier-2 runs."""
    if mode == 'interpreter':
        set_param(tla.tier1driver, 'threshold', -1)
        set_param(tla.tier1driver, 'function_threshold', -1)
        set_param(tla.tier2driver, 'threshold', -1)
        set_param(tla.tier2driver, 'function_threshold', -1)
    elif mode == 'tier1':
        set_param_to_default(tla.tier1driver, 'threshold')
        set_param_to_default(tla.tier1driver, 'function_threshold')
        set_param(tla.tier2driver, 'threshold', -1)
        set_param(tla.tier2driver, 'function_threshold', -1)
    elif mode == 'tier2':
        set_param(tla.tier1driver, 'threshold', -1)
        set_param(tla.tier1driver, 'function_threshold', -1)
        set_param_to_default(tla.tier2driver, 'threshold')
        set_param_to_default(tla.tier2driver, 'function_threshold')
    else:
        raise ValueError(mode)


def run_bench_timed(bytecode, x, tier, warmup, repeat):
    for _ in range(warmup):
        tla.run(bytecode, tla.W_IntObject(x), tier=tier)
    times = []
    w_res = None
    for _ in range(repeat):
        t0 = time()
        w_res = tla.run(bytecode, tla.W_IntObject(x), tier=tier)
        times.append(time() - t0)
    mean = sum(times) / len(times)
    last = w_res.getrepr() if w_res is not None else 'None'
    return mean, last

# name -> default W_IntObject int value for programs that use the initial stack arg
DEFAULT_ARG = {
    'tb_single_hotloop': 500000,
    # Nested pattern is much heavier per outer step than the single loop.
    'tb_backjump_nested': 5000,
    'tb_sum_tail_bench': 0,
    'tb_fact_bench': 0,
    'tb_gcd_bench': 0,
}

ALL_BENCHES = sorted(DEFAULT_ARG.keys())


def usage(msg=None):
    if msg:
        print >> sys.stderr, msg
    print >> sys.stderr, (
        'Usage: bench_harness.py [options] [bench_name ...]\n'
        '  --tier N       interpreter tier (default 2)\n'
        '  --warmup N     iterations before timing (default 2)\n'
        '  --repeat N     timed iterations per benchmark (default 5)\n'
        '  --x INT        override initial stack int for all benches\n'
        '  --jit ARG      pass to jit.set_user_param (repeatable)\n'
        '  --tla-inline-depth N   set global inline cap (0=off, 1=enable hints)\n'
        '  --mode MODE    interpreter | tier1 | tier2 (JIT off / tier-1 interp / tier-2)\n'
        '  --compare      run all three modes and print comparison (same --warmup/--repeat)\n'
        '  --list         print benchmark names and exit\n'
        '  With no names, runs: %s' % ' '.join(ALL_BENCHES))


def load_bench(path):
    mydict = {}
    execfile(path, mydict)
    return mydict['code']


def main(argv):
    tier = 2
    exec_mode = None
    do_compare = False
    warmup = 2
    repeat = 5
    x_override = None
    i = 1
    while i < len(argv):
        a = argv[i]
        if a == '--compare':
            do_compare = True
            i += 1
        elif a == '--mode':
            exec_mode = argv[i + 1]
            if exec_mode not in EXEC_MODES:
                usage('unknown --mode %r (use %s)' % (exec_mode, '|'.join(EXEC_MODES)))
                return 2
            i += 2
        elif a == '--tier':
            tier = int(argv[i + 1])
            i += 2
        elif a == '--warmup':
            warmup = int(argv[i + 1])
            i += 2
        elif a == '--repeat':
            repeat = int(argv[i + 1])
            i += 2
        elif a == '--x':
            x_override = int(argv[i + 1])
            i += 2
        elif a == '--jit':
            jit.set_user_param(None, argv[i + 1])
            i += 2
        elif a == '--tla-inline-depth':
            tla.set_global_inline_cap(int(argv[i + 1]))
            i += 2
        elif a == '--list':
            for n in ALL_BENCHES:
                print n
            return 0
        elif a.startswith('-'):
            usage('unknown option: %s' % a)
            return 2
        else:
            break

    names = argv[i:]
    if not names:
        names = list(ALL_BENCHES)

    for name in names:
        if name not in DEFAULT_ARG:
            usage('unknown benchmark: %r (try --list)' % name)
            return 2

    if do_compare and exec_mode is not None:
        usage('use either --compare or --mode, not both')
        return 2
    if exec_mode is not None:
        configure_jit_exec_mode(exec_mode)
        tier = 1 if exec_mode in ('interpreter', 'tier1') else 2

    if do_compare:
        print '# compare: interpreter=interp loop JIT off, tier1=interp+JIT, tier2=threaded+JIT'
        print '# warmup=%d repeat=%d' % (warmup, repeat)
        for name in names:
            path = os.path.join(BENCH_DIR, name + '.tla.py')
            if not os.path.isfile(path):
                print >> sys.stderr, 'missing file:', path
                return 2
            x = x_override if x_override is not None else DEFAULT_ARG[name]
            bytecode = Bytecode(assemble(load_bench(path)))
            parts = [name]
            for mode in EXEC_MODES:
                configure_jit_exec_mode(mode)
                tr = 1 if mode in ('interpreter', 'tier1') else 2
                try:
                    mean, last = run_bench_timed(bytecode, x, tr, warmup, repeat)
                    parts.append('%.4g' % mean)
                except Exception as e:
                    parts.append('ERR:%s' % e.__class__.__name__)
            print '\t'.join(parts)
        return 0

    for name in names:
        path = os.path.join(BENCH_DIR, name + '.tla.py')
        if not os.path.isfile(path):
            print >> sys.stderr, 'missing file:', path
            return 2
        x = x_override if x_override is not None else DEFAULT_ARG[name]
        code = load_bench(path)
        bytecode = Bytecode(assemble(code))

        if exec_mode is not None:
            mean, last = run_bench_timed(bytecode, x, tier, warmup, repeat)
            print '%s x=%s mode=%s tier=%s mean=%.6g s (n=%d) last=%s' % (
                name, x, exec_mode, tier, mean, repeat, last)
        else:
            mean, last = run_bench_timed(bytecode, x, tier, warmup, repeat)
            print '%s x=%s tier=%s mean=%.6g s (n=%d) last=%s' % (
                name, x, tier, mean, repeat, last)

    return 0


def target(driver, args):
    return main


if __name__ == '__main__':
    sys.exit(main(sys.argv))
