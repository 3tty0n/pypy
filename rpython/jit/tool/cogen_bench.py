#!/usr/bin/env python2
"""Run benchmarks/runner.py for a warmup/steady comparison of two pypy-c.

    cogen_bench.py OLD NEW -o result.json [-b a,b,c] [--fast] [--reverse]

Every benchmark subprocess gets PYPY_GC_NURSERY=1G; binaries are made
absolute (runner.py chdirs).  --reverse also runs with the roles swapped
into result.reverse.json, the pair plot_cogen_comparison.py takes.
Plot a single result with plot_cogen_warmup.py.
"""

import argparse
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
RUNNER = os.path.join(ROOT, "benchmarks", "runner.py")

# BENCHMARK_DEFAULT minus builds (translate, cpython_doc), twisted, and
# the sympy/genshi variants that only duplicate their siblings.
DEFAULT = ("ai,bm_chameleon,bm_dulwich_log,bm_krakatau,bm_mako,bm_mdp,"
           "chaos,crypto_pyaes,deltablue,django,eparse,genshi_xml,go,"
           "html5lib,json_bench,meteor-contest,pyflate-fast,pyxl_bench,"
           "raytrace-simple,richards,scimark_fft,scimark_lu,"
           "scimark_montecarlo,scimark_sor,scimark_sparsematmult,"
           "spambayes,spectral-norm,spitfire2,sqlalchemy_declarative,"
           "sqlalchemy_imperative,sqlitesynth,sympy_expand,telco")


def run(baseline, changed, output, benchmarks, args, fast):
    cmd = [sys.executable, RUNNER, "--full-store",
           "--inherit-env=PATH,PYPY_GC_NURSERY",
           "-b", benchmarks, "-o", output,
           "--baseline", baseline, "-c", changed]
    if args:
        cmd += ["-a", args]
    if fast:
        cmd.append("--fast")
    env = dict(os.environ, PYPY_GC_NURSERY="1G")
    subprocess.check_call(cmd, cwd=os.path.dirname(RUNNER), env=env)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("old")
    parser.add_argument("new")
    parser.add_argument("-o", "--output", default="result.json")
    parser.add_argument("-b", "--benchmarks", default=DEFAULT)
    parser.add_argument("-a", "--args", default="",
                        help="runner.py -a: 'OLDARGS,NEWARGS'")
    parser.add_argument("--fast", action="store_true")
    parser.add_argument("--reverse", action="store_true",
                        help="also run with old/new swapped")
    opts = parser.parse_args()
    old, new = os.path.abspath(opts.old), os.path.abspath(opts.new)
    output = os.path.abspath(opts.output)
    run(old, new, output, opts.benchmarks, opts.args, opts.fast)
    if opts.reverse:
        base, ext = os.path.splitext(output)
        args = opts.args
        if "," in args:
            a, b = args.split(",", 1)
            args = b + "," + a
        run(new, old, base + ".reverse" + ext, opts.benchmarks, args,
            opts.fast)


if __name__ == "__main__":
    main()
