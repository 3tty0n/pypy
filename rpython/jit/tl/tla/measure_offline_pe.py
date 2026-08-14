"""Measure TLA meta-tracing work with and without offline PE.

Run from the repository root with Python 2::

    python2 rpython/jit/tl/tla/measure_offline_pe.py --runs 5 --value 42

The profiler's TRACING counter measures time spent tracing, not translation
or test-process startup.  Recorded operations are the more deterministic
metric; timing should be compared using the reported median.
"""

from __future__ import print_function

import argparse
import os
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__),
                                         "..", "..", "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from rpython.jit.backend.llgraph.runner import LLGraphCPU
from rpython.jit.metainterp import pyjitpl
from rpython.jit.metainterp.jitprof import Profiler
from rpython.jit.metainterp.warmspot import ll_meta_interp
from rpython.jit.tl.tla import offline, tla
from rpython.rlib.jit import Counters


CODE = [
    tla.CONST_INT, 1,
    tla.SUB,
    tla.DUP,
    tla.JUMP_IF, 0,
    tla.RETURN,
]


def assemble(code):
    return ''.join(chr(op) for op in code)


BYTECODE = assemble(CODE)


def interpret(value):
    # Assemble inside the entry function, as a normal interpreter frontend
    # would.  Keeping the string as a literal makes the green bytecode appear
    # constant before warmspot splits the portal graph.
    bytecode = ''.join([chr(op) for op in CODE])
    result = tla.run(bytecode, tla.W_IntObject(value))
    return result.intvalue


def measure_once(value, use_offline_pe):
    options = {
        "CPUClass": LLGraphCPU,
        "ProfilerClass": Profiler,
        "listops": True,
    }
    if use_offline_pe:
        def install(codewriter, jitdriver_sd, translator):
            return offline.lower_and_install(
                codewriter, jitdriver_sd, translator, BYTECODE)

        options["pe_linked_setup"] = install

    result = ll_meta_interp(interpret, [value], **options)
    if result != 0:
        raise AssertionError("countdown returned %r" % (result,))
    profiler = pyjitpl._warmrunnerdesc.metainterp_sd.profiler
    return (profiler.get_times(Counters.TRACING),
            profiler.get_counter(Counters.RECORDED_OPS))


def median(values):
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[middle]
    return (ordered[middle - 1] + ordered[middle]) / 2.0


def reduction(before, after):
    if not before:
        return 0.0
    return 100.0 * (before - after) / before


def measure(runs, value):
    baseline = []
    offline_pe = []
    for index in range(runs):
        baseline.append(measure_once(value, False))
        offline_pe.append(measure_once(value, True))
        print("run %d: baseline %.6fs/%d ops, offline %.6fs/%d ops" %
              (index + 1, baseline[-1][0], baseline[-1][1],
               offline_pe[-1][0], offline_pe[-1][1]))

    baseline_time = median([item[0] for item in baseline])
    offline_time = median([item[0] for item in offline_pe])
    baseline_ops = median([item[1] for item in baseline])
    offline_ops = median([item[1] for item in offline_pe])
    print("\nmedian baseline : %.6fs, %g recorded ops" %
          (baseline_time, baseline_ops))
    print("median offline  : %.6fs, %g recorded ops" %
          (offline_time, offline_ops))
    print("reduction       : %.1f%% tracing time, %.1f%% recorded ops" %
          (reduction(baseline_time, offline_time),
           reduction(baseline_ops, offline_ops)))
    return baseline, offline_pe


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--value", type=int, default=42)
    args = parser.parse_args(argv)
    if args.runs < 1:
        parser.error("--runs must be at least 1")
    measure(args.runs, args.value)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
