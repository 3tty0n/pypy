"""Build and measure baseline/offline-PE native TLA binaries."""

from __future__ import print_function

import argparse
import os
import re
import subprocess
import sys
import tempfile
import time

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__),
                                    "..", "..", "..", ".."))
TARGET = os.path.join(ROOT, "rpython/jit/tl/tla/targettla_offline_bench.py")
RPYTHON = os.path.join(ROOT, "rpython/bin/rpython")


def build(output, offline_pe):
    env = os.environ.copy()
    env["TLA_OFFLINE_PE"] = "1" if offline_pe else "0"
    output_dir = os.path.dirname(output)
    command = [sys.executable, RPYTHON, "--batch", "--opt=jit",
               "--output", os.path.basename(output),
               TARGET]
    subprocess.check_call(command, cwd=output_dir, env=env)


def run(binary, bytecode, value):
    log = tempfile.NamedTemporaryFile(prefix="tla-jit-", delete=False)
    log.close()
    env = os.environ.copy()
    env["PYPYLOG"] = "jit-summary:%s" % log.name
    started = time.time()
    output = subprocess.check_output(
        [binary, bytecode, str(value)], env=env)
    elapsed = time.time() - started
    data = open(log.name).read()
    os.unlink(log.name)
    recorded_match = re.search(r"recorded ops:\s*(\d+)", data)
    raw_match = re.search(r"^ops:\s*(\d+)", data, re.MULTILINE)
    tracing_match = re.search(
        r"^Tracing:\s*\d+\s+([0-9.]+)", data, re.MULTILINE)
    recorded = int(recorded_match.group(1)) if recorded_match else -1
    raw_ops = int(raw_match.group(1)) if raw_match else -1
    tracing = float(tracing_match.group(1)) if tracing_match else -1.0
    return elapsed, tracing, raw_ops, recorded, output.strip()


def median(values):
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) & 1:
        return ordered[middle]
    return (ordered[middle - 1] + ordered[middle]) / 2.0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--value", type=int, default=1000000)
    parser.add_argument("--build-dir", default="./.tla-offline-pe-native")
    parser.add_argument("--skip-build", action="store_true")
    args = parser.parse_args(argv)
    if not os.path.isdir(args.build_dir):
        os.makedirs(args.build_dir)
    baseline = os.path.join(args.build_dir, "tla-baseline")
    offline_pe = os.path.join(args.build_dir, "tla-offline-pe")
    bytecode = os.path.join(args.build_dir, "countdown.tla")
    with open(bytecode, "wb") as stream:
        stream.write(''.join(chr(op) for op in [0, 1, 6, 5, 4, 0, 3]))
    if not args.skip_build:
        build(baseline, False)
        build(offline_pe, True)
    baseline_times = []
    offline_times = []
    baseline_tracing = []
    offline_tracing = []
    baseline_raw_ops = []
    offline_raw_ops = []
    baseline_ops = []
    offline_ops = []
    for index in range(args.runs):
        before = run(baseline, bytecode, args.value)
        after = run(offline_pe, bytecode, args.value)
        baseline_times.append(before[0])
        offline_times.append(after[0])
        baseline_tracing.append(before[1])
        offline_tracing.append(after[1])
        baseline_raw_ops.append(before[2])
        offline_raw_ops.append(after[2])
        baseline_ops.append(before[3])
        offline_ops.append(after[3])
        print("run %d: baseline %.6fs (trace %.6fs, %d->%d ops); "
              "offline %.6fs (trace %.6fs, %d->%d ops)" %
              (index + 1, before[0], before[1], before[2], before[3],
               after[0], after[1], after[2], after[3]))
    baseline_median = median(baseline_times)
    offline_median = median(offline_times)
    reduction = 100.0 * (baseline_median - offline_median) / baseline_median
    print("median: baseline %.6fs; offline %.6fs; reduction %.1f%%" %
          (baseline_median, offline_median, reduction))
    baseline_trace_median = median(baseline_tracing)
    offline_trace_median = median(offline_tracing)
    trace_reduction = (100.0 * (baseline_trace_median - offline_trace_median) /
                       baseline_trace_median)
    print("tracing median: baseline %.6fs; offline %.6fs; reduction %.1f%%" %
          (baseline_trace_median, offline_trace_median, trace_reduction))
    print("raw ops: baseline %s; offline %s" %
          (baseline_raw_ops, offline_raw_ops))
    print("optimized ops: baseline %s; offline %s" %
          (baseline_ops, offline_ops))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
