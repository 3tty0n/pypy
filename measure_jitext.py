#!/usr/bin/env pypy3
"""Benchmark script for measuring JIT compilation performance.

Designed to minimize noise and systematic bias so that real performance
differences (especially pypy-jit-ext-c improvements) are visible.

Key measures:
- CPU affinity: pin each run to a single core (--cpu)
- Cooldown: sleep between runs to avoid thermal throttling (--cooldown)
- Randomized order: shuffle binary execution order each iteration
- Drop caches: optionally drop OS file caches between runs (--drop-caches)
- Process priority: run benchmarks at high priority via nice (--nice)

Recommended pre-run setup (as root):
  # Fix CPU frequency to avoid turbo boost variance
  cpupower frequency-set -g performance
  # Or per-core:
  echo performance > /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor

Usage:
  python measure_jitext.py -n 5 -b all --cpu 2 --cooldown 3
"""

import os
import subprocess
import argparse
import sys
import random
import time

from datetime import datetime

from jitext_bench import *

this_dir = os.path.abspath(os.path.dirname(__file__))


def get_time():
    now = datetime.now()
    return now.strftime("%m%d%Y_%H%M")


def check_system():
    """Warn about system configuration issues that may add noise."""
    warnings = []

    # Check CPU governor
    try:
        gov_path = "/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor"
        if os.path.exists(gov_path):
            with open(gov_path) as f:
                gov = f.read().strip()
            if gov != "performance":
                warnings.append(
                    "CPU governor is '%s', not 'performance'. "
                    "Run: sudo cpupower frequency-set -g performance" % gov)
    except (IOError, OSError):
        pass

    # Check turbo boost
    for path in [
        "/sys/devices/system/cpu/intel_pstate/no_turbo",
        "/sys/devices/system/cpu/cpufreq/boost",
    ]:
        try:
            if os.path.exists(path):
                with open(path) as f:
                    val = f.read().strip()
                if "no_turbo" in path and val == "0":
                    warnings.append(
                        "Intel turbo boost is ON. For stable measurements: "
                        "echo 1 | sudo tee %s" % path)
                elif "boost" in path and val == "1":
                    warnings.append(
                        "CPU boost is ON. For stable measurements: "
                        "echo 0 | sudo tee %s" % path)
        except (IOError, OSError):
            pass

    if warnings:
        print("=" * 60)
        print("BENCHMARK WARNINGS:")
        for w in warnings:
            print("  - %s" % w)
        print("=" * 60)
        print()


def drop_file_caches():
    """Drop OS file caches. Requires root or appropriate permissions."""
    try:
        subprocess.run(
            ["sudo", "-n", "sh", "-c",
             "sync; echo 3 > /proc/sys/vm/drop_caches"],
            timeout=5, capture_output=True)
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        pass  # best-effort


def parse_args():
    parser = argparse.ArgumentParser(
        prog="measure_jitext",
        description="Measure JIT compilation performance (jit-summary data)")
    parser.add_argument("-n", "--number", type=int, default=5,
                        help="Number of measurement iterations (default: 5)")
    parser.add_argument("-d", "--dir", type=str,
                        help="Output directory for logs")
    parser.add_argument("-b", "--benchmark", type=str, default="all",
                        help="Benchmark suite: own-micro, own-macro, unladen, "
                             "macro-all, all (default: all)")
    parser.add_argument("--mode", type=str, default=None,
                        help="Log mode: genext-stats or jit-summary (default)")
    parser.add_argument("--cpu", type=int, default=None,
                        help="Pin processes to this CPU core via taskset")
    parser.add_argument("--cooldown", type=float, default=2.0,
                        help="Seconds to sleep between runs (default: 2.0)")
    parser.add_argument("--nice", type=int, default=None,
                        help="Nice priority (e.g. -20 for highest)")
    parser.add_argument("--drop-caches", action="store_true",
                        help="Drop OS file caches between runs (needs sudo)")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed for reproducible ordering")
    parser.add_argument("--no-check", action="store_true",
                        help="Skip system configuration check")
    parser.add_argument("--timeout", type=int, default=300,
                        help="Per-run timeout in seconds (default: 300)")
    args = parser.parse_args()
    return args


def build_command(exe_path, cpu=None, nice_val=None):
    """Wrap an executable path with taskset and/or nice."""
    prefix = []
    if nice_val is not None:
        prefix.extend(["nice", "-n", str(nice_val)])
    if cpu is not None:
        prefix.extend(["taskset", "-c", str(cpu)])
    return prefix + [exe_path]


def run_icbd(env, exe_path, cpu=None, nice_val=None, timeout=None):
    """Run the ICBD benchmark with exception-safe directory handling."""
    env["PYTHONPATH"] = "icbd"
    orig_dir = os.getcwd()
    try:
        os.chdir("benchmarks/own/icbd")
        command = build_command("%s/%s" % (this_dir, exe_path), cpu, nice_val)
        command.extend([
            "-m", "icbd.type_analyzer.analyze_all",
            "-I", "stdlib/python2.5_tiny",
            "-I", ".",
            "-E", "icbd/type_analyzer/tests",
            "-E", "icbd/compiler/benchmarks",
            "-E", "icbd/compiler/tests",
            "-I", "stdlib/type_mocks",
            "-n", "icbd",
        ])
        subprocess.run(command, env=env, stdout=subprocess.DEVNULL,
                       stderr=subprocess.PIPE, timeout=timeout, check=True)
    finally:
        os.chdir(orig_dir)


def run(num, dirname, typ, args):
    mode = args.mode
    cpu = args.cpu
    nice_val = args.nice
    cooldown = args.cooldown
    timeout = getattr(args, 'timeout', None)

    bm_path = setup_bm_path(typ)
    benchmarks = setup_bms(typ)

    for bm in benchmarks:
        print("\n>>> Benchmark: %s (%s)" % (bm, typ))

        for i in range(1, num + 1):
            # Shuffle execution order each iteration to eliminate
            # systematic bias from fixed ordering (thermal throttling,
            # cache warming, memory fragmentation)
            commands = list(COMMANDS)
            random.shuffle(commands)

            for cmd_idx, (exe_name, exe_path) in enumerate(commands):
                # Cooldown between runs (skip before the very first)
                if i > 1 or cmd_idx > 0:
                    if cooldown > 0:
                        time.sleep(cooldown)

                # Optionally drop file caches for equal footing
                if args.drop_caches:
                    drop_file_caches()

                print("  [%d/%d] %s" % (i, num, exe_name))
                env = setup_env(typ)
                env["PYTHONDONTWRITEBYTECODE"] = "1"
                log_output = "%s/%s/%s_%s_%i.log" % (
                    this_dir, dirname, exe_name, bm, i)

                if mode == "genext-stats":
                    env["PYPYLOG"] = "jit-genext:%s" % log_output
                else:
                    env["PYPYLOG"] = "jit-summary:%s" % log_output

                target_path = bm_path + "%s.py" % bm

                try:
                    if bm == "bm_icbd":
                        run_icbd(env, exe_path, cpu, nice_val, timeout=timeout)
                    else:
                        command = build_command(exe_path, cpu, nice_val)
                        command.append(target_path)
                        if bm == "bm_genshi":
                            command.append("--benchmark=xml")
                        elif bm == "bm_sympy":
                            command.append("--benchmark=str")
                        command.extend(["-n", "1"])
                        subprocess.run(
                            command, env=env,
                            stdout=subprocess.DEVNULL,
                            stderr=subprocess.PIPE,
                            timeout=timeout, check=True)
                except subprocess.TimeoutExpired:
                    print("    ERROR: timeout after %ss — data lost" % timeout)
                except subprocess.CalledProcessError as e:
                    print("    ERROR: exit code %d — data may be incomplete" % e.returncode)
                    if e.stderr:
                        print("    stderr: %s" % e.stderr.decode('utf-8', 'replace')[:200])
                except FileNotFoundError:
                    print("    ERROR: binary not found: %s" % exe_path)


if __name__ == "__main__":
    args = parse_args()

    if not args.no_check:
        check_system()

    if args.seed is not None:
        random.seed(args.seed)

    dirname = args.dir
    if not dirname:
        dirname = "pypylogs_%s" % get_time()
    if not os.path.exists(dirname):
        os.makedirs(dirname)

    typ = args.benchmark
    if typ == "all":
        for t in ("own-micro", "own-macro", "unladen"):
            run(args.number, dirname, t, args)
    elif typ == "macro-all":
        for t in ("own-macro", "unladen"):
            run(args.number, dirname, t, args)
    else:
        run(args.number, dirname, typ, args)

    print("\nDone. Results in %s/" % dirname)
