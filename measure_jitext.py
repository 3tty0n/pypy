#!/usr/bin/env pypy3

import os
import subprocess
import argparse
import sys

from statistics import mean, variance
from datetime import datetime

from jitext_bench import *

this_dir = os.path.abspath(os.path.dirname(__file__))


def get_time():
    now = datetime.now()
    return now.strftime("%m%d%Y_%H%M")


def parse_args():
    parser = argparse.ArgumentParser(prog="Measuring the jit summary data")
    parser.add_argument("-n", "--number", type=int)
    parser.add_argument("-d", "--dir", type=str)
    args = parser.parse_args()
    return args.number, args.dir


def run_icbd(env, exe_path, arg=None):
    env["PYTHONPATH"] = "icbd"
    os.chdir("benchmarks/own/icbd")
    command = ["%s/%s" % (this_dir, exe_path)]
    if arg:
        command.extend(arg)
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
    subprocess.run(command, env=env, stdout=subprocess.DEVNULL)
    os.chdir(this_dir)


WARMUP_NUMBER = 2


def run(typ, mode=None):
    num, dirname = parse_args()
    if not dirname:
        dirname = "pypylogs_%s" % (get_time())
    if not os.path.exists(dirname):
        os.mkdir(dirname)

    bm_path = setup_bm_path(typ)
    benchmarks = setup_bms(typ)
    for bm in benchmarks:
        for exe_name, exe_path in COMMANDS:
            print("Running %s against %s..." % (exe_name, bm))
            for i in range(num + WARMUP_NUMBER):
                env = setup_env(typ)

                if i < WARMUP_NUMBER:
                    env["PYTHONDONTWRITEBYTECODE"] = "1"
                    env["CCACHE_DISABLE"] = "1"
                else:
                    env.pop("PYTHONDONTWRITEBYTECODE", None)
                    env.pop("CCACHE_DISABLE", None)
                    log_output = "%s/%s/%s_%s_%i.log" % (
                        this_dir,
                        dirname,
                        exe_name,
                        bm,
                        i + 1 - WARMUP_NUMBER,
                    )

                    if mode == "genext-stats":
                        env["PYPYLOG"] = "jit-genext-stats:%s" % (log_output)
                    else:
                        env["PYPYLOG"] = "jit-summary:%s" % (log_output)

                target_path = bm_path + "%s.py" % (bm)
                command = [exe_path, target_path]
                if bm == "bm_icbd":
                    run_icbd(env, exe_path)
                else:
                    if bm == "bm_genshi":
                        command.append("--benchmark=xml")
                    elif bm == "bm_sympy":
                        command.append("--benchmark=str")
                    subprocess.run(command, env=env, stdout=subprocess.DEVNULL)


if __name__ == "__main__":
    run('own')
