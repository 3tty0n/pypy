#!/usr/bin/env pypy3

import os
import subprocess
import argparse
import sys

from statistics import mean, variance
from datetime import datetime

from bm import BENCHMARKS, BENCHMARKS_UNLADEN_SWALLOW, COMMANDS

this_dir = os.path.abspath(os.path.dirname(__file__))


def get_time():
    now = datetime.now()
    return now.strftime("%m%d%Y_%H%M")


def parse_args():
    parser = argparse.ArgumentParser(prog="Measuring the jit summary data")
    parser.add_argument("-n", "--number", type=int)
    parser.add_argument("-d", "--dir", type=str)
    args = parser.parse_args()
    return args.number


def setup_env():
    env = os.environ.copy()
    env["PYTHONPATH"] = ":".join(
        [
            "benchmarks/lib/" + x
            for x in [
                "chameleon/src",
                "dulwich-0.19.13",
                "jinja2",
                "pyxl",
                "monte",
                "pytz",
                "genshi",
                "mako",
                "sqlalchemy",
                "sympy",
                "sqlalchemy",
                "genshi",
                "twisted-trunk/twisted",
                "pypy",
            ]
        ]
    )
    return env


def setup_env_unladen():
    env = os.environ.copy()
    env["PYTHONPATH"] = ":".join(
        [
            "benchmarks/unladen_swallow/lib/" + x
            for x in ["django", "html5lib", "spambayes", "spitfire", "lockfile"]
        ]
    )
    env["PYTHONPATH"] = "benchmarks/lib:benchmarks/lib/pytz:" + env["PYTHONPATH"]
    return env


def setup_bm(typ):
    if typ == "own":
        return setup_env(), "benchmarks/own/", BENCHMARKS
    elif typ == "unladen_swallow":
        return (
            setup_env_unladen(),
            "benchmarks/unladen_swallow/performance/",
            BENCHMARKS_UNLADEN_SWALLOW,
        )
    else:
        raise Exception("unreachable path")


def run_icbd(env, exe_path):
    env["PYTHONPATH"] = "icbd"
    os.chdir("benchmarks/own/icbd")
    command = [
        "%s/%s" % (this_dir, exe_path),
        "-m", "icbd.type_analyzer.analyze_all",
        "-I", "stdlib/python2.5_tiny",
        "-I", ".",
        "-E", "icbd/type_analyzer/tests",
        "-E", "icbd/compiler/benchmarks",
        "-E", "icbd/compiler/tests",
        "-I", "stdlib/type_mocks",
        "-n", "icbd",
    ]
    subprocess.run(command, env=env, stdout=subprocess.DEVNULL)
    os.chdir(this_dir)


def run(typ, mode=None):
    dirname = "pypylogs_%s" % (get_time())
    if not os.path.exists(dirname):
        os.mkdir(dirname)

    env = setup_env()
    env, bm_path, benchmarks = setup_bm(typ)
    num = parse_args()
    for bm in benchmarks:
        for exe_name, exe_path in COMMANDS:
            for i in range(num):
                log_output = "%s/%s/%s_%s_%i.log" % (
                    this_dir,
                    dirname,
                    exe_name,
                    bm,
                    i + 1,
                )
                if mode == "genext-stats":
                    env["PYPYLOG"] = "jit-genext-stats:%s" % (log_output)
                else:
                    env["PYPYLOG"] = "jit-summary:%s" % (log_output)

                target_path = bm_path + "%s.py" % (bm)
                command = [exe_path, target_path]
                if bm in ('krakatau', 'icbd'):
                    command.extend(['-n', '2'])
                print("Running %s against %s..." % (exe_name, bm))
                if bm == "bm_icbd":
                    run_icbd(env, exe_path)
                else:
                    subprocess.run(command, env=env, stdout=subprocess.DEVNULL)


if __name__ == "__main__":
    run("own")
    run('unladen_swallow')
