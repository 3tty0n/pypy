#!/usr/bin/env pypy3

import os
import subprocess
import argparse

def parse_jit_summary(path):
    result = dict()
    with open(path) as f:
        while True:
            line = f.readline().rstrip()
            if not line:
                break
            if line.startswith("Tracing:"):
                items = line.split('\t')
                time = float(items[-1])
                result["Tracing"] = time
    os.remove(path)

    return result

def parse_args():
    parser = argparse.ArgumentParser(
        prog='Measuring the jit summary data'
    )
    parser.add_argument('-n', '--number', type=int)
    args = parser.parse_args()
    return args.number

def setup_env():
    env = os.environ.copy()
    env["PYTHONPATH"] = ':'.join(['benchmarks/lib/' + x for x in [
        'chameleon/src', 'dulwich-0.19.13', 'jinja2', 'pyxl',
        'monte', 'monte', 'pytz', 'genshi', 'mako', 'sqlalchemy',
        'sympy', 'sqlalchemy', 'genshi', 'twisted-trunk/twisted' ,
    ]])
    return env


BENCHMARKS = [
    'bm_chameleon', 'bm_dulwich_log', 'bm_gzip',
    'bm_icbd', 'bm_krakatau', 'bm_mako', 'bm_mdp', 'pyxl_bench',
    'raytrace-simple', 'scimark', 'spectral-norm', 'spitfire',
    'telco'
]

COMMANDS = [
    ("pypy-c", "./pypy/goal/pypy-c"),
    ("pypy-jit-ext-c", "./pypy/goal/pypy-jit-ext-c")
]

def run():
    dirname = 'pypylogs'
    if not os.path.exists(dirname):
        os.mkdir(dirname)
    env = setup_env()
    num = parse_args()
    for bm in BENCHMARKS:
        for i in range(num):
            for exe_name, exe_path in COMMANDS:
                if not os.path.exists('%s/%s' % (dirname, exe_name)):
                    os.mkdir('%s/%s' % (dirname, exe_name))
                log_output = '%s/%s/%s_%i.log' % (dirname, exe_name, bm, i+1)
                env["PYPYLOG"] = "jit-summary:%s" % (log_output)
                bm_path = "benchmarks/own/%s.py" % (bm)
                command = [exe_path, "--jit", "threshold=23", bm_path]
                print("Running %s against %s..." % (exe_name, bm))
                subprocess.run(command, env=env, stdout=subprocess.DEVNULL)

if __name__ == '__main__':
    run()
