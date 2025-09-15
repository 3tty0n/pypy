#!/usr/bin/env pypy3

import os
import subprocess
import argparse

BENCHMARKS = [
    'bm_chameleon', 'bm_dulwich_log', 'bm_icbd', 'bm_mako',
    'raytrace-simple', 'scimark', 'spectral-norm', 'spitfire',
    'telco', 'bm_gzip', 'bm_krakatau', 'bm_mdp', 'pyxl_bench',
]

BENCHMARKS_UNLADEN_SWALLOW = [
    'bm_ai',
    'bm_ai',
    'bm_call_simple',
    'bm_django',
    'bm_html5lib',
    'bm_nbody'
    'bm_pickle',
    'bm_regex_compile',
    'bm_regex_effbot',
    'bm_regex_v8',
    'bm_richards',
    'bm_rietveld',
    'bm_spambayes',
    'bm_spitfire',
    'bm_threading',
    'bm_unpack_sequence',
]

COMMANDS = [
    ("pypy-c", "./pypy/goal/pypy-c"),
    ("pypy-jit-ext-c", "./pypy/goal/pypy-jit-ext-c")
]

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

def setup_env_unladen():
    env = os.environ.copy()
    env["PYTHONPATH"] = ':'.join(['benchmarks/unladen_swallow/lib' + x for x in [
        'django', 'html5lib', 'lockfile', 'spambayes', 'spitfire'
    ]])
    return env

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
                command = [exe_path, bm_path]
                print("Running %s against %s..." % (exe_name, bm))
                subprocess.run(command, env=env, stdout=subprocess.DEVNULL)

def run_unladen_swallow():
    dirname = 'pypylogs'
    if not os.path.exists(dirname):
        os.mkdir(dirname)
    env = setup_env()
    num = parse_args()
    for bm in BENCHMARKS_UNLADEN_SWALLOW:
        for i in range(num):
            for exe_name, exe_path in COMMANDS:
                if not os.path.exists('%s/%s' % (dirname, exe_name)):
                    os.mkdir('%s/%s' % (dirname, exe_name))
                log_output = '%s/%s/%s_%i.log' % (dirname, exe_name, bm, i+1)
                env["PYPYLOG"] = "jit-summary:%s" % (log_output)
                bm_path = "benchmarks/unladen_swallow/performance/%s.py" % (bm)
                command = [exe_path,  bm_path]
                print("Running %s against %s..." % (exe_name, bm))
                subprocess.run(command, env=env, stdout=subprocess.DEVNULL)

if __name__ == '__main__':
    run_unladen_swallow()
