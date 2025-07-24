import os
import matplotlib.pyplot as plt
import pandas as pd

from statistics import geometric_mean, median, variance, mean

BENCHMARKS = [
    'bm_chameleon', 'bm_dulwich_log', 'bm_icbd', 'bm_mako',
    'raytrace-simple', 'scimark', 'spectral-norm', 'spitfire',
    'telco'
]

BENCHMARKS_NOT_RUNNABLE = [
    'bm_gzip', 'bm_krakatau', 'bm_gzip', 'bm_mdp', 'pyxl_bench',
]

COMMANDS = [
    ("pypy-c", "./pypy/goal/pypy-c"),
    ("pypy-jit-ext-c", "./pypy/goal/pypy-jit-ext-c")
]

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
    return result

def collect_data():
    dir = "pypylogs/"
    result = {}
    for exe_name, _ in COMMANDS:
        for bm in BENCHMARKS:
            for i in range(50):
                path = dir + exe_name + "/" + bm + "_" + str(i+1) + ".log"
                jit_summary = parse_jit_summary(path)
                if exe_name not in result:
                    result[exe_name] = {}
                if bm not in result[exe_name]:
                    result[exe_name][bm] = []

                if 'Tracing' in jit_summary:
                    result[exe_name][bm].append(jit_summary["Tracing"])
                else:
                    break

    return result


def measure():

    result = collect_data()

    output_ave = {}
    output_var = {}

    for exe_name, _ in COMMANDS:
        for bm in BENCHMARKS:
            ave = mean(result[exe_name][bm])
            var = variance(result[exe_name][bm])



            if exe_name not in output_ave and exe_name not in output_var:
                output_ave[exe_name] = {}
                output_var[exe_name] = {}

            output_ave[exe_name][bm] = ave
            output_var[exe_name][bm] = var

    return output_ave, output_var

def plot(output_ave, output_var):

    df_ave = pd.DataFrame(output_ave)
    df_var = pd.DataFrame(output_var)

    fig, axes = plt.subplots(1, 2, gridspec_kw={'width_ratios': [9, 1]})

    df_ave.plot.bar(yerr=df_var, ax=axes[0], title='Tracing time', ylabel='time (s)')
    df_ave.mean().plot.bar(ax=axes[1], ylim=[0, 0.5], title='average')

    plt.tight_layout()
    plt.savefig('pypylogs_tracing_time.pdf')


if __name__ == '__main__':
    output_ave, output_var = measure()
    plot(output_ave, output_var)
