#!/usr/bin/env python3
import os
import matplotlib.pyplot as plt
import pandas as pd
import argparse

from statistics import geometric_mean, median, variance, mean

from jitext_bench import *

def parse_args():
    parser = argparse.ArgumentParser(
        prog='Measuring the jit summary data'
    )
    parser.add_argument('-n', '--number', type=int)
    parser.add_argument('-d', '--dir', type=str)
    parser.add_argument('-b' ,'--benchmark', type=str)
    args = parser.parse_args()
    return args.number, args.dir, args.benchmark

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

def collect_data(num, dirname, benchmarks):
    result = {}
    for exe_name, _ in COMMANDS:
        for bm in benchmarks:
            for i in range(num):
                path = dirname + "/" + exe_name + "_" + bm + "_" + str(i+1) + ".log"
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


def measure(num, dirname, benchmarks):
    result = collect_data(num, dirname, benchmarks)
    output_ave = {}
    output_var = {}
    for exe_name, _ in COMMANDS:
        for bm in benchmarks:
            if bm == 'scimark': continue
            ave = mean(result[exe_name][bm])
            var = variance(result[exe_name][bm])

            if exe_name not in output_ave and exe_name not in output_var:
                output_ave[exe_name] = {}
                output_var[exe_name] = {}

            output_ave[exe_name][bm] = ave
            output_var[exe_name][bm] = var

    return output_ave, output_var


def measure_genext_stats(dirname):
    pass

def plot(output_ave, output_var, dirname):

    df_ave = pd.DataFrame(output_ave)
    df_var = pd.DataFrame(output_var)

    print(df_ave)
    print(df_ave.mean())

    fig, axes = plt.subplots(1, 2, gridspec_kw={'width_ratios': [9, 1]})

    df_ave.plot.bar(yerr=df_var, ax=axes[0], title='Tracing time', ylabel='time (s)')
    df_ave.mean().plot.bar(ax=axes[1], ylim=[0, 0.9], title='average')

    plt.tight_layout()
    plt.savefig('%s_tracing_time.pdf' % (dirname))

    fig, ax = plt.subplots()

    new_df_ave = df_ave['pypy-jit-ext-c'] / df_ave['pypy-c']

    new_df_ave.plot.bar(ax=ax, title='Tracing time', ylabel='Relative time (normalized to pypy-c)')
    ax.axhline(1.0)

    plt.tight_layout()
    plt.savefig('%s_tracing_time_norm.pdf' % (dirname))


if __name__ == '__main__':
    num, dirname, bm_typ = parse_args()
    benchmarks = setup_bms(bm_typ)
    output_ave, output_var = measure(num, dirname, benchmarks)
    plot(output_ave, output_var, dirname)
