#!/usr/bin/env python3
"""
Visualization tool for PyPy JIT benchmark statistics.

This script measures and plots JIT summary data including tracing time
and other performance metrics for PyPy benchmarks.
"""

import os
import matplotlib.pyplot as plt
import pandas as pd
import argparse

from statistics import geometric_mean, median, variance, mean

from jitext_bench import *

def parse_args():
    parser = argparse.ArgumentParser(
        description='Measure and visualize PyPy JIT benchmark statistics',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run benchmark plotting for 10 iterations
  %(prog)s -n 10 -d pypylogs_11172025_1431 -b own-macro

  # Plot data from existing logs
  %(prog)s -n 5 -d my_benchmark_logs -b own
        """
    )
    parser.add_argument('-n', '--number', type=int, required=True,
                        help='Number of benchmark iterations')
    parser.add_argument('-d', '--dir', type=str, required=True,
                        help='Directory containing benchmark log files')
    parser.add_argument('-b', '--benchmark', type=str, required=True,
                        help='Benchmark type (own, own-macro, own-micro)')
    args = parser.parse_args()
    return args

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


def plot(output_ave, output_var, dirname):

    df_ave = pd.DataFrame(output_ave)
    df_var = pd.DataFrame(output_var)

    print(df_ave)
    print("\nArithmetic Mean:")
    print(df_ave.mean())

    fig, axes = plt.subplots(1, 2, gridspec_kw={'width_ratios': [9, 1]})

    df_ave.plot.bar(yerr=df_var, ax=axes[0], title='Tracing time', ylabel='time (s)')
    df_ave.mean().plot.bar(ax=axes[1], ylim=[0, 0.9], title='mean')

    plt.tight_layout()
    plt.savefig('%s_tracing_time.pdf' % (dirname))

    # Calculate normalized values
    new_df_ave = df_ave['pypy-jit-ext-c'] / df_ave['pypy-c']

    # Calculate and print geometric mean for normalized values
    geomean_normalized = geometric_mean(new_df_ave)
    print("\nGeometric Mean (normalized):")
    print(f"pypy-jit-ext-c / pypy-c: {geomean_normalized:.6f}")

    fig, axes = plt.subplots(1, 2, gridspec_kw={'width_ratios': [9, 1]})

    new_df_ave.plot.bar(ax=axes[0], title='Tracing time', ylabel='Relative time (normalized to pypy-c)')
    axes[0].axhline(1.0, color='red', linestyle='--', linewidth=1, alpha=0.7)

    # Add geometric mean to normalized plot
    geomean_normalized = geometric_mean(new_df_ave)
    axes[1].bar(['geomean'], [geomean_normalized], color='C0')
    axes[1].axhline(1.0, color='red', linestyle='--', linewidth=1, alpha=0.7)
    axes[1].set_ylim([0, max(2.0, geomean_normalized * 1.2)])
    axes[1].set_ylabel('Relative time')
    axes[1].set_title('geomean')

    # Add text showing the value
    axes[1].text(0, geomean_normalized, f'{geomean_normalized:.3f}',
                ha='center', va='bottom', fontweight='bold')

    plt.tight_layout()
    plt.savefig('%s_tracing_time_norm.pdf' % (dirname))


if __name__ == '__main__':
    args = parse_args()

    benchmarks = setup_bms(args.benchmark)
    output_ave, output_var = measure(args.number, args.dir, benchmarks)
    plot(output_ave, output_var, args.dir)
