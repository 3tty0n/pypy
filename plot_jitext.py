#!/usr/bin/env python3
"""
Visualization tool for PyPy JIT benchmark statistics.

This script measures and plots JIT summary data including tracing time
and other performance metrics for PyPy benchmarks.
"""

import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
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
            if line.startswith("Tracing (total):"):
                items = line.split('\t')
                time = float(items[-1])
                result["Tracing (total)"] = time
            elif line.startswith("  Interpretation:"):
                items = line.split('\t')
                time = float(items[-1])
                result["Interpretation"] = time
            elif line.startswith("  Resume data"):
                items = line.split('\t')
                time = float(items[-1])
                result["Resume Data"] = time
            elif line.startswith("Optimization:"):
                items = line.split('\t')
                time = float(items[-1])
                result["Optimization"] = time
            elif line.startswith("Backend:"):
                items = line.split('\t')
                time = float(items[-1])
                result["Backend"] = time
            elif line.startswith("TOTAL:"):
                items = line.split('\t')
                time = float(items[-1])
                result["TOTAL"] = time
    return result

def collect_data(num, dirname, benchmarks):
    result = {}
    metrics = ["Tracing (total)", "Interpretation", "Resume Data", "Optimization", "Backend", "TOTAL"]

    for exe_name, _ in COMMANDS:
        for bm in benchmarks:
            for metric in metrics:
                if exe_name not in result:
                    result[exe_name] = {}
                if metric not in result[exe_name]:
                    result[exe_name][metric] = {}
                if bm not in result[exe_name][metric]:
                    result[exe_name][metric][bm] = []

            for i in range(num):
                path = dirname + "/" + exe_name + "_" + bm + "_" + str(i+1) + ".log"
                jit_summary = parse_jit_summary(path)

                for metric in metrics:
                    if metric in jit_summary:
                        result[exe_name][metric][bm].append(jit_summary[metric])

    return result


def measure(num, dirname, benchmarks):
    result = collect_data(num, dirname, benchmarks)
    metrics = ["Tracing (total)", "Interpretation", "Resume Data", "Optimization", "Backend", "TOTAL"]
    output_ave = {metric: {} for metric in metrics}
    output_var = {metric: {} for metric in metrics}

    for exe_name, _ in COMMANDS:
        for metric in metrics:
            for bm in benchmarks:
                if bm == 'scimark': continue
                if len(result[exe_name][metric][bm]) == 0:
                    continue

                ave = mean(result[exe_name][metric][bm])
                # variance requires at least 2 data points
                if len(result[exe_name][metric][bm]) > 1:
                    var = variance(result[exe_name][metric][bm])
                else:
                    var = 0

                if exe_name not in output_ave[metric]:
                    output_ave[metric][exe_name] = {}
                if exe_name not in output_var[metric]:
                    output_var[metric][exe_name] = {}

                output_ave[metric][exe_name][bm] = ave
                output_var[metric][exe_name][bm] = var

    return output_ave, output_var


def plot_stacked_bars(output_ave, output_var, dirname, include_opt_in_tracing):
    """
    Create stacked bar charts.

    Args:
        output_ave: Average data
        output_var: Variance data
        dirname: Output directory name
        include_opt_in_tracing: If True, include Optimization in Tracing bar (bug version)
                                 If False, show Optimization separately (correct version)
    """
    # Set style for better visuals
    plt.style.use('seaborn-v0_8-darkgrid')

    # Get all benchmarks from the data
    benchmarks = list(output_ave.get("Interpretation", {}).get("pypy-c", {}).keys())
    benchmarks = [bm for bm in benchmarks if bm != 'scimark']

    if not benchmarks:
        print("No benchmark data available")
        return

    # Prepare data for stacking
    exe_names = ['pypy-c', 'pypy-jit-ext-c']

    # Create figure with 2 rows: one for each executable
    fig, axes = plt.subplots(2, 1, figsize=(max(12, len(benchmarks) * 0.8), 10))
    fig.patch.set_facecolor('white')

    # Define colors for each component
    colors = {
        'Interpretation': '#3498db',      # Blue
        'Resume Data': '#9b59b6',         # Purple
        'Optimization': '#e67e22',        # Orange
        'Backend': '#e74c3c'              # Red
    }

    for idx, exe_name in enumerate(exe_names):
        ax = axes[idx]

        # Prepare data for this executable
        interpretation_data = []
        resume_data = []
        optimization_data = []
        backend_data = []

        for bm in benchmarks:
            interp = output_ave.get("Interpretation", {}).get(exe_name, {}).get(bm, 0)
            resume = output_ave.get("Resume Data", {}).get(exe_name, {}).get(bm, 0)
            opt = output_ave.get("Optimization", {}).get(exe_name, {}).get(bm, 0)
            backend = output_ave.get("Backend", {}).get(exe_name, {}).get(bm, 0)

            interpretation_data.append(interp)
            resume_data.append(resume)
            optimization_data.append(opt)
            backend_data.append(backend)

        x = np.arange(len(benchmarks))
        width = 0.6

        if include_opt_in_tracing:
            # Version 1 (bug): Optimization included in Tracing
            # Stack order: Interpretation + Resume Data + Optimization, then Backend
            p1 = ax.bar(x, interpretation_data, width, label='Interpretation',
                       color=colors['Interpretation'])
            p2 = ax.bar(x, resume_data, width, bottom=interpretation_data,
                       label='Resume Data', color=colors['Resume Data'])

            # Add Optimization on top of Resume Data
            bottom_for_opt = np.array(interpretation_data) + np.array(resume_data)
            p3 = ax.bar(x, optimization_data, width, bottom=bottom_for_opt,
                       label='Optimization (in Tracing)', color=colors['Optimization'])

            # Backend on top of everything
            bottom_for_backend = bottom_for_opt + np.array(optimization_data)
            p4 = ax.bar(x, backend_data, width, bottom=bottom_for_backend,
                       label='Backend', color=colors['Backend'])

            title_suffix = " (Bug: Optimization in Tracing)"
        else:
            # Version 2 (correct): Optimization separate
            # Stack order: Interpretation + Resume Data, then Optimization, then Backend
            p1 = ax.bar(x, interpretation_data, width, label='Interpretation',
                       color=colors['Interpretation'])
            p2 = ax.bar(x, resume_data, width, bottom=interpretation_data,
                       label='Resume Data', color=colors['Resume Data'])

            # Backend after tracing components
            bottom_for_opt = np.array(interpretation_data) + np.array(resume_data)
            p3 = ax.bar(x, optimization_data, width, bottom=bottom_for_opt,
                       label='Optimization', color=colors['Optimization'])

            # Backend on top
            bottom_for_backend = bottom_for_opt + np.array(optimization_data)
            p4 = ax.bar(x, backend_data, width, bottom=bottom_for_backend,
                       label='Backend', color=colors['Backend'])

            title_suffix = " (Correct: Optimization Separate)"

        ax.set_ylabel('Time (s)', fontsize=12, fontweight='bold')
        ax.set_title(f'{exe_name}{title_suffix}', fontsize=14, fontweight='bold', pad=10)
        ax.set_xticks(x)
        ax.set_xticklabels(benchmarks, rotation=90, ha='right')
        ax.legend(loc='upper left', fontsize=9, framealpha=0.9)
        ax.grid(axis='y', alpha=0.3, linestyle='--')

        # Print statistics
        total_interp = sum(interpretation_data)
        total_resume = sum(resume_data)
        total_opt = sum(optimization_data)
        total_backend = sum(backend_data)
        total_all = total_interp + total_resume + total_opt + total_backend

        print(f"\n=== {exe_name} ===")
        print(f"Interpretation:  {total_interp:.6f} s ({100*total_interp/total_all:.2f}%)")
        print(f"Resume Data:     {total_resume:.6f} s ({100*total_resume/total_all:.2f}%)")
        print(f"Optimization:    {total_opt:.6f} s ({100*total_opt/total_all:.2f}%)")
        print(f"Backend:         {total_backend:.6f} s ({100*total_backend/total_all:.2f}%)")
        print(f"Total:           {total_all:.6f} s")

    # Add overall title
    version_label = "WITH Optimization in Tracing (Bug)" if include_opt_in_tracing else "WITHOUT Optimization in Tracing (Correct)"
    fig.suptitle(f'PyPy JIT Component Breakdown - {version_label}',
                 fontsize=16, fontweight='bold', y=0.995)

    plt.tight_layout(rect=[0, 0, 1, 0.98])

    suffix = "_with_opt" if include_opt_in_tracing else "_without_opt"
    filename = f'{dirname}_stacked_bars{suffix}.pdf'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"\nSaved: {filename}")

    plt.close()


def plot(output_ave, output_var, dirname):
    """Generate both versions of stacked bar charts"""
    print("\n" + "="*80)
    print("Generating Version 1: Optimization INCLUDED in Tracing (Bug)")
    print("="*80)
    plot_stacked_bars(output_ave, output_var, dirname, include_opt_in_tracing=True)

    print("\n" + "="*80)
    print("Generating Version 2: Optimization SEPARATE from Tracing (Correct)")
    print("="*80)
    plot_stacked_bars(output_ave, output_var, dirname, include_opt_in_tracing=False)

    # Reset style to default after plotting
    plt.style.use('default')


if __name__ == '__main__':
    args = parse_args()
    benchmarks = setup_bms_plot(args.benchmark)
    output_ave, output_var = measure(args.number, args.dir, benchmarks)
    plot(output_ave, output_var, args.dir)
