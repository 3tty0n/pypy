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
                result["Resume data"] = time
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
            elif line.startswith("ops:"):
                items = line.split('\t')
                count = int(items[-1])
                result["ops"] = count
            elif line.startswith("recorded ops:"):
                items = line.split('\t')
                count = int(items[-1])
                result["recorded ops"] = count
            elif line.startswith("opt ops:"):
                items = line.split('\t')
                count = int(items[-1])
                result["opt ops"] = count
    return result

def collect_data(num, dirname, benchmarks):
    result = {}
    metrics = ["Tracing (total)", "Interpretation", "Resume data", "Optimization", "Backend", "TOTAL",
               "ops", "recorded ops", "opt ops"]

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
    metrics = ["Tracing (total)", "Interpretation", "Resume data", "Optimization", "Backend", "TOTAL",
               "ops", "recorded ops", "opt ops"]
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


def categorize_benchmarks_by_time(output_ave, benchmarks):
    """
    Categorize benchmarks into small, medium, and large based on total time.

    Returns:
        dict: Dictionary with 'small', 'medium', 'large' keys containing lists of benchmarks
    """
    # Calculate total time for each benchmark (average across both executables)
    benchmark_times = {}
    for bm in benchmarks:
        total_time = 0
        count = 0
        for exe_name in ['pypy-c', 'pypy-jit-ext-c']:
            interp = output_ave.get("Interpretation", {}).get(exe_name, {}).get(bm, 0)
            resume = output_ave.get("Resume data", {}).get(exe_name, {}).get(bm, 0)
            opt = output_ave.get("Optimization", {}).get(exe_name, {}).get(bm, 0)
            backend = output_ave.get("Backend", {}).get(exe_name, {}).get(bm, 0)
            bm_total = interp + resume + opt + backend
            if bm_total > 0:
                total_time += bm_total
                count += 1
        if count > 0:
            benchmark_times[bm] = total_time / count

    # Sort by time
    sorted_benchmarks = sorted(benchmark_times.items(), key=lambda x: x[1])

    # Use thresholds to categorize (can be adjusted)
    # Small: < 0.1s, Medium: 0.1s - 1.0s, Large: > 1.0s
    small = [bm for bm, t in sorted_benchmarks if t < 0.1]
    medium = [bm for bm, t in sorted_benchmarks if 0.1 <= t < 1.0]
    large = [bm for bm, t in sorted_benchmarks if t >= 1.0]

    print("\n=== Benchmark Categorization by Time ===")
    print(f"Small (< 0.1s): {len(small)} benchmarks")
    for bm in small:
        print(f"  {bm}: {benchmark_times[bm]:.4f}s")
    print(f"Medium (0.1s - 1.0s): {len(medium)} benchmarks")
    for bm in medium:
        print(f"  {bm}: {benchmark_times[bm]:.4f}s")
    print(f"Large (>= 1.0s): {len(large)} benchmarks")
    for bm in large:
        print(f"  {bm}: {benchmark_times[bm]:.4f}s")

    return {'small': small, 'medium': medium, 'large': large}


def plot_tracing_comparison_row(ax, benchmarks, output_ave, output_var, colors, show_legend=False):
    """
    Plot Tracing breakdown for both executables side by side with error bars.

    Returns:
        Tuple of (stats_pypy_c, stats_pypy_jit_ext)
    """
    # Prepare data for both executables
    interpretation_pypy_c = []
    resume_pypy_c = []
    optimization_pypy_c = []
    backend_pypy_c = []

    interpretation_jit_ext = []
    resume_jit_ext = []
    optimization_jit_ext = []
    backend_jit_ext = []

    # Variance data
    var_tracing_pypy_c = []
    var_tracing_jit_ext = []

    for bm in benchmarks:
        # pypy-c data
        interp_c = output_ave.get("Interpretation", {}).get("pypy-c", {}).get(bm, 0)
        resume_c = output_ave.get("Resume data", {}).get("pypy-c", {}).get(bm, 0)
        opt_c = output_ave.get("Optimization", {}).get("pypy-c", {}).get(bm, 0)
        backend_c = output_ave.get("Backend", {}).get("pypy-c", {}).get(bm, 0)

        interpretation_pypy_c.append(interp_c)
        resume_pypy_c.append(resume_c)
        optimization_pypy_c.append(opt_c)
        backend_pypy_c.append(backend_c)

        # pypy-jit-ext-c data
        interp_ext = output_ave.get("Interpretation", {}).get("pypy-jit-ext-c", {}).get(bm, 0)
        resume_ext = output_ave.get("Resume data", {}).get("pypy-jit-ext-c", {}).get(bm, 0)
        opt_ext = output_ave.get("Optimization", {}).get("pypy-jit-ext-c", {}).get(bm, 0)
        backend_ext = output_ave.get("Backend", {}).get("pypy-jit-ext-c", {}).get(bm, 0)

        interpretation_jit_ext.append(interp_ext)
        resume_jit_ext.append(resume_ext)
        optimization_jit_ext.append(opt_ext)
        backend_jit_ext.append(backend_ext)

        # Variance for total tracing (sum of variances for independent measurements)
        var_interp_c = output_var.get("Interpretation", {}).get("pypy-c", {}).get(bm, 0)
        var_resume_c = output_var.get("Resume data", {}).get("pypy-c", {}).get(bm, 0)
        var_tracing_pypy_c.append(np.sqrt(var_interp_c + var_resume_c))

        var_interp_ext = output_var.get("Interpretation", {}).get("pypy-jit-ext-c", {}).get(bm, 0)
        var_resume_ext = output_var.get("Resume data", {}).get("pypy-jit-ext-c", {}).get(bm, 0)
        var_tracing_jit_ext.append(np.sqrt(var_interp_ext + var_resume_ext))

    x = np.arange(len(benchmarks))
    width = 0.35

    # Position bars side by side
    x_pypy_c = x - width / 2
    x_jit_ext = x + width / 2

    # Calculate total heights for error bars
    total_pypy_c = np.array(interpretation_pypy_c) + np.array(resume_pypy_c)
    total_jit_ext = np.array(interpretation_jit_ext) + np.array(resume_jit_ext)

    # pypy-c bars
    p1 = ax.bar(x_pypy_c, interpretation_pypy_c, width,
               label='pypy-c: Interpretation',
               color=colors['Interpretation'], alpha=0.7)
    p2 = ax.bar(x_pypy_c, resume_pypy_c, width, bottom=interpretation_pypy_c,
               label='pypy-c: Resume data',
               color=colors['Resume data'], alpha=0.7)

    # pypy-jit-ext-c bars
    p3 = ax.bar(x_jit_ext, interpretation_jit_ext, width,
               label='pypy-jit-ext-c: Interpretation',
               color=colors['Interpretation'], alpha=1.0)
    p4 = ax.bar(x_jit_ext, resume_jit_ext, width, bottom=interpretation_jit_ext,
               label='pypy-jit-ext-c: Resume data',
               color=colors['Resume data'], alpha=1.0)

    # Add error bars on top of stacked bars
    ax.errorbar(x_pypy_c, total_pypy_c, yerr=var_tracing_pypy_c,
                fmt='none', ecolor='black', capsize=3, capthick=1, linewidth=1)
    ax.errorbar(x_jit_ext, total_jit_ext, yerr=var_tracing_jit_ext,
                fmt='none', ecolor='black', capsize=3, capthick=1, linewidth=1)

    ax.set_ylabel('Tracing Time (s)', fontsize=10, fontweight='bold')
    ax.set_title('Tracing (total) Comparison', fontsize=12, fontweight='bold', pad=8)
    ax.set_xticks(x)
    ax.set_xticklabels(benchmarks, rotation=90, ha='right', fontsize=8)
    if show_legend:
        ax.legend(loc='best', fontsize=7, framealpha=0.9, ncol=2)
    ax.grid(axis='y', alpha=0.3, linestyle='--')

    # Return statistics for both
    stats_pypy_c = {
        'interpretation': interpretation_pypy_c,
        'resume': resume_pypy_c,
        'optimization': optimization_pypy_c,
        'backend': backend_pypy_c
    }

    stats_jit_ext = {
        'interpretation': interpretation_jit_ext,
        'resume': resume_jit_ext,
        'optimization': optimization_jit_ext,
        'backend': backend_jit_ext
    }

    return stats_pypy_c, stats_jit_ext


def plot_trace_lengths(ax, benchmarks, output_ave, output_var, show_legend=False):
    """
    Plot trace length metrics (ops, recorded ops, opt ops) side by side for both executables.
    """
    # Prepare data for both executables
    data_pypy_c = {
        'ops': [],
        'recorded ops': [],
        'opt ops': []
    }
    data_jit_ext = {
        'ops': [],
        'recorded ops': [],
        'opt ops': []
    }
    var_pypy_c = {
        'ops': [],
        'recorded ops': [],
        'opt ops': []
    }
    var_jit_ext = {
        'ops': [],
        'recorded ops': [],
        'opt ops': []
    }

    for bm in benchmarks:
        # pypy-c data
        data_pypy_c['ops'].append(output_ave.get("ops", {}).get("pypy-c", {}).get(bm, 0))
        data_pypy_c['recorded ops'].append(output_ave.get("recorded ops", {}).get("pypy-c", {}).get(bm, 0))
        data_pypy_c['opt ops'].append(output_ave.get("opt ops", {}).get("pypy-c", {}).get(bm, 0))

        # pypy-jit-ext-c data
        data_jit_ext['ops'].append(output_ave.get("ops", {}).get("pypy-jit-ext-c", {}).get(bm, 0))
        data_jit_ext['recorded ops'].append(output_ave.get("recorded ops", {}).get("pypy-jit-ext-c", {}).get(bm, 0))
        data_jit_ext['opt ops'].append(output_ave.get("opt ops", {}).get("pypy-jit-ext-c", {}).get(bm, 0))

        # Variance
        var_pypy_c['ops'].append(np.sqrt(output_var.get("ops", {}).get("pypy-c", {}).get(bm, 0)))
        var_pypy_c['recorded ops'].append(np.sqrt(output_var.get("recorded ops", {}).get("pypy-c", {}).get(bm, 0)))
        var_pypy_c['opt ops'].append(np.sqrt(output_var.get("opt ops", {}).get("pypy-c", {}).get(bm, 0)))

        var_jit_ext['ops'].append(np.sqrt(output_var.get("ops", {}).get("pypy-jit-ext-c", {}).get(bm, 0)))
        var_jit_ext['recorded ops'].append(np.sqrt(output_var.get("recorded ops", {}).get("pypy-jit-ext-c", {}).get(bm, 0)))
        var_jit_ext['opt ops'].append(np.sqrt(output_var.get("opt ops", {}).get("pypy-jit-ext-c", {}).get(bm, 0)))

    x = np.arange(len(benchmarks))
    width = 0.14  # 6 bars per benchmark

    # Position bars: 3 metrics × 2 executables = 6 bars per benchmark
    x_ops_c = x - width * 2.5
    x_ops_ext = x - width * 1.5
    x_rec_c = x - width * 0.5
    x_rec_ext = x + width * 0.5
    x_opt_c = x + width * 1.5
    x_opt_ext = x + width * 2.5

    # Colors for trace length metrics
    color_ops = '#2ecc71'  # Green
    color_recorded = '#3498db'  # Blue
    color_opt = '#9b59b6'  # Purple

    # Plot bars
    p1 = ax.bar(x_ops_c, data_pypy_c['ops'], width,
               label='pypy-c: ops', color=color_ops, alpha=0.6)
    p2 = ax.bar(x_ops_ext, data_jit_ext['ops'], width,
               label='pypy-jit-ext-c: ops', color=color_ops, alpha=1.0)

    p3 = ax.bar(x_rec_c, data_pypy_c['recorded ops'], width,
               label='pypy-c: recorded', color=color_recorded, alpha=0.6)
    p4 = ax.bar(x_rec_ext, data_jit_ext['recorded ops'], width,
               label='pypy-jit-ext-c: recorded', color=color_recorded, alpha=1.0)

    p5 = ax.bar(x_opt_c, data_pypy_c['opt ops'], width,
               label='pypy-c: opt ops', color=color_opt, alpha=0.6)
    p6 = ax.bar(x_opt_ext, data_jit_ext['opt ops'], width,
               label='pypy-jit-ext-c: opt ops', color=color_opt, alpha=1.0)

    # Add error bars
    ax.errorbar(x_ops_c, data_pypy_c['ops'], yerr=var_pypy_c['ops'],
                fmt='none', ecolor='black', capsize=1.5, capthick=0.6, linewidth=0.6)
    ax.errorbar(x_ops_ext, data_jit_ext['ops'], yerr=var_jit_ext['ops'],
                fmt='none', ecolor='black', capsize=1.5, capthick=0.6, linewidth=0.6)
    ax.errorbar(x_rec_c, data_pypy_c['recorded ops'], yerr=var_pypy_c['recorded ops'],
                fmt='none', ecolor='black', capsize=1.5, capthick=0.6, linewidth=0.6)
    ax.errorbar(x_rec_ext, data_jit_ext['recorded ops'], yerr=var_jit_ext['recorded ops'],
                fmt='none', ecolor='black', capsize=1.5, capthick=0.6, linewidth=0.6)
    ax.errorbar(x_opt_c, data_pypy_c['opt ops'], yerr=var_pypy_c['opt ops'],
                fmt='none', ecolor='black', capsize=1.5, capthick=0.6, linewidth=0.6)
    ax.errorbar(x_opt_ext, data_jit_ext['opt ops'], yerr=var_jit_ext['opt ops'],
                fmt='none', ecolor='black', capsize=1.5, capthick=0.6, linewidth=0.6)

    ax.set_ylabel('Number of Operations', fontsize=10, fontweight='bold')
    ax.set_title('Trace Length (Number of Operations)', fontsize=12, fontweight='bold', pad=8)
    ax.set_xticks(x)
    ax.set_xticklabels(benchmarks, rotation=90, ha='right', fontsize=8)
    if show_legend:
        ax.legend(loc='best', fontsize=7, framealpha=0.9, ncol=3)
    ax.grid(axis='y', alpha=0.3, linestyle='--')

    # Calculate and display geometric means
    ratios_ops = [ext / c if c > 0 else 0 for ext, c in zip(data_jit_ext['ops'], data_pypy_c['ops'])]
    ratios_rec = [ext / c if c > 0 else 0 for ext, c in zip(data_jit_ext['recorded ops'], data_pypy_c['recorded ops'])]
    ratios_opt = [ext / c if c > 0 else 0 for ext, c in zip(data_jit_ext['opt ops'], data_pypy_c['opt ops'])]

    valid_ratios_ops = [r for r in ratios_ops if r > 0]
    valid_ratios_rec = [r for r in ratios_rec if r > 0]
    valid_ratios_opt = [r for r in ratios_opt if r > 0]

    geomean_text = "Geometric Mean (jit-ext-c / pypy-c):\n"
    if valid_ratios_ops:
        geomean_ops = geometric_mean(valid_ratios_ops)
        geomean_text += f"  ops: {geomean_ops:.4f}"
    if valid_ratios_rec:
        geomean_rec = geometric_mean(valid_ratios_rec)
        geomean_text += f"  recorded: {geomean_rec:.4f}"
    if valid_ratios_opt:
        geomean_opt = geometric_mean(valid_ratios_opt)
        geomean_text += f"  opt ops: {geomean_opt:.4f}"

    # Add text box with geometric means
    ax.text(0.02, 0.98, geomean_text, transform=ax.transAxes,
            fontsize=8, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))


def plot_all_components_side_by_side(ax, benchmarks, output_ave, output_var, colors, show_legend=False):
    """
    Plot all 4 components side by side for both executables.
    For each benchmark: 8 bars showing Interp, Resume, Opt, Backend for both pypy-c and pypy-jit-ext-c
    """
    # Prepare data for both executables
    data_pypy_c = {
        'interpretation': [],
        'resume': [],
        'optimization': [],
        'backend': []
    }
    data_jit_ext = {
        'interpretation': [],
        'resume': [],
        'optimization': [],
        'backend': []
    }
    var_pypy_c = {
        'interpretation': [],
        'resume': [],
        'optimization': [],
        'backend': []
    }
    var_jit_ext = {
        'interpretation': [],
        'resume': [],
        'optimization': [],
        'backend': []
    }

    for bm in benchmarks:
        # pypy-c data
        data_pypy_c['interpretation'].append(output_ave.get("Interpretation", {}).get("pypy-c", {}).get(bm, 0))
        data_pypy_c['resume'].append(output_ave.get("Resume data", {}).get("pypy-c", {}).get(bm, 0))
        data_pypy_c['optimization'].append(output_ave.get("Optimization", {}).get("pypy-c", {}).get(bm, 0))
        data_pypy_c['backend'].append(output_ave.get("Backend", {}).get("pypy-c", {}).get(bm, 0))

        # pypy-jit-ext-c data
        data_jit_ext['interpretation'].append(output_ave.get("Interpretation", {}).get("pypy-jit-ext-c", {}).get(bm, 0))
        data_jit_ext['resume'].append(output_ave.get("Resume data", {}).get("pypy-jit-ext-c", {}).get(bm, 0))
        data_jit_ext['optimization'].append(output_ave.get("Optimization", {}).get("pypy-jit-ext-c", {}).get(bm, 0))
        data_jit_ext['backend'].append(output_ave.get("Backend", {}).get("pypy-jit-ext-c", {}).get(bm, 0))

        # Variance
        var_pypy_c['interpretation'].append(np.sqrt(output_var.get("Interpretation", {}).get("pypy-c", {}).get(bm, 0)))
        var_pypy_c['resume'].append(np.sqrt(output_var.get("Resume data", {}).get("pypy-c", {}).get(bm, 0)))
        var_pypy_c['optimization'].append(np.sqrt(output_var.get("Optimization", {}).get("pypy-c", {}).get(bm, 0)))
        var_pypy_c['backend'].append(np.sqrt(output_var.get("Backend", {}).get("pypy-c", {}).get(bm, 0)))

        var_jit_ext['interpretation'].append(np.sqrt(output_var.get("Interpretation", {}).get("pypy-jit-ext-c", {}).get(bm, 0)))
        var_jit_ext['resume'].append(np.sqrt(output_var.get("Resume data", {}).get("pypy-jit-ext-c", {}).get(bm, 0)))
        var_jit_ext['optimization'].append(np.sqrt(output_var.get("Optimization", {}).get("pypy-jit-ext-c", {}).get(bm, 0)))
        var_jit_ext['backend'].append(np.sqrt(output_var.get("Backend", {}).get("pypy-jit-ext-c", {}).get(bm, 0)))

    x = np.arange(len(benchmarks))
    width = 0.11  # 8 bars per benchmark

    # Position bars: 4 components × 2 executables = 8 bars per benchmark
    x_interp_c = x - width * 3.5
    x_interp_ext = x - width * 2.5
    x_resume_c = x - width * 1.5
    x_resume_ext = x - width * 0.5
    x_opt_c = x + width * 0.5
    x_opt_ext = x + width * 1.5
    x_backend_c = x + width * 2.5
    x_backend_ext = x + width * 3.5

    # Plot bars with different colors for each component
    # Lighter colors for pypy-c, darker for pypy-jit-ext-c
    p1 = ax.bar(x_interp_c, data_pypy_c['interpretation'], width,
               label='pypy-c: Interp', color=colors['Interpretation'], alpha=0.6)
    p2 = ax.bar(x_interp_ext, data_jit_ext['interpretation'], width,
               label='pypy-jit-ext-c: Interp', color=colors['Interpretation'], alpha=1.0)

    p3 = ax.bar(x_resume_c, data_pypy_c['resume'], width,
               label='pypy-c: Resume', color=colors['Resume data'], alpha=0.6)
    p4 = ax.bar(x_resume_ext, data_jit_ext['resume'], width,
               label='pypy-jit-ext-c: Resume', color=colors['Resume data'], alpha=1.0)

    p5 = ax.bar(x_opt_c, data_pypy_c['optimization'], width,
               label='pypy-c: Opt', color=colors['Optimization'], alpha=0.6)
    p6 = ax.bar(x_opt_ext, data_jit_ext['optimization'], width,
               label='pypy-jit-ext-c: Opt', color=colors['Optimization'], alpha=1.0)

    p7 = ax.bar(x_backend_c, data_pypy_c['backend'], width,
               label='pypy-c: Backend', color=colors['Backend'], alpha=0.6)
    p8 = ax.bar(x_backend_ext, data_jit_ext['backend'], width,
               label='pypy-jit-ext-c: Backend', color=colors['Backend'], alpha=1.0)

    # Add error bars
    ax.errorbar(x_interp_c, data_pypy_c['interpretation'], yerr=var_pypy_c['interpretation'],
                fmt='none', ecolor='black', capsize=1.5, capthick=0.6, linewidth=0.6)
    ax.errorbar(x_interp_ext, data_jit_ext['interpretation'], yerr=var_jit_ext['interpretation'],
                fmt='none', ecolor='black', capsize=1.5, capthick=0.6, linewidth=0.6)
    ax.errorbar(x_resume_c, data_pypy_c['resume'], yerr=var_pypy_c['resume'],
                fmt='none', ecolor='black', capsize=1.5, capthick=0.6, linewidth=0.6)
    ax.errorbar(x_resume_ext, data_jit_ext['resume'], yerr=var_jit_ext['resume'],
                fmt='none', ecolor='black', capsize=1.5, capthick=0.6, linewidth=0.6)
    ax.errorbar(x_opt_c, data_pypy_c['optimization'], yerr=var_pypy_c['optimization'],
                fmt='none', ecolor='black', capsize=1.5, capthick=0.6, linewidth=0.6)
    ax.errorbar(x_opt_ext, data_jit_ext['optimization'], yerr=var_jit_ext['optimization'],
                fmt='none', ecolor='black', capsize=1.5, capthick=0.6, linewidth=0.6)
    ax.errorbar(x_backend_c, data_pypy_c['backend'], yerr=var_pypy_c['backend'],
                fmt='none', ecolor='black', capsize=1.5, capthick=0.6, linewidth=0.6)
    ax.errorbar(x_backend_ext, data_jit_ext['backend'], yerr=var_jit_ext['backend'],
                fmt='none', ecolor='black', capsize=1.5, capthick=0.6, linewidth=0.6)

    ax.set_ylabel('Time (s)', fontsize=10, fontweight='bold')
    ax.set_title('All Components Side by Side', fontsize=12, fontweight='bold', pad=8)
    ax.set_xticks(x)
    ax.set_xticklabels(benchmarks, rotation=90, ha='right', fontsize=8)
    if show_legend:
        ax.legend(loc='best', fontsize=6, framealpha=0.9, ncol=4)
    ax.grid(axis='y', alpha=0.3, linestyle='--')

    # Calculate and display geometric means as text annotation
    # Calculate ratios for each component
    ratios_interp = [ext / c if c > 0 else 0 for ext, c in zip(data_jit_ext['interpretation'], data_pypy_c['interpretation'])]
    ratios_resume = [ext / c if c > 0 else 0 for ext, c in zip(data_jit_ext['resume'], data_pypy_c['resume'])]
    ratios_opt = [ext / c if c > 0 else 0 for ext, c in zip(data_jit_ext['optimization'], data_pypy_c['optimization'])]
    ratios_backend = [ext / c if c > 0 else 0 for ext, c in zip(data_jit_ext['backend'], data_pypy_c['backend'])]

    # Geometric means
    valid_ratios_interp = [r for r in ratios_interp if r > 0]
    valid_ratios_resume = [r for r in ratios_resume if r > 0]
    valid_ratios_opt = [r for r in ratios_opt if r > 0]
    valid_ratios_backend = [r for r in ratios_backend if r > 0]

    geomean_text = "Geometric Mean (jit-ext-c / pypy-c):\n"
    if valid_ratios_interp:
        geomean_interp = geometric_mean(valid_ratios_interp)
        geomean_text += f"  Interp: {geomean_interp:.4f}"
    if valid_ratios_resume:
        geomean_resume = geometric_mean(valid_ratios_resume)
        geomean_text += f"  Resume: {geomean_resume:.4f}"
    if valid_ratios_opt:
        geomean_opt = geometric_mean(valid_ratios_opt)
        geomean_text += f"  Opt: {geomean_opt:.4f}"
    if valid_ratios_backend:
        geomean_backend = geometric_mean(valid_ratios_backend)
        geomean_text += f"  Backend: {geomean_backend:.4f}"

    # Add text box with geometric means
    ax.text(0.02, 0.98, geomean_text, transform=ax.transAxes,
            fontsize=8, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))


def plot_opt_backend_row(ax, benchmarks, output_ave, output_var, colors, show_legend=False):
    """
    Plot Optimization and Backend data for both executables side by side with error bars.
    """
    optimization_pypy_c = []
    optimization_jit_ext = []
    backend_pypy_c = []
    backend_jit_ext = []

    var_opt_pypy_c = []
    var_opt_jit_ext = []
    var_backend_pypy_c = []
    var_backend_jit_ext = []

    for bm in benchmarks:
        opt_c = output_ave.get("Optimization", {}).get("pypy-c", {}).get(bm, 0)
        opt_ext = output_ave.get("Optimization", {}).get("pypy-jit-ext-c", {}).get(bm, 0)
        backend_c = output_ave.get("Backend", {}).get("pypy-c", {}).get(bm, 0)
        backend_ext = output_ave.get("Backend", {}).get("pypy-jit-ext-c", {}).get(bm, 0)

        optimization_pypy_c.append(opt_c)
        optimization_jit_ext.append(opt_ext)
        backend_pypy_c.append(backend_c)
        backend_jit_ext.append(backend_ext)

        # Variance (standard deviation)
        var_opt_pypy_c.append(np.sqrt(output_var.get("Optimization", {}).get("pypy-c", {}).get(bm, 0)))
        var_opt_jit_ext.append(np.sqrt(output_var.get("Optimization", {}).get("pypy-jit-ext-c", {}).get(bm, 0)))
        var_backend_pypy_c.append(np.sqrt(output_var.get("Backend", {}).get("pypy-c", {}).get(bm, 0)))
        var_backend_jit_ext.append(np.sqrt(output_var.get("Backend", {}).get("pypy-jit-ext-c", {}).get(bm, 0)))

    x = np.arange(len(benchmarks))
    width = 0.18  # Narrower bars for better readability

    # Create grouped bars for Optimization and Backend
    # For each benchmark, show 4 bars: opt(pypy-c), opt(jit-ext), backend(pypy-c), backend(jit-ext)
    x_opt_c = x - width * 1.5
    x_opt_ext = x - width * 0.5
    x_backend_c = x + width * 0.5
    x_backend_ext = x + width * 1.5

    p1 = ax.bar(x_opt_c, optimization_pypy_c, width, label='Opt: pypy-c',
               color=colors['Optimization'], alpha=0.7, edgecolor='black', linewidth=0.5)
    p2 = ax.bar(x_opt_ext, optimization_jit_ext, width, label='Opt: pypy-jit-ext-c',
               color=colors['Optimization'], alpha=1.0, edgecolor='black', linewidth=0.5)
    p3 = ax.bar(x_backend_c, backend_pypy_c, width, label='Backend: pypy-c',
               color=colors['Backend'], alpha=0.7, edgecolor='black', linewidth=0.5)
    p4 = ax.bar(x_backend_ext, backend_jit_ext, width, label='Backend: pypy-jit-ext-c',
               color=colors['Backend'], alpha=1.0, edgecolor='black', linewidth=0.5)

    # Add error bars
    ax.errorbar(x_opt_c, optimization_pypy_c, yerr=var_opt_pypy_c,
                fmt='none', ecolor='black', capsize=2, capthick=0.8, linewidth=0.8)
    ax.errorbar(x_opt_ext, optimization_jit_ext, yerr=var_opt_jit_ext,
                fmt='none', ecolor='black', capsize=2, capthick=0.8, linewidth=0.8)
    ax.errorbar(x_backend_c, backend_pypy_c, yerr=var_backend_pypy_c,
                fmt='none', ecolor='black', capsize=2, capthick=0.8, linewidth=0.8)
    ax.errorbar(x_backend_ext, backend_jit_ext, yerr=var_backend_jit_ext,
                fmt='none', ecolor='black', capsize=2, capthick=0.8, linewidth=0.8)

    ax.set_ylabel('Time (s)', fontsize=10, fontweight='bold')
    ax.set_title('Optimization and Backend', fontsize=12, fontweight='bold', pad=8)
    ax.set_xticks(x)
    ax.set_xticklabels(benchmarks, rotation=90, ha='right', fontsize=8)
    if show_legend:
        ax.legend(loc='upper left', fontsize=7, framealpha=0.9, ncol=2)
    ax.grid(axis='y', alpha=0.3, linestyle='--')


def plot(output_ave, output_var, dirname):
    """Generate a single multi-page PDF with all visualizations (correct version only)"""
    from matplotlib.backends.backend_pdf import PdfPages

    # Set style for better visuals
    plt.style.use('seaborn-v0_8-darkgrid')

    # Get all benchmarks
    all_benchmarks = list(output_ave.get("Interpretation", {}).get("pypy-c", {}).keys())
    all_benchmarks = [bm for bm in all_benchmarks if bm != 'scimark']

    # Categorize benchmarks by time
    benchmark_categories = categorize_benchmarks_by_time(output_ave, all_benchmarks)

    # Define colors for each component
    colors = {
        'Interpretation': '#3498db',      # Blue
        'Resume data': '#9b59b6',         # Purple
        'Optimization': '#e67e22',        # Orange
        'Backend': '#e74c3c'              # Red
    }

    # Create a single PDF file
    pdf_filename = f'{dirname}_stacked_bars.pdf'

    with PdfPages(pdf_filename) as pdf:
        # Page 1: All benchmarks
        print("\n" + "="*80)
        print("Page 1: ALL BENCHMARKS")
        print("="*80)
        create_comparison_page(output_ave, output_var, all_benchmarks, "ALL", colors, pdf)

        # Page 2: Small benchmarks
        if benchmark_categories['small']:
            print("\n" + "="*80)
            print("Page 2: SMALL benchmarks (< 0.1s)")
            print("="*80)
            create_comparison_page(output_ave, output_var, benchmark_categories['small'], "SMALL", colors, pdf)

        # Page 3: Medium benchmarks
        if benchmark_categories['medium']:
            print("\n" + "="*80)
            print("Page 3: MEDIUM benchmarks (0.1s - 1.0s)")
            print("="*80)
            create_comparison_page(output_ave, output_var, benchmark_categories['medium'], "MEDIUM", colors, pdf)

        # Page 4: Large benchmarks
        if benchmark_categories['large']:
            print("\n" + "="*80)
            print("Page 4: LARGE benchmarks (>= 1.0s)")
            print("="*80)
            create_comparison_page(output_ave, output_var, benchmark_categories['large'], "LARGE", colors, pdf)

    print(f"\n{'='*80}")
    print(f"Saved all visualizations to: {pdf_filename}")
    print(f"{'='*80}")

    # Reset style to default after plotting
    plt.style.use('default')


def create_comparison_page(output_ave, output_var, benchmarks, subset_name, colors, pdf):
    """
    Create a single page with 3 rows:
    1. All components side by side (unstacked)
    2. Trace lengths (ops, recorded ops, opt ops)
    3. Normalized Tracing comparison
    """
    if not benchmarks:
        print(f"No benchmark data for {subset_name}")
        return

    # Create figure with 3 rows
    fig, axes = plt.subplots(3, 1, figsize=(max(14, len(benchmarks) * 0.8), 16))
    fig.patch.set_facecolor('white')

    # Row 1: All components side by side (unstacked)
    plot_all_components_side_by_side(axes[0], benchmarks, output_ave, output_var, colors, show_legend=True)

    # Row 2: Trace lengths
    plot_trace_lengths(axes[1], benchmarks, output_ave, output_var, show_legend=True)

    # Collect statistics for printing
    stats_pypy_c = {
        'interpretation': [],
        'resume': [],
        'optimization': [],
        'backend': []
    }
    stats_pypy_jit = {
        'interpretation': [],
        'resume': [],
        'optimization': [],
        'backend': []
    }

    for bm in benchmarks:
        stats_pypy_c['interpretation'].append(output_ave.get("Interpretation", {}).get("pypy-c", {}).get(bm, 0))
        stats_pypy_c['resume'].append(output_ave.get("Resume data", {}).get("pypy-c", {}).get(bm, 0))
        stats_pypy_c['optimization'].append(output_ave.get("Optimization", {}).get("pypy-c", {}).get(bm, 0))
        stats_pypy_c['backend'].append(output_ave.get("Backend", {}).get("pypy-c", {}).get(bm, 0))

        stats_pypy_jit['interpretation'].append(output_ave.get("Interpretation", {}).get("pypy-jit-ext-c", {}).get(bm, 0))
        stats_pypy_jit['resume'].append(output_ave.get("Resume data", {}).get("pypy-jit-ext-c", {}).get(bm, 0))
        stats_pypy_jit['optimization'].append(output_ave.get("Optimization", {}).get("pypy-jit-ext-c", {}).get(bm, 0))
        stats_pypy_jit['backend'].append(output_ave.get("Backend", {}).get("pypy-jit-ext-c", {}).get(bm, 0))

    # Print statistics
    for exe_name, stats in [('pypy-c', stats_pypy_c), ('pypy-jit-ext-c', stats_pypy_jit)]:
        total_interp = sum(stats['interpretation'])
        total_resume = sum(stats['resume'])
        total_opt = sum(stats['optimization'])
        total_backend = sum(stats['backend'])
        total_all = total_interp + total_resume + total_opt + total_backend
        total_tracing = total_interp + total_resume

        if total_all > 0:
            print(f"\n=== {exe_name} ({subset_name}) ===")
            print(f"Tracing (total): {total_tracing:.6f} s ({100*total_tracing/total_all:.2f}%)")
            print(f"  Interpretation:  {total_interp:.6f} s ({100*total_interp/total_all:.2f}%)")
            print(f"  Resume data:     {total_resume:.6f} s ({100*total_resume/total_all:.2f}%)")
            print(f"Optimization:    {total_opt:.6f} s ({100*total_opt/total_all:.2f}%)")
            print(f"Backend:         {total_backend:.6f} s ({100*total_backend/total_all:.2f}%)")
            print(f"Total:           {total_all:.6f} s")

    # Row 3: Normalized Tracing comparison (pypy-jit-ext-c / pypy-c)
    ax_norm = axes[2]

    norm_interpretation = []
    norm_resume = []

    for bm in benchmarks:
        pypy_c_interp = output_ave.get("Interpretation", {}).get("pypy-c", {}).get(bm, 0)
        pypy_c_resume = output_ave.get("Resume data", {}).get("pypy-c", {}).get(bm, 0)
        pypy_c_tracing = pypy_c_interp + pypy_c_resume

        jit_ext_interp = output_ave.get("Interpretation", {}).get("pypy-jit-ext-c", {}).get(bm, 0)
        jit_ext_resume = output_ave.get("Resume data", {}).get("pypy-jit-ext-c", {}).get(bm, 0)

        # Normalize each component by pypy-c tracing total
        if pypy_c_tracing > 0:
            norm_interpretation.append(jit_ext_interp / pypy_c_tracing)
            norm_resume.append(jit_ext_resume / pypy_c_tracing)
        else:
            norm_interpretation.append(0)
            norm_resume.append(0)

    x = np.arange(len(benchmarks))
    width = 0.6

    # Stack: Interpretation | Resume data
    p1 = ax_norm.bar(x, norm_interpretation, width, label='Interpretation',
                     color=colors['Interpretation'])
    p2 = ax_norm.bar(x, norm_resume, width, bottom=norm_interpretation,
                     label='Resume data', color=colors['Resume data'])

    ax_norm.axhline(1.0, color='black', linestyle='--', linewidth=2, alpha=0.5, label='Baseline (pypy-c)')
    ax_norm.set_ylabel('Normalized Time', fontsize=10, fontweight='bold')
    ax_norm.set_title('Normalized Tracing: pypy-jit-ext-c / pypy-c', fontsize=12, fontweight='bold', pad=8)
    ax_norm.set_xticks(x)
    ax_norm.set_xticklabels(benchmarks, rotation=90, ha='right', fontsize=8)
    ax_norm.legend(loc='upper left', fontsize=8, framealpha=0.9)
    ax_norm.grid(axis='y', alpha=0.3, linestyle='--')

    # Add green/red shading
    y_max = ax_norm.get_ylim()[1]
    ax_norm.axhspan(0, 1, alpha=0.05, color='green', zorder=0)
    ax_norm.axhspan(1, y_max, alpha=0.05, color='red', zorder=0)

    # Calculate and print geometric mean for Tracing
    total_ratios = np.array(norm_interpretation) + np.array(norm_resume)
    valid_ratios = [r for r in total_ratios if r > 0]
    if valid_ratios:
        geomean = geometric_mean(valid_ratios)
        print(f"\nGeometric Mean Tracing Ratio ({subset_name}): {geomean:.6f}")

        # Display geometric mean on the plot
        geomean_color = 'green' if geomean < 1.0 else 'red' if geomean > 1.0 else 'black'
        geomean_text = f"Geometric Mean: {geomean:.4f}"
        ax_norm.text(0.98, 0.98, geomean_text, transform=ax_norm.transAxes,
                    fontsize=10, verticalalignment='top', horizontalalignment='right',
                    fontweight='bold', color=geomean_color,
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.9, edgecolor=geomean_color, linewidth=2))

    # Add overall title
    fig.suptitle(f'PyPy JIT Component Breakdown - {subset_name} Benchmarks',
                 fontsize=14, fontweight='bold', y=0.995)

    plt.tight_layout(rect=[0, 0, 1, 0.99])
    pdf.savefig(fig, dpi=300, bbox_inches='tight')
    plt.close()


if __name__ == '__main__':
    args = parse_args()
    benchmarks = setup_bms_plot(args.benchmark)
    output_ave, output_var = measure(args.number, args.dir, benchmarks)
    plot(output_ave, output_var, args.dir)
