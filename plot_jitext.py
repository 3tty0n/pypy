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
            if line.startswith("Tracing:"):
                items = line.split('\t')
                time = float(items[-1])
                result["Tracing"] = time
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
    metrics = ["Tracing", "Backend", "TOTAL"]

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
    metrics = ["Tracing", "Backend", "TOTAL"]
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


def plot(output_ave, output_var, dirname):
    metrics = ["Tracing", "Backend", "TOTAL"]

    # Filter out empty metrics
    available_metrics = [m for m in metrics if m in output_ave and output_ave[m]]
    n_metrics = len(available_metrics)

    if n_metrics == 0:
        print("No data to plot")
        return

    # Set style for better visuals
    plt.style.use('seaborn-v0_8-darkgrid')

    # Create a large figure with subplots: n_metrics rows, 4 columns
    # Each row: [absolute bar | absolute mean | normalized bar | normalized geomean]
    fig = plt.figure(figsize=(22, 5.5 * n_metrics))
    fig.patch.set_facecolor('white')

    for idx, metric in enumerate(available_metrics):
        df_ave = pd.DataFrame(output_ave[metric])
        df_var = pd.DataFrame(output_var[metric])

        print(f"\n=== {metric} ===")
        print(df_ave)
        print(f"\nArithmetic Mean ({metric}):")
        print(df_ave.mean())

        # Calculate ratio of arithmetic means
        mean_pypy_c = df_ave['pypy-c'].mean()
        mean_pypy_jit_ext_c = df_ave['pypy-jit-ext-c'].mean()
        ratio_of_means = mean_pypy_jit_ext_c / mean_pypy_c
        print(f"\nRatio of Arithmetic Means ({metric}):")
        print(f"pypy-jit-ext-c / pypy-c: {ratio_of_means:.6f}")

        row = idx

        # Define colors
        colors = ['#3498db', '#e74c3c']  # Blue for pypy-c, Red for pypy-jit-ext-c

        # Absolute values plot (left side)
        ax1 = plt.subplot(n_metrics, 4, row * 4 + 1)
        df_ave.plot.bar(yerr=df_var, ax=ax1, legend=(row == 0),
                       color=colors, width=0.8, capsize=4, error_kw={'linewidth': 1.5})
        ax1.set_title(f'{metric} time', fontsize=14, fontweight='bold', pad=10)
        ax1.set_ylabel('Time (s)', fontsize=12, fontweight='bold')
        ax1.set_xlabel('')
        ax1.tick_params(axis='both', labelsize=10)
        ax1.grid(axis='y', alpha=0.3, linestyle='--')
        if row == 0:
            ax1.legend(fontsize=10, framealpha=0.9, loc='upper right')
        # Rotate x-labels if they're long
        ax1.tick_params(axis='x', rotation=90)

        ax2 = plt.subplot(n_metrics, 4, row * 4 + 2)
        df_ave.mean().plot.bar(ax=ax2, legend=False, color=colors, width=0.7)
        ax2.set_title('Mean', fontsize=14, fontweight='bold', pad=10)
        ax2.set_ylabel('Time (s)', fontsize=12, fontweight='bold')
        ax2.set_xlabel('')
        ax2.tick_params(axis='both', labelsize=10)
        ax2.grid(axis='y', alpha=0.3, linestyle='--')
        ax2.tick_params(axis='x', rotation=90)

        # Normalized values plot (right side)
        new_df_ave = df_ave['pypy-jit-ext-c'] / df_ave['pypy-c']
        geomean_normalized = geometric_mean(new_df_ave.values)

        print(f"\nGeometric Mean ({metric}, normalized):")
        if pd.notna(geomean_normalized):
            print(f"pypy-jit-ext-c / pypy-c: {geomean_normalized:.6f}")
        else:
            print(f"pypy-jit-ext-c / pypy-c: N/A (no valid data)")

        ax3 = plt.subplot(n_metrics, 4, row * 4 + 3)
        new_df_ave.plot.bar(ax=ax3, legend=False, color='#2ecc71', width=0.8)
        ax3.set_title(f'{metric} time (normalized)', fontsize=14, fontweight='bold', pad=10)
        ax3.set_ylabel('Relative time (pypy-jit-ext-c / pypy-c)', fontsize=11, fontweight='bold')
        ax3.set_xlabel('')
        ax3.tick_params(axis='both', labelsize=10)
        ax3.grid(axis='y', alpha=0.3, linestyle='--')
        ax3.axhline(1.0, color='#e74c3c', linestyle='--', linewidth=2, alpha=0.8, label='Baseline')
        ax3.tick_params(axis='x', rotation=90)
        # Add shading for better/worse regions
        ax3.axhspan(0, 1, alpha=0.05, color='green', zorder=0)
        ax3.axhspan(1, ax3.get_ylim()[1], alpha=0.05, color='red', zorder=0)

        ax4 = plt.subplot(n_metrics, 4, row * 4 + 4)
        if pd.notna(geomean_normalized):
            # Show both geomean and ratio of means
            x_pos = [0, 1]
            values = [geomean_normalized, ratio_of_means]
            labels = ['Geomean', 'Ratio of\nMeans']
            bar_colors = ['#9b59b6', '#f39c12']  # Purple and orange
            bars = ax4.bar(x_pos, values, color=bar_colors, width=0.7,
                          edgecolor='black', linewidth=1.5)
            ax4.axhline(1.0, color='#e74c3c', linestyle='--', linewidth=2, alpha=0.8)
            y_max = max(2.0, max(values) * 1.15)
            ax4.set_ylim([min(0, min(values) * 0.9), y_max])
            ax4.set_ylabel('Relative time', fontsize=12, fontweight='bold')
            ax4.set_title('Summary', fontsize=14, fontweight='bold', pad=10)
            ax4.set_xticks(x_pos)
            ax4.set_xticklabels(labels, fontsize=10, fontweight='bold')
            ax4.tick_params(axis='y', labelsize=10)
            ax4.grid(axis='y', alpha=0.3, linestyle='--')
            # Add shading for better/worse regions
            ax4.axhspan(0, 1, alpha=0.05, color='green', zorder=0)
            ax4.axhspan(1, y_max, alpha=0.05, color='red', zorder=0)
            # Add value labels with background
            for i, v in enumerate(values):
                # Determine if value is better (< 1) or worse (> 1)
                color = 'green' if v < 1.0 else 'red' if v > 1.0 else 'black'
                ax4.text(i, v, f'{v:.4f}',
                        ha='center', va='bottom', fontweight='bold', fontsize=11,
                        color=color,
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                                 edgecolor=color, alpha=0.8))
        else:
            ax4.set_title('Summary (N/A)', fontsize=14, fontweight='bold', pad=10)
            ax4.text(0.5, 0.5, 'No valid data', ha='center', va='center',
                    transform=ax4.transAxes, fontsize=12, fontweight='bold')

    # Add overall title
    fig.suptitle('PyPy JIT Performance Metrics Comparison',
                 fontsize=18, fontweight='bold', y=0.995)

    plt.tight_layout(rect=[0, 0, 1, 0.99])  # Adjust for suptitle
    filename = f'{dirname}_all_metrics.pdf'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"\nSaved: {filename}")

    # Reset style to default after plotting
    plt.style.use('default')


if __name__ == '__main__':
    args = parse_args()
    benchmarks = setup_bms_plot(args.benchmark)
    output_ave, output_var = measure(args.number, args.dir, benchmarks)
    plot(output_ave, output_var, args.dir)
