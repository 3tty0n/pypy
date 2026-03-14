#!/usr/bin/env python3
"""
Visualization tool for PyPy JIT benchmark statistics.

This script measures and plots JIT summary data including tracing time
and other performance metrics for PyPy benchmarks.
"""

import os
import re
import sys
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import argparse

from statistics import geometric_mean, median, variance, mean

from jitext_bench import *

def collect_paper_data(log_dir):
    """Collect jit-summary data from all logs, grouped by benchmark and variant.

    Returns dict: {benchmark: {'baseline': [list of dicts], 'genext': [list of dicts]}}
    """
    from collections import defaultdict
    data = defaultdict(lambda: {'baseline': [], 'genext': []})
    for fname in sorted(os.listdir(log_dir)):
        if not fname.endswith('.log'):
            continue
        path = os.path.join(log_dir, fname)
        if fname.startswith('pypy-jit-ext-c_'):
            variant = 'genext'
            rest = fname[len('pypy-jit-ext-c_'):]
        elif fname.startswith('pypy-c_'):
            variant = 'baseline'
            rest = fname[len('pypy-c_'):]
        else:
            continue
        m = re.match(r'^(.+)_(\d+)\.log$', rest)
        if not m:
            continue
        benchmark = m.group(1)
        summary = parse_jit_summary(path)
        if summary and 'Tracing (total)' in summary:
            data[benchmark][variant].append(summary)
    return dict(data)


def plot_paper(log_dir, output_file):
    """Generate a two-panel publication-quality figure for the paper."""
    data = collect_paper_data(log_dir)

    # Filter to benchmarks that have both variants with sufficient runs
    benchmarks = []
    for bm, variants in sorted(data.items()):
        if len(variants['baseline']) >= 3 and len(variants['genext']) >= 3:
            benchmarks.append(bm)

    if not benchmarks:
        print("Error: no benchmarks found with both baseline and genext data")
        sys.exit(1)

    # Compute medians for each benchmark
    stats = {}
    for bm in benchmarks:
        bm_stats = {}
        for variant in ('baseline', 'genext'):
            runs = data[bm][variant]
            bm_stats[variant] = {
                'interp': np.median([r['Interpretation'] for r in runs]),
                'resume': np.median([r['Resume data'] for r in runs]),
                'optim': np.median([r['Optimization'] for r in runs]),
                'backend': np.median([r['Backend'] for r in runs]),
                'tracing': np.median([r['Tracing (total)'] for r in runs]),
                'ops': np.median([r.get('ops', 0) for r in runs]),
                'recorded_ops': np.median([r.get('recorded ops', 0) for r in runs]),
                'opt_ops': np.median([r.get('opt ops', 0) for r in runs]),
            }
            # IQR for total compilation time (for error bars)
            total_compile = sorted([r['Tracing (total)'] + r['Optimization'] + r['Backend'] for r in runs])
            bm_stats[variant]['compile_q25'] = np.percentile(total_compile, 25)
            bm_stats[variant]['compile_q75'] = np.percentile(total_compile, 75)
            bm_stats[variant]['compile_median'] = np.median(total_compile)
            # TOTAL time (includes everything)
            total_vals = sorted([r['TOTAL'] for r in runs if 'TOTAL' in r])
            if total_vals:
                bm_stats[variant]['total'] = np.median(total_vals)
                bm_stats[variant]['total_q25'] = np.percentile(total_vals, 25)
                bm_stats[variant]['total_q75'] = np.percentile(total_vals, 75)
            else:
                bm_stats[variant]['total'] = 0
                bm_stats[variant]['total_q25'] = 0
                bm_stats[variant]['total_q75'] = 0
        stats[bm] = bm_stats

    # Sort benchmarks by baseline total tracing time (ascending)
    benchmarks.sort(key=lambda bm: stats[bm]['baseline']['tracing'])

    # Compute geometric mean ratios
    ratios = {k: [] for k in ('tracing', 'interp', 'resume', 'optim', 'backend', 'ops', 'recorded', 'opt')}
    for bm in benchmarks:
        b = stats[bm]['baseline']
        g = stats[bm]['genext']
        for key, bk, gk in [
            ('tracing', 'tracing', 'tracing'), ('interp', 'interp', 'interp'),
            ('resume', 'resume', 'resume'), ('optim', 'optim', 'optim'),
            ('backend', 'backend', 'backend'), ('ops', 'ops', 'ops'),
            ('recorded', 'recorded_ops', 'recorded_ops'), ('opt', 'opt_ops', 'opt_ops'),
        ]:
            if b[bk] > 0:
                ratios[key].append(g[gk] / b[bk])

    def geomean(vals):
        return np.exp(np.mean(np.log(vals))) if vals else float('nan')

    gm = {k: geomean(v) for k, v in ratios.items()}

    print(f"Geometric mean ratios (GenExt / baseline):")
    print(f"  Tracing total: {gm['tracing']:.2f}")
    print(f"  Interpretation: {gm['interp']:.2f}")
    print(f"  Resume data: {gm['resume']:.2f}")
    print(f"  Optimization: {gm['optim']:.2f}")
    print(f"  Backend: {gm['backend']:.2f}")
    print(f"  Total ops: {gm['ops']:.2f}")
    print(f"  Recorded ops: {gm['recorded']:.2f}")
    print(f"  Opt ops: {gm['opt']:.2f}")

    # Categorize benchmarks by baseline compilation time
    # (interp + resume + optim + backend)
    def compile_time(bm):
        b = stats[bm]['baseline']
        return b['interp'] + b['resume'] + b['optim'] + b['backend']

    small = [bm for bm in benchmarks if compile_time(bm) < 0.1]
    medium = [bm for bm in benchmarks if 0.1 <= compile_time(bm) < 1.0]
    large = [bm for bm in benchmarks if compile_time(bm) >= 1.0]

    print(f"\nBenchmark categories: {len(small)} small, {len(medium)} medium, {len(large)} large")

    # Compute per-run normalized TOTAL ratios for IQR error bars
    for bm in benchmarks:
        baseline_runs = data[bm]['baseline']
        genext_runs = data[bm]['genext']
        b_median_total = stats[bm]['baseline']['total']
        if b_median_total > 0:
            norm_ratios = sorted([r['TOTAL'] / b_median_total for r in genext_runs if 'TOTAL' in r])
        else:
            norm_ratios = [0]
        stats[bm]['norm_total_median'] = np.median(norm_ratios)
        stats[bm]['norm_total_q25'] = np.percentile(norm_ratios, 25)
        stats[bm]['norm_total_q75'] = np.percentile(norm_ratios, 75)

    # Publication styling
    def apply_paper_style():
        matplotlib.rcParams.update({
            'font.family': 'serif',
            'font.size': 8,
            'axes.labelsize': 8,
            'axes.titlesize': 9,
            'xtick.labelsize': 7,
            'ytick.labelsize': 7,
            'legend.fontsize': 7,
            'figure.dpi': 300,
            'savefig.dpi': 300,
            'axes.linewidth': 0.5,
            'xtick.major.width': 0.5,
            'ytick.major.width': 0.5,
            'lines.linewidth': 0.5,
        })

    colors = {
        'interp': '#4878CF',   # blue
        'resume': '#6ACC65',   # green
        'optim': '#D65F5F',    # red
        'backend': '#B47CC7',  # purple
    }

    n = len(benchmarks)
    ns, nm, nl = len(small), len(medium), len(large)
    short_names = [bm.replace('bm_', '').replace('_', ' ') for bm in benchmarks]
    x = np.arange(n)
    bar_width = 0.35

    # Derive output filenames from the base
    base, ext = os.path.splitext(output_file)
    file_a = f'{base}_a{ext}'
    file_b = f'{base}_b{ext}'

    # ===== Figure (a): Compilation time breakdown by category =====
    apply_paper_style()

    fig_a, axes_a = plt.subplots(1, 3, figsize=(7.0, 2.3),
                                  gridspec_kw={'width_ratios': [ns, nm, nl], 'wspace': 0.25})

    def plot_abs_panel(ax, bms, title_suffix, show_legend=False, show_ylabel=True):
        nb = len(bms)
        if nb == 0:
            ax.set_visible(False)
            return
        xb = np.arange(nb)
        for variant, offset in [('baseline', -bar_width/2), ('genext', bar_width/2)]:
            interp_vals = [stats[bm][variant]['interp'] for bm in bms]
            resume_vals = [stats[bm][variant]['resume'] for bm in bms]
            optim_vals = [stats[bm][variant]['optim'] for bm in bms]
            backend_vals = [stats[bm][variant]['backend'] for bm in bms]

            bottom = np.zeros(nb)
            for phase, vals, color in [
                ('Interpretation', interp_vals, colors['interp']),
                ('Resume data', resume_vals, colors['resume']),
                ('Optimization', optim_vals, colors['optim']),
                ('Backend', backend_vals, colors['backend']),
            ]:
                label = phase if (variant == 'baseline' and show_legend) else None
                ax.bar(xb + offset, vals, bar_width, bottom=bottom,
                       color=color, label=label, edgecolor='white', linewidth=0.3)
                bottom += np.array(vals)

            medians = [stats[bm][variant]['compile_median'] for bm in bms]
            err_lo = [max(0, medians[i] - stats[bm][variant]['compile_q25']) for i, bm in enumerate(bms)]
            err_hi = [max(0, stats[bm][variant]['compile_q75'] - medians[i]) for i, bm in enumerate(bms)]
            ax.errorbar(xb + offset, bottom, yerr=[err_lo, err_hi],
                        fmt='none', ecolor='black', elinewidth=0.4, capsize=1.5, capthick=0.4)

        names = [bm.replace('bm_', '').replace('_', ' ') for bm in bms]
        ax.set_xticks(xb)
        ax.set_xticklabels(names, rotation=45, ha='right', fontsize=5.5)
        ax.set_title(title_suffix, fontsize=8, pad=2)
        if show_ylabel:
            ax.set_ylabel('Compilation time (s)')
        ax.set_xlim(-0.5, nb - 0.5)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    plot_abs_panel(axes_a[0], small, f'Small ({ns})', show_legend=True, show_ylabel=True)
    plot_abs_panel(axes_a[1], medium, f'Medium ({nm})', show_ylabel=False)
    plot_abs_panel(axes_a[2], large, f'Large ({nl})', show_ylabel=False)

    handles, labels = axes_a[0].get_legend_handles_labels()
    fig_a.legend(handles, labels, loc='upper center', ncol=4, frameon=False,
                 fontsize=6, bbox_to_anchor=(0.5, 1.02))
    fig_a.suptitle('Compilation time breakdown: Baseline (left) vs GenExt (right)',
                   fontsize=9, y=1.08)

    fig_a.savefig(file_a, bbox_inches='tight', pad_inches=0.05)
    plt.close(fig_a)
    print(f"  (a) saved to {file_a}")

    # ===== Figure (b): Combined normalized tracing + TOTAL time =====
    apply_paper_style()

    fig_b, (ax_norm, ax_total) = plt.subplots(2, 1, figsize=(7.0, 4.2),
                                               gridspec_kw={'hspace': 0.30})

    # --- Panel (a) top: Normalized tracing time ---
    norm_interp = []
    norm_resume = []
    norm_overhead = []
    for bm in benchmarks:
        b = stats[bm]['baseline']
        g = stats[bm]['genext']
        bt = b['tracing']
        if bt > 0:
            gi = g['interp'] / bt
            gr = g['resume'] / bt
            gt = g['tracing'] / bt
            norm_interp.append(gi)
            norm_resume.append(gr)
            norm_overhead.append(gt - gi - gr)
        else:
            norm_interp.append(0)
            norm_resume.append(0)
            norm_overhead.append(0)

    # Append geomean bar
    gm_tracing = gm['tracing']
    gm_interp_val = geomean(norm_interp)
    gm_resume_val = geomean(norm_resume)
    gm_overhead_val = gm_tracing - gm_interp_val - gm_resume_val

    # x positions: benchmarks + gap + geomean
    x_gap = 1.0  # gap before geomean bar
    x_gm = n + x_gap
    x_all = np.append(x, x_gm)
    norm_interp_all = norm_interp + [gm_interp_val]
    norm_resume_all = norm_resume + [gm_resume_val]
    norm_overhead_all = norm_overhead + [gm_overhead_val]
    n_all = len(x_all)

    bottom_norm = np.zeros(n_all)
    ax_norm.bar(x_all, norm_interp_all, 0.6, bottom=bottom_norm,
                color=colors['interp'], label='Interpretation', edgecolor='white', linewidth=0.3)
    bottom_norm += np.array(norm_interp_all)
    ax_norm.bar(x_all, norm_resume_all, 0.6, bottom=bottom_norm,
                color=colors['resume'], label='Resume data', edgecolor='white', linewidth=0.3)
    bottom_norm += np.array(norm_resume_all)
    ax_norm.bar(x_all, norm_overhead_all, 0.6, bottom=bottom_norm,
                color='#999999', label='Overhead', edgecolor='white', linewidth=0.3)

    ax_norm.axhline(y=1.0, color='black', linestyle='--', linewidth=0.5, zorder=5)
    ax_norm.text(0.01, 1.0, 'baseline', fontsize=5.5, ha='left', va='bottom',
                 color='black', transform=ax_norm.get_yaxis_transform())

    # Annotate geomean value on top of the bar
    ax_norm.text(x_gm, bottom_norm[-1] + 0.03, f'{gm_tracing:.2f}',
                 fontsize=6, fontweight='bold', ha='center', va='bottom', color='black')

    # Separator line before geomean
    sep_x = n + x_gap / 2 - 0.5
    ax_norm.axvline(x=sep_x, color='gray', linestyle=':', linewidth=0.5)

    ax_norm.set_xticks(x_all)
    ax_norm.set_xticklabels([])  # No x-labels on top panel
    ax_norm.set_ylabel('Norm. tracing time')
    ax_norm.set_title('(a) Tracing time', fontsize=8, pad=4)
    ax_norm.legend(ncol=3, loc='lower left', frameon=True, fontsize=5.5,
                    edgecolor='#cccccc', fancybox=False)
    ax_norm.set_xlim(-0.5, x_gm + 0.5)
    ax_norm.set_ylim(0, max(bottom_norm) * 1.15)
    ax_norm.spines['top'].set_visible(False)
    ax_norm.spines['right'].set_visible(False)

    # --- Panel (b) bottom: Normalized TOTAL time ---
    norm_total = [stats[bm]['norm_total_median'] for bm in benchmarks]
    err_lo = [stats[bm]['norm_total_median'] - stats[bm]['norm_total_q25'] for bm in benchmarks]
    err_hi = [stats[bm]['norm_total_q75'] - stats[bm]['norm_total_median'] for bm in benchmarks]
    err_lo = [max(0, e) for e in err_lo]
    err_hi = [max(0, e) for e in err_hi]

    # Compute TOTAL geomean
    total_ratios = [v for v in norm_total if v > 0]
    gm_total = np.exp(np.mean(np.log(total_ratios))) if total_ratios else float('nan')

    # Append geomean bar
    norm_total_all = norm_total + [gm_total]
    err_lo_all = err_lo + [0]
    err_hi_all = err_hi + [0]

    bar_colors = ['#6ACC65' if v <= 1.0 else '#D65F5F' for v in norm_total_all]
    ax_total.bar(x_all, norm_total_all, 0.6, color=bar_colors, edgecolor='white', linewidth=0.3)
    ax_total.errorbar(x_all, norm_total_all, yerr=[err_lo_all, err_hi_all],
                      fmt='none', ecolor='black', elinewidth=0.4, capsize=1.5, capthick=0.4)

    # Baseline reference line
    ax_total.axhline(y=1.0, color='black', linestyle='--', linewidth=0.5, zorder=5)
    ax_total.text(0.01, 1.01, 'baseline', fontsize=5.5, ha='left', va='bottom',
                  color='black', transform=ax_total.get_yaxis_transform())

    # Annotate geomean value on top of the bar
    ax_total.text(x_gm, gm_total + 0.02, f'{gm_total:.2f}',
                  fontsize=6, fontweight='bold', ha='center', va='bottom', color='black')

    # Separator line before geomean
    ax_total.axvline(x=sep_x, color='gray', linestyle=':', linewidth=0.5)

    short_names_all = short_names + ['geomean']
    ax_total.set_xticks(x_all)
    ax_total.set_xticklabels(short_names_all, rotation=45, ha='right', fontsize=5.5)
    ax_total.set_ylabel('Norm. total time')
    ax_total.set_title('(b) Total execution time', fontsize=8, pad=4)
    ax_total.set_xlim(-0.5, x_gm + 0.5)
    y_max = max(max(v + e for v, e in zip(norm_total_all, err_hi_all)) * 1.05, 1.05)
    y_min = min(min(v - e for v, e in zip(norm_total_all, err_lo_all)) * 0.95, 0.95)
    ax_total.set_ylim(y_min, y_max)
    ax_total.spines['top'].set_visible(False)
    ax_total.spines['right'].set_visible(False)

    fig_b.savefig(file_b, bbox_inches='tight', pad_inches=0.05)
    plt.close(fig_b)
    print(f"  (b) saved to {file_b}")

    # ===== Figure (c): Normalized trace lengths =====
    apply_paper_style()
    file_c = f'{base}_c{ext}'

    fig_c, ax_trace = plt.subplots(1, 1, figsize=(7.0, 2.3))

    # Compute normalized trace length ratios (genext / baseline)
    norm_ops = []
    norm_recorded = []
    norm_opt = []
    for bm in benchmarks:
        b = stats[bm]['baseline']
        g = stats[bm]['genext']
        norm_ops.append(g['ops'] / b['ops'] if b['ops'] > 0 else 0)
        norm_recorded.append(g['recorded_ops'] / b['recorded_ops'] if b['recorded_ops'] > 0 else 0)
        norm_opt.append(g['opt_ops'] / b['opt_ops'] if b['opt_ops'] > 0 else 0)

    # Compute geometric means
    valid_ops = [r for r in norm_ops if r > 0]
    valid_rec = [r for r in norm_recorded if r > 0]
    valid_opt = [r for r in norm_opt if r > 0]
    gm_ops = geomean(valid_ops)
    gm_rec = geomean(valid_rec)
    gm_opt_ops = geomean(valid_opt)

    # Append geomean bars
    x_gm_c = n + x_gap
    x_all_c = np.append(x, x_gm_c)
    norm_ops_all = norm_ops + [gm_ops]
    norm_recorded_all = norm_recorded + [gm_rec]
    norm_opt_all = norm_opt + [gm_opt_ops]

    n_all_c = len(x_all_c)
    bar_w = 0.25
    colors_trace = {
        'ops': '#2ecc71',       # green
        'recorded': '#3498db',  # blue
        'opt': '#9b59b6',       # purple
    }

    ax_trace.bar(x_all_c - bar_w, norm_ops_all, bar_w,
                 color=colors_trace['ops'], label='Total ops',
                 edgecolor='white', linewidth=0.3)
    ax_trace.bar(x_all_c, norm_recorded_all, bar_w,
                 color=colors_trace['recorded'], label='Recorded ops',
                 edgecolor='white', linewidth=0.3)
    ax_trace.bar(x_all_c + bar_w, norm_opt_all, bar_w,
                 color=colors_trace['opt'], label='Optimized ops',
                 edgecolor='white', linewidth=0.3)

    ax_trace.axhline(y=1.0, color='black', linestyle='--', linewidth=0.5, zorder=5)
    ax_trace.text(0.01, 1.0, 'baseline', fontsize=5.5, ha='left', va='bottom',
                  color='black', transform=ax_trace.get_yaxis_transform())

    # Separator line before geomean
    sep_x_c = n + x_gap / 2 - 0.5
    ax_trace.axvline(x=sep_x_c, color='gray', linestyle=':', linewidth=0.5)

    short_names_all_c = short_names + ['geomean']
    ax_trace.set_xticks(x_all_c)
    ax_trace.set_xticklabels(short_names_all_c, rotation=45, ha='right', fontsize=5.5)
    ax_trace.set_ylabel('Norm. operation count')
    ax_trace.set_title('(c) Trace length (GenExt / Baseline)', fontsize=8, pad=4)
    ax_trace.legend(ncol=3, loc='upper right', frameon=True, fontsize=5.5,
                    edgecolor='#cccccc', fancybox=False)
    ax_trace.set_xlim(-0.5, x_gm_c + 0.5)
    all_vals = norm_ops_all + norm_recorded_all + norm_opt_all
    ax_trace.set_ylim(0, max(all_vals) * 1.15)
    ax_trace.spines['top'].set_visible(False)
    ax_trace.spines['right'].set_visible(False)

    fig_c.savefig(file_c, bbox_inches='tight', pad_inches=0.05)
    plt.close(fig_c)
    print(f"  (c) saved to {file_c}")

    # ===== Figure (d): Absolute TOTAL time, small/medium/large split =====
    apply_paper_style()
    file_d = f'{base}_d{ext}'

    fig_d, axes_d = plt.subplots(1, 3, figsize=(7.0, 2.8),
                                  gridspec_kw={'width_ratios': [ns, nm, nl], 'wspace': 0.25})

    def plot_total_panel(ax, bms, title_suffix, show_ylabel=True):
        nb = len(bms)
        if nb == 0:
            ax.set_visible(False)
            return
        xb = np.arange(nb)
        bw = 0.35

        baseline_vals = [stats[bm]['baseline']['total'] for bm in bms]
        genext_vals = [stats[bm]['genext']['total'] for bm in bms]

        ax.bar(xb - bw/2, baseline_vals, bw, color='#AAAAAA',
               label='Baseline', edgecolor='white', linewidth=0.3)
        ax.bar(xb + bw/2, genext_vals, bw, color='#4878CF',
               label='GenExt', edgecolor='white', linewidth=0.3)

        # IQR error bars
        for variant, offset, vals in [('baseline', -bw/2, baseline_vals),
                                       ('genext', bw/2, genext_vals)]:
            q25 = [stats[bm][variant]['total_q25'] for bm in bms]
            q75 = [stats[bm][variant]['total_q75'] for bm in bms]
            med = [stats[bm][variant]['total'] for bm in bms]
            err_lo = [max(0, med[i] - q25[i]) for i in range(nb)]
            err_hi = [max(0, q75[i] - med[i]) for i in range(nb)]
            ax.errorbar(xb + offset, med, yerr=[err_lo, err_hi],
                        fmt='none', ecolor='black', elinewidth=0.4, capsize=1.5, capthick=0.4)

        names = [bm.replace('bm_', '').replace('_', ' ') for bm in bms]
        ax.set_xticks(xb)
        ax.set_xticklabels(names, rotation=45, ha='right', fontsize=5.5)
        ax.set_title(title_suffix, fontsize=8, pad=4)
        if show_ylabel:
            ax.set_ylabel('TOTAL time (s)')
        ax.set_xlim(-0.5, nb - 0.5)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    plot_total_panel(axes_d[0], small, f'Small ({ns})', show_ylabel=True)
    plot_total_panel(axes_d[1], medium, f'Medium ({nm})', show_ylabel=False)
    plot_total_panel(axes_d[2], large, f'Large ({nl})', show_ylabel=False)

    handles, labels_d = axes_d[0].get_legend_handles_labels()
    fig_d.legend(handles, labels_d, loc='upper center', ncol=2, frameon=False,
                 fontsize=6, bbox_to_anchor=(0.5, 1.0))
    fig_d.suptitle('Absolute TOTAL time: Baseline vs GenExt',
                   fontsize=9, y=1.05)

    fig_d.savefig(file_d, bbox_inches='tight', pad_inches=0.05)
    plt.close(fig_d)
    print(f"  (d) saved to {file_d}")

    print(f"\n  {n} benchmarks ({ns} small, {nm} medium, {nl} large)")
    print(f"  Geomean tracing ratio: {gm_tracing:.2f}")
    print(f"  Geomean TOTAL ratio: {gm_total:.2f}")
    print(f"  Geomean ops ratio: {gm_ops:.2f}")
    print(f"  Geomean recorded ops ratio: {gm_rec:.2f}")
    print(f"  Geomean opt ops ratio: {gm_opt_ops:.2f}")


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
    parser.add_argument('-n', '--number', type=int,
                        help='Number of benchmark iterations')
    parser.add_argument('-d', '--dir', type=str, required=True,
                        help='Directory containing benchmark log files')
    parser.add_argument('-b', '--benchmark', type=str,
                        help='Benchmark type (own, own-macro, own-micro)')
    parser.add_argument('--paper', action='store_true',
                        help='Generate publication-quality two-panel figure for the paper')
    parser.add_argument('-o', '--output', type=str,
                        help='Output file path (used with --paper)')
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
    if args.paper:
        output_file = args.output if args.output else 'paper_figure.pdf'
        plot_paper(args.dir, output_file)
    else:
        if not args.number or not args.benchmark:
            print("Error: -n and -b are required for non-paper mode")
            sys.exit(1)
        benchmarks = setup_bms_plot(args.benchmark)
        output_ave, output_var = measure(args.number, args.dir, benchmarks)
        plot(output_ave, output_var, args.dir)
