#!/usr/bin/env python3
"""
Visualization tool for genext (generating extension) statistics.

This script analyzes and visualizes specialized vs unspecialized handler
execution statistics from PyPy's JIT genext logs.
"""

import os
import matplotlib.pyplot as plt
import argparse
from collections import defaultdict


def parse_genext_log(log_path):
    """Parse genext log to extract specialized/unspecialized handler counts."""
    specialized = {}
    unspecialized = {}
    current_section = None

    with open(log_path) as f:
        for line in f:
            line = line.strip()

            if '{jit-genext-specialized' in line:
                current_section = 'specialized'
                continue
            elif '{jit-genext-unspecialized' in line:
                current_section = 'unspecialized'
                continue
            elif 'jit-genext-specialized}' in line or 'jit-genext-unspecialized}' in line:
                current_section = None
                continue

            if current_section and line and ',' in line and line != 'jitcodename,count':
                parts = line.rsplit(',', 1)
                if len(parts) == 2:
                    name, count = parts
                    try:
                        count = int(count)
                        if current_section == 'specialized':
                            specialized[name] = count
                        else:
                            unspecialized[name] = count
                    except ValueError:
                        pass

    return specialized, unspecialized


def plot_genext_stats(log_path, output_prefix='genext_stats', top_n=20):
    """Visualize specialized/unspecialized handler execution statistics."""
    specialized, unspecialized = parse_genext_log(log_path)

    # Calculate totals
    total_specialized = sum(specialized.values())
    total_unspecialized = sum(unspecialized.values())

    print(f"Total specialized executions: {total_specialized:,}")
    print(f"Total unspecialized executions: {total_unspecialized:,}")
    print(f"Total executions: {total_specialized + total_unspecialized:,}")
    print(f"Specialized ratio: {100 * total_specialized / (total_specialized + total_unspecialized):.2f}%")
    print(f"Number of unique specialized handlers: {len(specialized)}")
    print(f"Number of unique unspecialized handlers: {len(unspecialized)}")

    # Create figure with multiple subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # 1. Total counts comparison (pie chart)
    ax1 = axes[0, 0]
    labels = ['Specialized', 'Unspecialized']
    sizes = [total_specialized, total_unspecialized]
    colors = ['#66c2a5', '#fc8d62']
    ax1.pie(sizes, labels=labels, autopct='%1.1f%%', colors=colors, startangle=90)
    ax1.set_title('Total Handler Executions: Specialized vs Unspecialized')

    # 2. Top N specialized handlers
    ax2 = axes[0, 1]
    top_specialized = sorted(specialized.items(), key=lambda x: x[1], reverse=True)[:top_n]
    names_spec = [name[:50] for name, _ in top_specialized]  # Truncate long names
    counts_spec = [count for _, count in top_specialized]
    ax2.barh(range(len(names_spec)), counts_spec, color='#66c2a5')
    ax2.set_yticks(range(len(names_spec)))
    ax2.set_yticklabels(names_spec, fontsize=8)
    ax2.set_xlabel('Execution Count')
    ax2.set_title(f'Top {top_n} Specialized Handlers')
    ax2.invert_yaxis()

    # 3. Top N unspecialized handlers
    ax3 = axes[1, 0]
    top_unspecialized = sorted(unspecialized.items(), key=lambda x: x[1], reverse=True)[:top_n]
    names_unspec = [name[:50] for name, _ in top_unspecialized]
    counts_unspec = [count for _, count in top_unspecialized]
    ax3.barh(range(len(names_unspec)), counts_unspec, color='#fc8d62')
    ax3.set_yticks(range(len(names_unspec)))
    ax3.set_yticklabels(names_unspec, fontsize=8)
    ax3.set_xlabel('Execution Count')
    ax3.set_title(f'Top {top_n} Unspecialized Handlers')
    ax3.invert_yaxis()

    # 4. Handler count distribution
    ax4 = axes[1, 1]
    categories = ['Specialized\nHandlers', 'Unspecialized\nHandlers']
    handler_counts = [len(specialized), len(unspecialized)]
    ax4.bar(categories, handler_counts, color=['#66c2a5', '#fc8d62'])
    ax4.set_ylabel('Number of Unique Handlers')
    ax4.set_title('Number of Unique Handlers')
    for i, count in enumerate(handler_counts):
        ax4.text(i, count, str(count), ha='center', va='bottom', fontsize=12, fontweight='bold')

    plt.tight_layout()
    plt.savefig(f'{output_prefix}.pdf')
    print(f"\nPlot saved to {output_prefix}.pdf")

    return specialized, unspecialized


def aggregate_genext_stats(log_dir, pattern='pypy-jit-ext-c_'):
    """Aggregate genext stats from all log files matching the pattern."""
    all_specialized = defaultdict(int)
    all_unspecialized = defaultdict(int)
    benchmark_stats = {}

    log_files = sorted([f for f in os.listdir(log_dir) if f.startswith(pattern) and f.endswith('.log')])

    for log_file in log_files:
        # Extract benchmark name
        benchmark = log_file.replace(pattern, '').replace('_1.log', '')
        log_path = os.path.join(log_dir, log_file)

        specialized, unspecialized = parse_genext_log(log_path)

        # Aggregate counts
        for name, count in specialized.items():
            all_specialized[name] += count
        for name, count in unspecialized.items():
            all_unspecialized[name] += count

        # Store per-benchmark stats
        total_spec = sum(specialized.values())
        total_unspec = sum(unspecialized.values())
        benchmark_stats[benchmark] = {
            'specialized': total_spec,
            'unspecialized': total_unspec,
            'total': total_spec + total_unspec,
            'spec_ratio': 100 * total_spec / (total_spec + total_unspec) if (total_spec + total_unspec) > 0 else 0
        }

    return dict(all_specialized), dict(all_unspecialized), benchmark_stats


def plot_aggregated_genext_stats(log_dir, output_file='aggregated_genext_stats.pdf', top_n=30, pattern='pypy-jit-ext-c_'):
    """Create comprehensive visualization of aggregated genext statistics."""

    all_specialized, all_unspecialized, benchmark_stats = aggregate_genext_stats(log_dir, pattern)

    # Calculate totals
    total_specialized = sum(all_specialized.values())
    total_unspecialized = sum(all_unspecialized.values())
    total_executions = total_specialized + total_unspecialized

    print("=" * 80)
    print("AGGREGATED GENEXT STATISTICS (All Benchmarks Combined)")
    print("=" * 80)
    print(f"Total specialized executions: {total_specialized:,}")
    print(f"Total unspecialized executions: {total_unspecialized:,}")
    print(f"Total executions: {total_executions:,}")
    print(f"Specialized ratio: {100 * total_specialized / total_executions:.2f}%")
    print(f"Number of unique specialized handlers: {len(all_specialized)}")
    print(f"Number of unique unspecialized handlers: {len(all_unspecialized)}")
    print()

    print("Per-Benchmark Statistics:")
    print("-" * 80)
    print(f"{'Benchmark':<30} {'Total':>12} {'Specialized':>12} {'Unspec':>12} {'Spec %':>8}")
    print("-" * 80)
    for bm, stats in sorted(benchmark_stats.items()):
        print(f"{bm:<30} {stats['total']:>12,} {stats['specialized']:>12,} "
              f"{stats['unspecialized']:>12,} {stats['spec_ratio']:>7.2f}%")
    print("=" * 80)

    # Create figure with multiple subplots
    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

    # 1. Overall pie chart (top left)
    ax1 = fig.add_subplot(gs[0, 0])
    labels = ['Specialized', 'Unspecialized']
    sizes = [total_specialized, total_unspecialized]
    colors = ['#66c2a5', '#fc8d62']
    wedges, texts, autotexts = ax1.pie(sizes, labels=labels, autopct='%1.1f%%',
                                         colors=colors, startangle=90)
    for autotext in autotexts:
        autotext.set_fontsize(12)
        autotext.set_weight('bold')
    ax1.set_title('Overall: Specialized vs Unspecialized\n(All Benchmarks Combined)',
                  fontsize=12, fontweight='bold')

    # 2. Per-benchmark specialization ratios (top middle and right)
    ax2 = fig.add_subplot(gs[0, 1:])
    benchmarks = sorted(benchmark_stats.keys())
    spec_ratios = [benchmark_stats[bm]['spec_ratio'] for bm in benchmarks]
    bars = ax2.barh(range(len(benchmarks)), spec_ratios, color='#66c2a5')
    ax2.set_yticks(range(len(benchmarks)))
    ax2.set_yticklabels(benchmarks, fontsize=9)
    ax2.set_xlabel('Specialization Ratio (%)', fontsize=10)
    ax2.set_title('Specialization Ratio by Benchmark', fontsize=12, fontweight='bold')
    ax2.axvline(x=50, color='red', linestyle='--', alpha=0.5, linewidth=1)
    ax2.invert_yaxis()
    # Add percentage labels on bars
    for i, (bar, ratio) in enumerate(zip(bars, spec_ratios)):
        ax2.text(ratio + 1, i, f'{ratio:.1f}%', va='center', fontsize=8)

    # 3. Top N aggregated specialized handlers (middle left)
    ax3 = fig.add_subplot(gs[1, :])
    top_specialized = sorted(all_specialized.items(), key=lambda x: x[1], reverse=True)[:top_n]
    names_spec = [name[:60] for name, _ in top_specialized]
    counts_spec = [count for _, count in top_specialized]
    bars_spec = ax3.barh(range(len(names_spec)), counts_spec, color='#66c2a5')
    ax3.set_yticks(range(len(names_spec)))
    ax3.set_yticklabels(names_spec, fontsize=8)
    ax3.set_xlabel('Total Execution Count (All Benchmarks)', fontsize=10)
    ax3.set_title(f'Top {top_n} Specialized Handlers (Aggregated)', fontsize=12, fontweight='bold')
    ax3.invert_yaxis()
    # Add count labels
    for i, (bar, count) in enumerate(zip(bars_spec, counts_spec)):
        ax3.text(count, i, f' {count:,}', va='center', fontsize=7)

    # 4. Top N aggregated unspecialized handlers (bottom)
    ax4 = fig.add_subplot(gs[2, :])
    top_unspecialized = sorted(all_unspecialized.items(), key=lambda x: x[1], reverse=True)[:top_n]
    names_unspec = [name[:60] for name, _ in top_unspecialized]
    counts_unspec = [count for _, count in top_unspecialized]
    bars_unspec = ax4.barh(range(len(names_unspec)), counts_unspec, color='#fc8d62')
    ax4.set_yticks(range(len(names_unspec)))
    ax4.set_yticklabels(names_unspec, fontsize=8)
    ax4.set_xlabel('Total Execution Count (All Benchmarks)', fontsize=10)
    ax4.set_title(f'Top {top_n} Unspecialized Handlers (Aggregated)', fontsize=12, fontweight='bold')
    ax4.invert_yaxis()
    # Add count labels
    for i, (bar, count) in enumerate(zip(bars_unspec, counts_unspec)):
        ax4.text(count, i, f' {count:,}', va='center', fontsize=7)

    plt.suptitle('Aggregated Genext Statistics - All Benchmarks',
                 fontsize=16, fontweight='bold', y=0.995)

    plt.savefig(output_file, bbox_inches='tight', dpi=150)
    print(f"\nAggregated plot saved to {output_file}")

    return all_specialized, all_unspecialized, benchmark_stats


def parse_args():
    parser = argparse.ArgumentParser(
        description='Visualize PyPy JIT genext (generating extension) statistics',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Visualize a single genext log file
  %(prog)s -g pypylogs/pypy-jit-ext-c_benchmark_1.log -o genext_benchmark

  # Aggregate all genext logs in a directory
  %(prog)s -a -d pypylogs_11172025_1431 -o aggregated_genext_stats.pdf

  # Change the log file pattern and show top 50 handlers
  %(prog)s -a -d pypylogs -p custom-pattern_ -t 50
        """
    )
    parser.add_argument('-g', '--genext-log', type=str,
                        help='Path to genext log file to visualize specialized/unspecialized handlers')
    parser.add_argument('-a', '--aggregate', action='store_true',
                        help='Aggregate all genext logs in the directory')
    parser.add_argument('-d', '--dir', type=str,
                        help='Directory containing genext log files (required for --aggregate)')
    parser.add_argument('-p', '--pattern', type=str, default='pypy-jit-ext-c_',
                        help='Log file pattern for aggregation (default: pypy-jit-ext-c_)')
    parser.add_argument('-o', '--output', type=str, default='genext_stats',
                        help='Output file/prefix for genext stats plot (default: genext_stats)')
    parser.add_argument('-t', '--top', type=int, default=20,
                        help='Number of top handlers to show (default: 20)')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()

    # If aggregate mode is specified, aggregate all genext logs in directory
    if args.aggregate:
        if not args.dir:
            print("Error: --dir is required when using --aggregate")
            import sys
            sys.exit(1)
        output_file = args.output if args.output != 'genext_stats' else 'aggregated_genext_stats.pdf'
        plot_aggregated_genext_stats(args.dir, output_file, args.top, args.pattern)
    # If genext-log is specified, visualize genext stats
    elif args.genext_log:
        plot_genext_stats(args.genext_log, args.output, args.top)
    else:
        print("Error: Provide one of:")
        print("  -g/--genext-log <file> [--top N] [--output <prefix>]")
        print("       for single genext log visualization")
        print("  -a/--aggregate -d/--dir <directory> [--pattern <pattern>] [--top N] [--output <file>]")
        print("       for aggregated genext stats visualization")
        import sys
        sys.exit(1)
