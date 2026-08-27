#!/usr/bin/env python3
"""Where does JIT time go?  Stacked bars per benchmark and config from
jit-summary logs (PYPYLOG=jit-summary:...) plus wall time.

    plot_jit_breakdown.py --pattern 'dec-{cfg}-{bench}.log' \\
        --time-pattern 'time-{cfg}-{bench}.txt' \\
        --configs A,B,C,D --benches eparse,go -o breakdown.pdf

Page 1: seconds of Tracing / Optimizing / Backend / Blackhole / PE cogen,
with wall time as a marker.  Page 2: loops, bridges, aborts.
"""

import argparse
import os
import re

TIMES = [("Tracing", "tracing"), ("Optimizing", "optimizing"),
         ("Backend", "backend"), ("Blackhole", "blackhole"),
         ("PE cogen scan", "cogen"), ("PE cogen install", "cogen")]
COUNTS = [("Total # of loops", "loops"), ("Total # of bridges", "bridges")]


def parse_summary(path):
    out = {"tracing": 0.0, "optimizing": 0.0, "backend": 0.0,
           "blackhole": 0.0, "cogen": 0.0, "loops": 0, "bridges": 0,
           "aborts": 0, "total": 0.0}
    with open(path) as stream:
        for line in stream:
            line = re.sub(r"^\[[0-9a-f]+\] ", "", line).rstrip()
            for label, key in TIMES:
                m = re.match(r"^%s:\s+(\d+)\s+([\d.]+)$" % re.escape(label),
                             line)
                if m:
                    out[key] += float(m.group(2))
            for label, key in COUNTS:
                m = re.match(r"^%s:\s+(\d+)$" % re.escape(label), line)
                if m:
                    out[key] = int(m.group(1))
            m = re.match(r"^abort: [a-z -]+:\s+(\d+)$", line)
            if m:
                out["aborts"] += int(m.group(1))
            m = re.match(r"^TOTAL:\s+([\d.]+)$", line)
            if m:
                out["total"] = float(m.group(1))
    return out


def parse_time(path):
    if path is None or not os.path.exists(path):
        return None
    with open(path) as stream:
        for line in stream:
            m = re.match(r"^real\s+([\d.]+)", line)
            if m:
                return float(m.group(1))
    return None


def load(pattern, time_pattern, configs, benches):
    data = {}
    for bench in benches:
        for cfg in configs:
            path = pattern.format(cfg=cfg, bench=bench)
            if not os.path.exists(path):
                continue
            row = parse_summary(path)
            tpath = time_pattern.format(cfg=cfg, bench=bench) \
                if time_pattern else None
            row["real"] = parse_time(tpath)
            data[bench, cfg] = row
    return data


def plot(data, configs, benches, output, title):
    import matplotlib
    matplotlib.use("pdf")
    from matplotlib import pyplot
    from matplotlib.backends.backend_pdf import PdfPages

    parts = [("tracing", "C0"), ("optimizing", "C1"), ("backend", "C2"),
             ("blackhole", "C3"), ("cogen", "C4")]
    cols = 4
    lines = (len(benches) + cols - 1) // cols
    width = 0.8
    with PdfPages(output) as pdf:
        fig, axes = pyplot.subplots(lines, cols,
                                    figsize=(4 * cols, 3 * lines),
                                    squeeze=False)
        for ax, bench in zip(axes.flat, benches):
            xs = list(range(len(configs)))
            bottom = [0.0] * len(configs)
            for key, color in parts:
                vals = [data.get((bench, c), {}).get(key, 0.0)
                        for c in configs]
                ax.bar(xs, vals, width, bottom=bottom, color=color,
                       label=key)
                bottom = [b + v for b, v in zip(bottom, vals)]
            reals = [data.get((bench, c), {}).get("real") for c in configs]
            if any(r is not None for r in reals):
                ax2 = ax.twinx()
                ax2.plot(xs, [r if r is not None else float("nan")
                              for r in reals], "k_", markersize=18,
                         markeredgewidth=2, label="wall")
                ax2.set_ylim(0, max(r for r in reals if r) * 1.15)
                ax2.tick_params(labelsize=7)
            ax.set_xticks(xs)
            ax.set_xticklabels(configs, fontsize=8)
            ax.set_title(bench, fontsize=9)
            ax.tick_params(labelsize=7)
            ax.set_ylabel("JIT seconds", fontsize=7)
        for ax in list(axes.flat)[len(benches):]:
            ax.axis("off")
        axes.flat[0].legend(fontsize=7)
        fig.suptitle(title + "  (bars: JIT time; dash: wall time)")
        fig.tight_layout(rect=(0, 0, 1, 0.96))
        pdf.savefig(fig)
        pyplot.close(fig)

        fig, axes = pyplot.subplots(lines, cols,
                                    figsize=(4 * cols, 3 * lines),
                                    squeeze=False)
        for ax, bench in zip(axes.flat, benches):
            xs = list(range(len(configs)))
            for offset, key, color in ((-0.27, "loops", "C0"),
                                       (0.0, "bridges", "C3"),
                                       (0.27, "aborts", "C1")):
                vals = [data.get((bench, c), {}).get(key, 0)
                        for c in configs]
                ax.bar([x + offset for x in xs], vals, 0.27, color=color,
                       label=key)
            ax.set_yscale("symlog")
            ax.set_xticks(xs)
            ax.set_xticklabels(configs, fontsize=8)
            ax.set_title(bench, fontsize=9)
            ax.tick_params(labelsize=7)
        for ax in list(axes.flat)[len(benches):]:
            ax.axis("off")
        axes.flat[0].legend(fontsize=7)
        fig.suptitle(title + "  (counts, symlog)")
        fig.tight_layout(rect=(0, 0, 1, 0.96))
        pdf.savefig(fig)
        pyplot.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pattern", required=True)
    parser.add_argument("--time-pattern", default=None)
    parser.add_argument("--configs", required=True)
    parser.add_argument("--benches", required=True)
    parser.add_argument("-o", "--output", default="breakdown.pdf")
    parser.add_argument("--title", default="JIT time breakdown")
    opts = parser.parse_args()
    configs = opts.configs.split(",")
    benches = opts.benches.split(",")
    data = load(opts.pattern, opts.time_pattern, configs, benches)
    for bench in benches:
        for cfg in configs:
            row = data.get((bench, cfg))
            if row is None:
                continue
            print("%-22s %-4s real %-6s trace %.3f opt %.3f backend %.3f "
                  "blackhole %.3f cogen %.3f loops %d bridges %d aborts %d"
                  % (bench, cfg, row["real"], row["tracing"],
                     row["optimizing"], row["backend"], row["blackhole"],
                     row["cogen"], row["loops"], row["bridges"],
                     row["aborts"]))
    plot(data, configs, benches, opts.output, opts.title)


if __name__ == "__main__":
    main()
