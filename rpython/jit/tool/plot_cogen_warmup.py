#!/usr/bin/env python3
"""Plot warmup curves and warm/steady ratios from one cogen_bench.py JSON.

Page 1: per-benchmark iteration time, baseline vs changed (log y).
Page 2: changed/baseline ratio of warm (sum of the first 5 iterations)
and steady (mean of the last half), one bar pair per benchmark.
"""

import argparse
import json
import math

WARM = 5


def load(path):
    with open(path) as stream:
        data = json.load(stream)
    return dict((row[0], row[2]) for row in data["results"]
                if "base_times" in row[2])


def warm_steady(times):
    half = times[len(times) // 2:]
    return sum(times[:WARM]), sum(half) / len(half)


def ratios(results):
    rows = []
    for name in sorted(results):
        bw, bs = warm_steady(results[name]["base_times"])
        cw, cs = warm_steady(results[name]["changed_times"])
        rows.append((name, cw / bw, cs / bs))
    return rows


def gmean(values):
    return math.exp(sum(math.log(v) for v in values) / len(values))


def plot(results, output, title):
    import matplotlib
    matplotlib.use("pdf")
    from matplotlib import pyplot
    from matplotlib.backends.backend_pdf import PdfPages

    names = sorted(results)
    cols = 4
    lines = int(math.ceil(len(names) / float(cols)))
    with PdfPages(output) as pdf:
        fig, axes = pyplot.subplots(lines, cols,
                                    figsize=(4 * cols, 2.6 * lines))
        for ax, name in zip(axes.flat, names):
            r = results[name]
            ax.plot(r["base_times"], label="baseline", color="gray")
            ax.plot(r["changed_times"], label="changed", color="C3")
            ax.set_yscale("log")
            ax.set_title(name, fontsize=9)
            ax.tick_params(labelsize=7)
        for ax in list(axes.flat)[len(names):]:
            ax.axis("off")
        axes.flat[0].legend(fontsize=7)
        fig.suptitle(title)
        fig.tight_layout(rect=(0, 0, 1, 0.97))
        pdf.savefig(fig)
        pyplot.close(fig)

        rows = ratios(results)
        rows.append(("geomean", gmean([r[1] for r in rows]),
                     gmean([r[2] for r in rows])))
        fig, ax = pyplot.subplots(figsize=(max(8, 0.45 * len(rows)), 4.5))
        xs = range(len(rows))
        ax.bar([x - 0.2 for x in xs], [r[1] for r in rows], 0.4,
               label="warm (first %d)" % WARM, color="C0")
        ax.bar([x + 0.2 for x in xs], [r[2] for r in rows], 0.4,
               label="steady (last half)", color="C3")
        ax.axhline(1.0, color="black", linewidth=0.8)
        ax.set_xticks(list(xs))
        ax.set_xticklabels([r[0] for r in rows], rotation=90, fontsize=7)
        ax.set_ylabel("changed / baseline")
        ax.axvline(len(rows) - 1.5, color="black", linewidth=0.8,
                   linestyle=":")
        ax.set_title("%s  gmean warm %.3f  steady %.3f" % (
            title, rows[-1][1], rows[-1][2]))
        ax.legend(fontsize=8)
        fig.tight_layout()
        pdf.savefig(fig)
        pyplot.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result")
    parser.add_argument("-o", "--output", default="warmup.pdf")
    parser.add_argument("--title", default="cogen warmup")
    opts = parser.parse_args()
    results = load(opts.result)
    for name, w, s in ratios(results):
        print("%-24s warm %.3f  steady %.3f" % (name, w, s))
    plot(results, opts.output, opts.title)


if __name__ == "__main__":
    main()
