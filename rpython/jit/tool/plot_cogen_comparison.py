#!/usr/bin/env python3
"""Plot warmup and steady changes from two PyPy suite raw-result files.

The forward file uses baseline=old and changed=new.  The reverse file uses
baseline=new and changed=old.  Rendering uses matplotlib's PDF backend.
"""

import argparse
import json
import math
import os


METRICS = ("first", "early", "stable", "total")


def load(path):
    with open(path) as stream:
        data = json.load(stream)
    return dict((row[0], row[2]) for row in data["results"])


def gmean(values):
    return math.exp(sum(math.log(value) for value in values) / len(values))


def median(values):
    values = sorted(values)
    middle = len(values) // 2
    if len(values) % 2:
        return values[middle]
    return (values[middle - 1] + values[middle]) / 2.0


def summarize(values):
    count = len(values)
    early_count = min(5, max(1, count // 2))
    return {
        "first": values[0],
        "early": sum(values[:early_count]),
        "stable": median(values[count // 2:]),
        "total": sum(values),
    }


def compare(forward, reverse):
    rows = []
    for name in sorted(set(forward).intersection(reverse)):
        pairs = (
            (summarize(forward[name]["base_times"]),
             summarize(forward[name]["changed_times"])),
            (summarize(reverse[name]["changed_times"]),
             summarize(reverse[name]["base_times"])),
        )
        row = {"name": name}
        for metric in METRICS:
            ratios = [new[metric] / old[metric] for old, new in pairs]
            row[metric] = (gmean(ratios) - 1.0) * 100.0
        rows.append(row)
    return rows


def suite_summary(rows):
    return dict((metric, (gmean(
        [1.0 + row[metric] / 100.0 for row in rows]) - 1.0) * 100.0)
        for metric in METRICS)


def format_pct(value):
    return "%+.2f%%" % value


def import_matplotlib():
    try:
        import matplotlib
    except ImportError:
        raise SystemExit(
            "matplotlib is required; install it with "
            "'python3 -m pip install matplotlib'")
    matplotlib.use("pdf")
    import matplotlib.pyplot as pyplot
    from matplotlib.lines import Line2D
    return pyplot, Line2D


def plot(rows, output, title, sort_key):
    pyplot, Line2D = import_matplotlib()
    colors = {"first": "#215aa8", "early": "#e67e22",
              "stable": "#208e5b"}
    markers = {"first": "^", "early": "s", "stable": "D"}
    if sort_key == "name":
        ordered = sorted(rows, key=lambda row: row["name"])
    else:
        ordered = sorted(rows, key=lambda row: row[sort_key])

    largest = max(abs(row[metric]) for row in rows
                  for metric in ("first", "early", "stable"))
    limit = max(5.0, math.ceil(largest / 5.0) * 5.0)
    tick_step = 5 if limit <= 25 else 10

    pyplot.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 8,
        "axes.edgecolor": "#b7babd",
        "axes.labelcolor": "#4d5258",
        "xtick.color": "#5e646b",
        "ytick.color": "#202328",
        "pdf.fonttype": 42,
    })
    figure = pyplot.figure(figsize=(11, 8.5), facecolor="#fbfbf8")
    axes = figure.add_axes((0.17, 0.065, 0.67, 0.78))
    axes.set_facecolor("#fbfbf8")
    figure.text(0.04, 0.955, title, fontsize=17, weight="bold",
                color="#161a20")

    summary = suite_summary(rows)
    summaries = (("First", "first"), ("Early", "early"),
                 ("Steady", "stable"), ("TOTAL", "total"))
    for index, (label, metric) in enumerate(summaries):
        x = 0.445 + index * 0.132
        figure.text(x, 0.908, label, fontsize=6.5, color="#60676f",
                    weight="bold", ha="right")
        value = summary[metric]
        color = "#19824c" if value < 0 else "#c53329"
        figure.text(x + 0.008, 0.908, format_pct(value), fontsize=8,
                    color=color, ha="left")

    positions = list(range(len(ordered)))
    axes.set_ylim(-0.7, len(ordered) - 0.3)
    axes.invert_yaxis()
    for index, row in enumerate(ordered):
        if row["early"] > 1.0:
            axes.axhspan(index - 0.5, index + 0.5, color="#fff0ee",
                         zorder=0)
        elif row["early"] < -1.0:
            axes.axhspan(index - 0.5, index + 0.5, color="#edf8ef",
                         zorder=0)
        axes.plot((row["first"], row["stable"]),
                  (index - 0.18, index + 0.18), color="#aeb3b8",
                  linewidth=0.55, zorder=1)
        for offset, metric in ((-0.18, "first"), (0, "early"),
                               (0.18, "stable")):
            size = 26 if metric == "early" else 16
            axes.scatter(row[metric], index + offset, s=size,
                         marker=markers[metric], color=colors[metric],
                         edgecolors="none", zorder=3)
        total_color = ("#19824c" if row["total"] < -1.0 else
                       "#c53329" if row["total"] > 1.0 else "#596069")
        axes.text(1.075, index, format_pct(row["total"]),
                  transform=axes.get_yaxis_transform(), va="center",
                  fontsize=7, color=total_color, clip_on=False)

    axes.set_yticks(positions)
    axes.set_yticklabels([row["name"] for row in ordered], fontsize=7)
    axes.tick_params(axis="y", length=0, pad=6)
    axes.set_xlim(-limit, limit)
    axes.set_xticks(list(range(-int(limit), int(limit) + 1, tick_step)))
    axes.axvline(0, color="#777d84", linewidth=0.9, zorder=0)
    axes.grid(axis="x", color="#d9dcdf", linewidth=0.5)
    axes.set_axisbelow(True)
    axes.spines["top"].set_visible(False)
    axes.spines["right"].set_visible(False)
    axes.spines["left"].set_visible(False)
    axes.set_xlabel("Runtime change (%) - lower is better", fontsize=8,
                    weight="bold")
    figure.text(0.885, 0.855, "TOTAL", fontsize=7, weight="bold",
                color="#626971")

    handles = [Line2D([], [], linestyle="none", marker=markers[metric],
                      markerfacecolor=colors[metric], markeredgecolor="none",
                      markersize=5, label=label)
               for metric, label in (
                   ("first", "First: first sample"),
                   ("early", "Early: first up to 5 samples"),
                   ("stable", "Steady: median of latter half"))]
    axes.legend(handles=handles, loc="lower center",
                bbox_to_anchor=(0.5, 1.005), ncol=3, frameon=False,
                handletextpad=0.35, columnspacing=1.4, fontsize=7)
    figure.text(
        0.04, 0.018,
        ("Sorted by Early warmup improvement. Order-balanced against the "
         "baseline; TOTAL is the sum of all samples in the right column."),
        fontsize=6, color="#666d75")

    directory = os.path.dirname(os.path.abspath(output))
    if not os.path.isdir(directory):
        os.makedirs(directory)
    figure.savefig(output, format="pdf", facecolor=figure.get_facecolor(),
                   metadata={"Title": title, "Creator": "matplotlib"})
    pyplot.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("forward")
    parser.add_argument("reverse")
    parser.add_argument("-o", "--output",
                        default="output/pdf/cogen-benchmark-comparison.pdf")
    parser.add_argument("--title", default="PyPy online cogen benchmark suite")
    parser.add_argument("--sort",
                        choices=("first", "early", "stable", "total",
                                 "name"), default="early")
    args = parser.parse_args()
    rows = compare(load(args.forward), load(args.reverse))
    if not rows:
        raise SystemExit("no common benchmark results")
    plot(rows, args.output, args.title, args.sort)
    print("wrote %s (%d benchmarks)" % (args.output, len(rows)))


if __name__ == "__main__":
    main()
