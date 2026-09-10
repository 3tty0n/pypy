"""Render the paper figures from a $OUT result directory.

    benchmark/paper/bench.sh plot                 # every figure into $OUT/figures
    benchmark/paper/plot.py $OUT --only models    # one figure
    benchmark/paper/plot.py $OUT --format pdf,png --texture
    benchmark/paper/plot.py $OUT --compare $OTHER   # two machines side by side

Each figure ships a `.tex` tabular of the same numbers next to it, so a value
that is hard to read off the plot is still available to the reader (and to the
paper text) without going back to the tsv.

Figure forms are picked from what the data has to show, not from habit:

    micro_speedup   ratios around a 1.0 baseline   -> diverging bars from 1.0
    fusion          the same variant before/after  -> dumbbell
    models          three systems per model        -> grouped bars, linear us
    dynamic         a measure over a length axis   -> lines
    precision       two variants, three dtypes     -> small multiples
    ablation        one knob at a time             -> grouped bars

--compare adds a second (or third) result directory and switches to the
cross-machine figures, which compare ratios rather than microseconds: absolute
times are a property of the accelerator, ratios are a property of the system
under test, and the claim those figures have to support is that the ordering
between fused, torch.compile and eager survives a change of GPU.

    compare_speedup speedup per benchmark, one series per machine
    compare_models  end-to-end ratio per model, one series per machine
    compare_dynamic ratio against torch.compile over sequence length

Absolute times span four orders of magnitude across the micro grid, so those
are shown as ratios or as dumbbells on a log axis; a bar whose baseline is not
zero misencodes magnitude, so no bar chart here uses a log length.
"""

import argparse
import collections
import csv
import os
import re
import statistics
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter

# --- palette ---------------------------------------------------------------
# Slots 1-4 of the reference categorical palette, used in fixed order and never
# cycled.  Checked against a white page surface (OKLab dE x100, Machado 2009
# CVD simulation at severity 1.0); worst pair on the active pairlist:
#
#                                   CVD dE          normal dE
#   2 machines (compare figures)    9.2 (deutan)    27.6      PASS
#   3 series, adjacent + all-pairs  9.2 (deutan)    24.0      PASS
#   4 series, adjacent              9.1 (protan)    22.9      PASS
#   diverging blue<->red            21.6 (protan)   32.3      PASS
#
# Aqua (2.82:1) and yellow (2.17:1) sit below 3:1 against white, which the
# contrast check flags as "relief required": the .tex table beside each figure
# is that relief.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
POS, NEG = "#2a78d6", "#e34948"      # diverging poles, neutral midpoint is the rule line
INK, INK_2, GRID = "#0b0b0b", "#52514e", "#d9d8d4"
# Opt-in only: texture is for print and full CVD, never decoration.
HATCH = ["", "///", "...", "xxx"]

SINGLE_COL, DOUBLE_COL = 3.25, 6.75   # MLSys column and text widths, inches
# Figures whose row labels are long enough that a single column would leave no
# room for the plot itself.  --column overrides this.
WIDE = {"micro_speedup", "fusion", "models", "ablation",
        "compare_speedup", "compare_models"}

SYSTEM_LABEL = {
    "ours": "MetaTensor",
    "torch-compile": "torch.compile",
    "torch-eager": "PyTorch eager",
    "torch-compile-dynamic": "torch.compile (dynamic)",
    "torch-compile-static": "torch.compile (static)",
}
VARIANT_LABEL = {
    0: "elementwise", 1: "guard", 2: "force", 3: "value guard", 4: "exception",
    5: "side effect", 6: "MLP", 7: "MLP train", 8: "TF block", 9: "CNN",
    10: "TF train", 11: "reduction", 12: "matmul", 13: "attention",
}


def style(font):
    plt.rcParams.update({
        "font.family": font,
        "font.serif": ["STIX Two Text", "DejaVu Serif", "Times New Roman"],
        "font.sans-serif": ["DejaVu Sans", "Helvetica", "Arial"],
        "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8,
        "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
        # Hairline, solid, recessive chrome - never dashed.
        "axes.edgecolor": GRID, "axes.linewidth": 0.6,
        "axes.labelcolor": INK, "text.color": INK,
        "xtick.color": INK_2, "ytick.color": INK_2,
        "xtick.major.width": 0.6, "ytick.major.width": 0.6,
        "xtick.major.size": 2.5, "ytick.major.size": 2.5,
        "grid.color": GRID, "grid.linewidth": 0.6, "grid.linestyle": "-",
        "legend.frameon": False, "legend.handlelength": 1.6,
        "figure.dpi": 200, "savefig.dpi": 200,
        "figure.constrained_layout.use": True,
        "figure.constrained_layout.h_pad": 0.02,
        "figure.constrained_layout.w_pad": 0.02,
        # Embed TrueType rather than Type 3, which most venues reject.
        "pdf.fonttype": 42, "ps.fonttype": 42,
    })


def despine(ax, keep=("left", "bottom")):
    for side, spine in ax.spines.items():
        spine.set_visible(side in keep)


# --- data ------------------------------------------------------------------

def read_tsv(path):
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return list(csv.DictReader(f, delimiter="\t"))


def med(xs):
    xs = [float(x) for x in xs if x not in (None, "")]
    return statistics.median(xs) if xs else None


def micro_groups(rows):
    """(dtype, variant, k, n) -> {mode: median steady_us}.

    The dtype lands in a different column for our rows and torch's, exactly as
    summarize.py reads it; keep the two in step.
    """
    g = collections.defaultdict(lambda: collections.defaultdict(list))
    for r in rows:
        dtype = r.get("breaks") or r.get("graphs") or "float64"
        key = (dtype, int(r["variant"]), int(r["k"]), int(r["n"]))
        g[key][r["mode"]].append(float(r["steady_us"]))
    return {k: {m: med(v) for m, v in d.items()} for k, d in g.items()}


def micro_label(variant, k, n):
    name = VARIANT_LABEL.get(variant, "v%d" % variant)
    if variant <= 5:
        return "%s k=%d n=%s" % (name, k, si(n))
    return "%s n=%s" % (name, si(n))


def si(n):
    for div, suf in ((1000000, "M"), (1000, "k")):
        if n >= div:
            v = n / float(div)
            return "%s%s" % (("%.1f" % v).rstrip("0").rstrip("."), suf)
    return str(n)


# --- table companion -------------------------------------------------------

def write_table(path, caption, header, rows, align=None):
    align = align or ("l" + "r" * (len(header) - 1))
    esc = lambda s: str(s).replace("_", r"\_").replace("%", r"\%")
    out = [r"\begin{tabular}{%s}" % align, r"\toprule",
           " & ".join(esc(h) for h in header) + r" \\", r"\midrule"]
    out += [" & ".join(esc(c) for c in row) + r" \\" for row in rows]
    out += [r"\bottomrule", r"\end{tabular}", "%% %s" % caption]
    with open(path, "w") as f:
        f.write("\n".join(out) + "\n")


# --- figures ---------------------------------------------------------------

def fig_micro_speedup(out, args):
    """Ratios sit either side of a 1.0 baseline, so the form is a diverging bar."""
    g = micro_groups(read_tsv(os.path.join(out, "micro.tsv")))
    pts = []
    for (dtype, v, k, n), m in g.items():
        if dtype != "float64" or not m.get("fused") or not m.get("torch-compile"):
            continue
        pts.append((m["torch-compile"] / m["fused"], micro_label(v, k, n)))
    if not pts:
        return None
    pts.sort()
    vals = [p[0] for p in pts]
    labels = [p[1] for p in pts]
    y = range(len(pts))

    fig, ax = plt.subplots(figsize=(args.width, 0.16 * len(pts) + 0.75))
    colors = [POS if v >= 1 else NEG for v in vals]
    # Bars grow from the 1.0 rule, which is the neutral midpoint of the scale.
    ax.barh(list(y), [v - 1 for v in vals], left=1, height=0.55,
            color=colors, linewidth=0,
            hatch=(HATCH[0] if not args.texture else None))
    ax.axvline(1, color=INK_2, linewidth=0.8, zorder=3)
    ax.set_xscale("log", base=2)
    ax.set_xticks([0.5, 1, 2, 4, 8, 16])
    ax.xaxis.set_major_formatter(FuncFormatter(
        lambda v, _: ("%g" % v).rstrip("0").rstrip(".") + "\u00d7"))
    ax.set_yticks(list(y), labels)
    ax.set_xlim(min(vals) / 1.6, max(vals) * 1.35)
    ax.set_xlabel("speedup over torch.compile")
    ax.xaxis.grid(True, zorder=0)
    ax.set_axisbelow(True)
    despine(ax, keep=("left",))
    ax.tick_params(axis="y", length=0)
    # Direct-label the extremes only; the rest are in the table.
    for i in (0, len(pts) - 1):
        v = vals[i]
        ax.annotate("%.2f\u00d7" % v, (v, i), xytext=(4 if v >= 1 else -4, 0),
                    textcoords="offset points", va="center",
                    ha="left" if v >= 1 else "right", fontsize=7, color=INK_2)
    ax.legend(handles=[
        Line2D([], [], color=POS, lw=4, label="MetaTensor faster"),
        Line2D([], [], color=NEG, lw=4, label="torch.compile faster")],
        loc="lower right", bbox_to_anchor=(1.0, 0.0))
    if args.titles:
        ax.set_title("Microbenchmark speedup, float64")
    write_table(os.path.join(args.outdir, "micro_speedup.tex"),
                "steady-state us per iteration, median of rounds",
                ["benchmark", "MetaTensor", "torch.compile", "speedup"],
                [[lab, "%.1f" % g[k]["fused"], "%.1f" % g[k]["torch-compile"],
                  "%.2f" % (g[k]["torch-compile"] / g[k]["fused"])]
                 for (k, lab) in sorted(
                     ((k, micro_label(k[1], k[2], k[3])) for k in g
                      if k[0] == "float64" and g[k].get("fused")
                      and g[k].get("torch-compile")),
                     key=lambda kl: g[kl[0]]["torch-compile"] / g[kl[0]]["fused"])])
    return fig, "micro_speedup"


def fig_fusion(out, args):
    """The same benchmark before and after fusion: a dumbbell, one row each."""
    g = micro_groups(read_tsv(os.path.join(out, "micro.tsv")))
    pts = []
    for (dtype, v, k, n), m in g.items():
        if dtype != "float64" or not m.get("fused") or not m.get("nojit"):
            continue
        pts.append((m["nojit"] / m["fused"], m["nojit"], m["fused"],
                    micro_label(v, k, n)))
    if not pts:
        return None
    pts.sort()
    y = range(len(pts))

    fig, ax = plt.subplots(figsize=(args.width, 0.16 * len(pts) + 0.75))
    for i, (_, slow, fast, _lab) in enumerate(pts):
        ax.plot([fast, slow], [i, i], color=GRID, linewidth=1.2, zorder=1,
                solid_capstyle="round")
    ax.scatter([p[1] for p in pts], list(y), s=14, color=SERIES[1],
               zorder=2, linewidth=0, label="interpreted (nojit)")
    ax.scatter([p[2] for p in pts], list(y), s=14, color=SERIES[0],
               zorder=2, linewidth=0, label="fused (ours)")
    ax.set_xscale("log")
    ax.set_yticks(list(y), [p[3] for p in pts])
    ax.set_xlabel("steady-state time per iteration (\u00b5s, log)")
    ax.xaxis.grid(True, zorder=0)
    ax.set_axisbelow(True)
    despine(ax, keep=("left",))
    ax.tick_params(axis="y", length=0)
    best = pts[-1]
    ax.annotate("%.1f\u00d7" % best[0], (best[2], len(pts) - 1),
                xytext=(-4, 0), textcoords="offset points",
                ha="right", va="center", fontsize=7, color=INK_2)
    ax.legend(loc="lower right")
    if args.titles:
        ax.set_title("Effect of kernel fusion")
    write_table(os.path.join(args.outdir, "fusion.tex"),
                "steady-state us per iteration",
                ["benchmark", "fused", "nojit", "gain"],
                [[p[3], "%.1f" % p[2], "%.1f" % p[1], "%.2f" % p[0]]
                 for p in reversed(pts)])
    return fig, "fusion"


def fig_models(out, args):
    """Three systems per model, all in the same unit: grouped bars from zero."""
    rows = read_tsv(os.path.join(out, "models.tsv"))
    if not rows:
        return None
    g = collections.defaultdict(lambda: collections.defaultdict(list))
    for r in rows:
        g[r["model"]][r["system"]].append(float(r["steady_us"]))
    systems = ["ours", "torch-compile", "torch-eager"]
    models = sorted(g, key=lambda m: med(g[m].get("ours", [])) or 0)
    vals = {s: [med(g[m].get(s, [])) or 0 for m in models] for s in systems}

    h = 0.26
    fig, ax = plt.subplots(figsize=(args.width, 0.42 * len(models) + 0.8))
    y = [i for i in range(len(models))]
    for si_, s in enumerate(systems):
        off = (si_ - 1) * h
        ax.barh([v + off for v in y], vals[s], height=h * 0.88,
                color=SERIES[si_], linewidth=0,
                hatch=HATCH[si_] if args.texture else None,
                label=SYSTEM_LABEL[s])
    ax.set_yticks(y, models)
    ax.set_xlabel("steady-state time per iteration (\u00b5s)")
    ax.xaxis.grid(True, zorder=0)
    ax.set_axisbelow(True)
    despine(ax, keep=("left",))
    ax.tick_params(axis="y", length=0)
    ax.legend(loc="lower right", ncol=1)
    if args.titles:
        ax.set_title("End-to-end inference")
    write_table(os.path.join(args.outdir, "models.tex"),
                "steady-state us per iteration, median of rounds",
                ["model", "MetaTensor", "torch.compile", "eager", "ratio"],
                [[m, "%.0f" % vals["ours"][i], "%.0f" % vals["torch-compile"][i],
                  "%.0f" % vals["torch-eager"][i],
                  "%.2f" % (vals["ours"][i] / vals["torch-compile"][i])
                  if vals["torch-compile"][i] else "n/a"]
                 for i, m in enumerate(models)])
    return fig, "models"


def fig_dynamic(out, args):
    """A measure across a length axis: lines, one per system."""
    rows = read_tsv(os.path.join(out, "dynamic_summary.tsv"))
    if not rows:
        return None
    g = collections.defaultdict(list)
    for r in rows:
        g[(r["system"], int(r["length"]))].append(float(r["median_us"]))
    systems = [s for s in ["ours", "torch-compile-static",
                           "torch-compile-dynamic", "torch-eager"]
               if any(k[0] == s for k in g)]
    lengths = sorted({k[1] for k in g})

    fig, ax = plt.subplots(figsize=(args.width, args.width * 0.62))
    markers = ["o", "s", "^", "D"]
    for i, s in enumerate(systems):
        ys = [med(g[(s, L)]) for L in lengths]
        ax.plot(lengths, ys, color=SERIES[i], linewidth=1.4, marker=markers[i],
                markersize=3.4, markeredgewidth=0, label=SYSTEM_LABEL[s],
                clip_on=False)
    ax.set_xticks(lengths)
    ax.set_xlim(lengths[0] - 3, lengths[-1] + 3)
    ax.set_xlabel("sequence length (tokens)")
    ax.set_ylabel("median time per step (\u00b5s)")
    ax.set_ylim(0, None)
    ax.yaxis.grid(True, zorder=0)
    ax.set_axisbelow(True)
    despine(ax)
    ax.legend(loc="upper left")
    if args.titles:
        ax.set_title("Changing sequence length")
    write_table(os.path.join(args.outdir, "dynamic.tex"),
                "median us per step; recompiles over the whole sweep",
                ["system"] + [str(L) for L in lengths] + ["recompiles"],
                [[SYSTEM_LABEL[s]] + ["%.0f" % med(g[(s, L)]) for L in lengths]
                 + [next((r["recompiles"] for r in rows if r["system"] == s), "")]
                 for s in systems])
    return fig, "dynamic"


def fig_precision(out, args):
    """Two panels with unrelated y-scales would invite a misread, so the panels
    are collapsed into one axis of speedup: a ratio is unitless and shares a
    scale no matter how far apart the absolute times are."""
    g = micro_groups(read_tsv(os.path.join(out, "micro.tsv")))
    dtypes = ["float64", "float32", "float16"]
    variants = sorted({k[1] for k in g if k[0] != "float64"})
    if not variants:
        return None
    fig, ax = plt.subplots(figsize=(args.width, args.width * 0.62))
    x = range(len(dtypes))
    w = 0.8 / max(len(variants), 1)
    table = []
    for vi, v in enumerate(variants):
        n = max(k[3] for k in g if k[1] == v)
        ratios, absolute = [], []
        for d in dtypes:
            m = g.get((d, v, 1, n)) or {}
            ours, comp = m.get("fused"), m.get("torch-compile")
            ratios.append(comp / ours if ours and comp else 0)
            absolute.append((ours, comp))
        off = (vi - (len(variants) - 1) / 2) * w
        ax.bar([i + off for i in x], ratios, w * 0.86, color=SERIES[vi],
               linewidth=0, hatch=HATCH[vi] if args.texture else None,
               label="%s (n=%s)" % (VARIANT_LABEL.get(v, "v%d" % v), si(n)))
        for d, r, (ours, comp) in zip(dtypes, ratios, absolute):
            table.append([VARIANT_LABEL.get(v, "v%d" % v), d,
                          "%.1f" % ours if ours else "n/a",
                          "%.1f" % comp if comp else "n/a",
                          "%.2f" % r if r else "n/a"])
    ax.axhline(1, color=INK_2, linewidth=0.8, zorder=3)
    ax.set_xticks(list(x), ["float64", "float32", "float16"])
    ax.set_ylabel("speedup over torch.compile")
    ax.yaxis.set_major_formatter(FuncFormatter(
        lambda v, _: ("%g" % v) + "\u00d7"))
    ax.yaxis.grid(True, zorder=0)
    ax.set_axisbelow(True)
    despine(ax)
    ax.legend(loc="upper left")
    if args.titles:
        ax.set_title("Precision sweep")
    write_table(os.path.join(args.outdir, "precision.tex"),
                "steady-state us per iteration",
                ["variant", "dtype", "MetaTensor", "torch.compile", "speedup"],
                table)
    return fig, "precision"


def fig_ablation(out, args):
    """Each row is one knob at two settings against the same model.  The
    settings mean something different in every experiment - float16 against
    float32, fusion on against off - so painting them by position would attach
    identity to a slot rather than to a thing.  One hue, and the setting is
    written next to its own bar."""
    rows = read_tsv(os.path.join(out, "ablation.tsv"))
    if not rows:
        return None
    g = collections.defaultdict(list)
    for r in rows:
        if r.get("steady_us"):
            g[(r["experiment"], r["model"], r["variant"])].append(
                float(r["steady_us"]))
    if not g:
        return None

    def setting_key(v):
        # "8" before "64": these are numbers wherever they look like numbers.
        try:
            return (0, float(v), "")
        except ValueError:
            return (1, 0.0, v)

    groups = collections.OrderedDict()
    for exp, model, var in sorted(g, key=lambda k: (k[0], k[1], setting_key(k[2]))):
        groups.setdefault((exp, model), []).append((var, med(g[(exp, model, var)])))

    keys = list(groups)
    fig, ax = plt.subplots(figsize=(args.width, 0.52 * len(keys) + 0.8))
    height = 0.30
    ys, xs, texts, ticks = [], [], [], []
    for i, key in enumerate(keys):
        items = groups[key]
        for slot, (var, value) in enumerate(items):
            ys.append(i + ((len(items) - 1) / 2 - slot) * height)
            xs.append(value)
            texts.append(var)
        ticks.append("%s / %s" % (key[0].replace("_", " "), key[1]))
    ax.barh(ys, xs, height=height * 0.88, color=SERIES[0], linewidth=0,
            hatch=HATCH[0] if args.texture else None)
    for xv, yv, t in zip(xs, ys, texts):
        ax.annotate(t, (xv, yv), xytext=(3, 0), textcoords="offset points",
                    va="center", fontsize=6.5, color=INK_2)
    ax.set_yticks(range(len(keys)), ticks)
    ax.set_xlabel("steady-state time per iteration (\u00b5s)")
    ax.set_xlim(0, max(xs) * 1.18)
    ax.xaxis.grid(True, zorder=0)
    ax.set_axisbelow(True)
    despine(ax, keep=("left",))
    ax.tick_params(axis="y", length=0)
    if args.titles:
        ax.set_title("Ablations")
    write_table(os.path.join(args.outdir, "ablation.tex"),
                "steady-state us per iteration, median of rounds",
                ["experiment", "model", "setting", "us"],
                [[e.replace("_", " "), m, v, "%.0f" % med(g[(e, m, v)])]
                 for (e, m, v) in sorted(g, key=lambda k: (k[0], k[1], setting_key(k[2])))])
    return fig, "ablation"


# --- machines --------------------------------------------------------------

def machine_label(path):
    """Name a result directory by the accelerator that produced it.

    machine.txt is authoritative; runs recorded before it existed fall back to
    the <host>-<gpu> directory name, which is why the layout carries both.
    """
    host = gpu = None
    txt = os.path.join(path, "machine.txt")
    if os.path.exists(txt):
        for line in open(txt):
            key, _, value = line.partition(" ")
            value = value.strip()
            if key == "host" and value:
                host = value
            elif key == "gpu" and value:
                gpu = value
    if not (host and gpu):
        parent = os.path.basename(os.path.dirname(os.path.abspath(path)))
        if "-" in parent:
            h, _, g = parent.partition("-")
            host = host or h
            gpu = gpu or g
    gpu = pretty_gpu(gpu or "unknown GPU")
    return "%s (%s)" % (gpu, host) if host else gpu


def pretty_gpu(name):
    name = name.replace("NVIDIA ", "").replace("GeForce ", "").strip()
    m = re.match(r"^rtx(\d{3,4})(ti|super)?$", name, re.I)
    if m:
        return "RTX %s%s" % (m.group(1), " Ti" if (m.group(2) or "").lower() == "ti"
                             else " SUPER" if m.group(2) else "")
    return name


def _speedup_rows(out):
    """benchmark label -> fused speedup over torch.compile, float64 only."""
    g = micro_groups(read_tsv(os.path.join(out, "micro.tsv")))
    rows = {}
    for (dtype, v, k, n), m in g.items():
        if dtype != "float64" or not m.get("fused") or not m.get("torch-compile"):
            continue
        rows[(v, k, n)] = (micro_label(v, k, n),
                           m["torch-compile"] / m["fused"])
    return rows


def _grouped_ratio_bars(ax, keys, per_machine, labels, args, xlabel):
    """One row per benchmark, one bar per machine, all growing from the 1.0 rule."""
    y = list(range(len(keys)))
    h = 0.8 / len(labels)
    for mi, label in enumerate(labels):
        vals = [per_machine[mi][k] for k in keys]
        off = ((len(labels) - 1) / 2 - mi) * h
        ax.barh([i + off for i in y], [v - 1 for v in vals], left=1,
                height=h * 0.86, color=SERIES[mi], linewidth=0,
                hatch=HATCH[mi] if args.texture else None, label=label)
    ax.axvline(1, color=INK_2, linewidth=0.8, zorder=3)
    ax.set_xscale("log", base=2)
    ax.xaxis.set_major_formatter(FuncFormatter(
        lambda v, _: ("%g" % v).rstrip("0").rstrip(".") + "\u00d7"))
    ax.set_xlabel(xlabel)
    ax.xaxis.grid(True, zorder=0)
    ax.set_axisbelow(True)
    despine(ax, keep=("left",))
    ax.tick_params(axis="y", length=0)


def fig_compare_speedup(out, args):
    """The same speedup measured on each machine, so the reader can see whether
    the ordering holds where the absolute times do not."""
    outs = [out] + args.compare
    rows = [_speedup_rows(o) for o in outs]
    labels = [machine_label(o) for o in outs]
    common = set(rows[0])
    for r in rows[1:]:
        common &= set(r)
    dropped = sorted(set().union(*(set(r) for r in rows)) - common)
    if not common:
        return None
    keys = sorted(common, key=lambda k: rows[0][k][1])
    per = [{k: r[k][1] for k in keys} for r in rows]

    fig, ax = plt.subplots(figsize=(args.width, 0.22 * len(keys) + 0.9))
    _grouped_ratio_bars(ax, keys, per, labels, args,
                        "speedup over torch.compile")
    ax.set_yticks(list(range(len(keys))), [rows[0][k][0] for k in keys])
    ax.legend(loc="lower right")
    if args.titles:
        ax.set_title("Microbenchmark speedup across machines")
    if dropped:
        print("compare_speedup: %d point(s) not measured at the same size on "
              "every machine, left out: %s"
              % (len(dropped), ", ".join(micro_label(*k) for k in dropped)),
              file=sys.stderr)
    write_table(os.path.join(args.outdir, "compare_speedup.tex"),
                "fused speedup over torch.compile, float64",
                ["benchmark"] + labels,
                [[rows[0][k][0]] + ["%.2f" % p[k] for p in per] for k in keys])
    return fig, "compare_speedup"


def fig_compare_models(out, args):
    """End-to-end ratio to torch.compile, per model, per machine."""
    outs = [out] + args.compare
    labels = [machine_label(o) for o in outs]
    per = []
    for o in outs:
        g = collections.defaultdict(lambda: collections.defaultdict(list))
        for r in read_tsv(os.path.join(o, "models.tsv")):
            g[r["model"]][r["system"]].append(float(r["steady_us"]))
        per.append({m: med(d.get("torch-compile", [])) / med(d["ours"])
                    for m, d in g.items()
                    if d.get("ours") and d.get("torch-compile")})
    common = set(per[0])
    for p in per[1:]:
        common &= set(p)
    if not common:
        return None
    keys = sorted(common, key=lambda m: per[0][m])

    fig, ax = plt.subplots(figsize=(args.width, 0.30 * len(keys) + 0.9))
    _grouped_ratio_bars(ax, keys, per, labels, args,
                        "speedup over torch.compile")
    ax.set_yticks(list(range(len(keys))), keys)
    ax.legend(loc="lower right")
    if args.titles:
        ax.set_title("End-to-end speedup across machines")
    write_table(os.path.join(args.outdir, "compare_models.tex"),
                "end-to-end speedup over torch.compile",
                ["model"] + labels,
                [[m] + ["%.2f" % p[m] for p in per] for m in keys])
    return fig, "compare_models"


def fig_compare_dynamic(out, args):
    """Cost against sequence length, as a ratio so two GPUs share one axis."""
    outs = [out] + args.compare
    labels = [machine_label(o) for o in outs]
    series, lengths = [], None
    for o in outs:
        g = collections.defaultdict(list)
        for r in read_tsv(os.path.join(o, "dynamic_summary.tsv")):
            g[(r["system"], int(r["length"]))].append(float(r["median_us"]))
        if not g:
            return None
        ls = sorted({k[1] for k in g})
        lengths = ls if lengths is None else [L for L in lengths if L in ls]
        series.append(g)
    if not lengths:
        return None

    fig, ax = plt.subplots(figsize=(args.width, args.width * 0.62))
    markers = ["o", "s", "^", "D"]
    for mi, (g, label) in enumerate(zip(series, labels)):
        ys = []
        for L in lengths:
            ours = med(g.get(("ours", L), []))
            comp = med(g.get(("torch-compile-dynamic", L), []))
            ys.append(comp / ours if ours and comp else float("nan"))
        ax.plot(lengths, ys, color=SERIES[mi], linewidth=1.4,
                marker=markers[mi], markersize=3.4, markeredgewidth=0,
                label=label, clip_on=False)
    ax.axhline(1, color=INK_2, linewidth=0.8, zorder=3)
    ax.set_xticks(lengths)
    ax.set_xlim(lengths[0] - 3, lengths[-1] + 3)
    ax.set_xlabel("sequence length (tokens)")
    ax.set_ylabel("speedup over torch.compile")
    ax.yaxis.set_major_formatter(FuncFormatter(
        lambda v, _: ("%g" % v) + "\u00d7"))
    ax.yaxis.grid(True, zorder=0)
    ax.set_axisbelow(True)
    despine(ax)
    ax.legend(loc="best")
    if args.titles:
        ax.set_title("Dynamic sequence length across machines")
    write_table(os.path.join(args.outdir, "compare_dynamic.tex"),
                "speedup over torch.compile (dynamic) per sequence length",
                ["machine"] + [str(L) for L in lengths],
                [[label] + ["%.2f" % (med(g.get(("torch-compile-dynamic", L), []))
                                      / med(g.get(("ours", L), [])))
                            for L in lengths]
                 for g, label in zip(series, labels)])
    return fig, "compare_dynamic"


FIGURES = collections.OrderedDict([
    ("micro_speedup", fig_micro_speedup),
    ("fusion", fig_fusion),
    ("models", fig_models),
    ("dynamic", fig_dynamic),
    ("precision", fig_precision),
    ("ablation", fig_ablation),
])

COMPARE_FIGURES = collections.OrderedDict([
    ("compare_speedup", fig_compare_speedup),
    ("compare_models", fig_compare_models),
    ("compare_dynamic", fig_compare_dynamic),
])


def main(argv=None):
    here = os.path.dirname(os.path.abspath(__file__))
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("out", nargs="?", default=os.environ.get("OUT"),
                   help="result directory holding micro.tsv etc.")
    p.add_argument("--outdir", help="where the figures go (default $OUT/figures)")
    p.add_argument("--compare", action="append", default=[], metavar="DIR",
                   help="a second result directory to compare against "
                        "(repeatable, at most three in total)")
    p.add_argument("--only", action="append",
                   choices=list(FIGURES) + list(COMPARE_FIGURES),
                   help="render just this figure (repeatable)")
    p.add_argument("--format", default="pdf",
                   help="comma-separated: pdf, png, svg (default pdf)")
    p.add_argument("--column", choices=["single", "double"],
                   help="figure width: one MLSys column or the full text width "
                        "(default: whichever suits the figure)")
    p.add_argument("--font", choices=["serif", "sans-serif"], default="serif")
    p.add_argument("--texture", action="store_true",
                   help="hatch the fills, for grayscale print and full CVD")
    p.add_argument("--titles", action="store_true",
                   help="draw titles on the axes (for slides; a paper uses captions)")
    args = p.parse_args(argv)

    if not args.out:
        p.error("no result directory given and $OUT is unset")
    if not os.path.isdir(args.out):
        p.error("%s is not a directory" % args.out)
    for d in args.compare:
        if not os.path.isdir(d):
            p.error("%s is not a directory" % d)
    # The categorical palette is used in fixed order and never cycled, so a
    # fourth machine would need a hue that does not exist rather than a
    # generated one.
    if len(args.compare) > 2:
        p.error("at most three result directories can share a figure")
    args.outdir = args.outdir or os.path.join(args.out, "figures")
    explicit_width = args.column is not None
    os.makedirs(args.outdir, exist_ok=True)
    style(args.font)

    table = dict(FIGURES)
    table.update(COMPARE_FIGURES)
    default = list(COMPARE_FIGURES) if args.compare else list(FIGURES)
    wanted = args.only or default
    written = []
    for name in wanted:
        if name in COMPARE_FIGURES and not args.compare:
            p.error("%s needs --compare" % name)
        if explicit_width:
            args.width = SINGLE_COL if args.column == "single" else DOUBLE_COL
        else:
            args.width = DOUBLE_COL if name in WIDE else SINGLE_COL
        made = table[name](args.out, args)
        if made is None:
            print("skip %s (no rows in %s)" % (name, args.out), file=sys.stderr)
            continue
        fig, stem = made
        for ext in args.format.split(","):
            path = os.path.join(args.outdir, "%s.%s" % (stem, ext.strip()))
            fig.savefig(path)
            written.append(path)
        plt.close(fig)
    for path in written:
        print(path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
