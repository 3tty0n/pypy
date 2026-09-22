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
    batch           per-sequence cost over a batch axis -> lines, log-log
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
import glob
import json
import os
import math
import re
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import matplotlib
import matplotlib.ticker
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
SERIES = ["#1a4fa0", "#eb6834", "#1baf7a", "#eda100", "#8a5cd6", "#6b6b6b",
          "#c04a8a", "#4fa8e0", "#a0522d"]
POS, NEG = "#1a4fa0", "#e34948"      # diverging poles, neutral midpoint is the rule line
INK, INK_2, GRID = "#0b0b0b", "#52514e", "#d9d8d4"
# Opt-in only: texture is for print and full CVD, never decoration.
HATCH = ["", "///", "...", "xxx", "\\\\", "++", "oo", "--", "||"]

SINGLE_COL, DOUBLE_COL = 3.25, 6.75   # MLSys column and text widths, inches
# Figures the paper does not include at column width; they are drawn at the
# full text width.  --column overrides this.
WIDE = {"micro_speedup", "models", "ablation", "micro_baselines",
        "compile_overhead", "integration", "lazy"}
# The width the paper actually draws the figure at, so nothing is rescaled on
# the page and 7pt in the pdf is 7pt in print.  \columnwidth is 3.25in; the
# entries are the \includegraphics fractions in sections/*.tex.
PAGE_WIDTH = {
    "lazy_col": 0.98 * SINGLE_COL,
    "batch": 0.82 * SINGLE_COL,
    "models_col": 0.85 * SINGLE_COL,
    "micro_speedup_col": 0.90 * SINGLE_COL,
}

SYSTEM_LABEL = {
    "ours": "MetaTensor",
    "torch-compile": "torch.compile",
    "torch-compile-ro": "torch.compile (reduce-overhead)",
    "torch-compile-mat": "torch.compile (max-autotune)",
    "torch-eager": "PyTorch eager",
    "torch-compile-dynamic": "torch.compile (dynamic)",
    "torch-compile-static": "torch.compile (static)",
    "jax": "JAX/XLA",
    "iree": "IREE",
    "torch-tensorrt": "Torch-TensorRT",
    "triton": "Triton (handwritten)",
    # where the DAG lives (bench.sh lazy)
    "virtual": "MetaTensor (DAG in the JIT)",
    "deferred": "deferred library (DAG at run time)",
    "eager": "eager (no DAG)",
    # early exit (bench.sh control)
    "jax-perlayer": "JAX/XLA (per-layer jit, host sync)",
    "jax-while": "JAX/XLA (lax.while_loop)",
}
# Names come from benchmarks.toml, the same file the grid is read from, so a
# new benchmark is named once. A run whose variants predate an entry still
# plots - the fallback is the bare variant number.
try:
    import grid as _grid
    VARIANT_LABEL = _grid.labels(_grid.load())
except Exception:  # no benchmarks.toml next to this checkout
    VARIANT_LABEL = {}


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


def read_jsonl(path):
    if not os.path.exists(path):
        return []
    out = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except ValueError:
                continue
    return out


def med(xs):
    xs = [float(x) for x in xs if x not in (None, "")]
    return statistics.median(xs) if xs else None


def micro_values(rows):
    """The dtype lands in a different tsv column for our rows and torch's;
    read it the way summarize.py does and keep the two in step."""
    g = collections.defaultdict(lambda: collections.defaultdict(list))
    for r in rows:
        dtype = r.get("breaks") or r.get("graphs") or "float64"
        key = (dtype, int(r["variant"]), int(r["k"]), int(r["n"]))
        g[key][r["mode"]].append(float(r["steady_us"]))
    return g


def micro_groups(rows):
    return {k: {m: med(v) for m, v in d.items()}
            for k, d in micro_values(rows).items()}


def band(xs):
    """(median, min, max) over the rounds. With three rounds a standard
    deviation is noise; the observed range is what there is to report."""
    xs = [float(x) for x in xs if x not in (None, "")]
    if not xs:
        return None, None, None
    return statistics.median(xs), min(xs), max(xs)


def ratio_band(num, den):
    """Envelope of num/den when the rounds are not paired across the two
    systems, so the extremes are taken from opposite ends."""
    nm, nlo, nhi = band(num)
    dm, dlo, dhi = band(den)
    if not nm or not dm:
        return None, None, None
    return nm / dm, nlo / dhi, nhi / dlo


def err(values, lows, highs):
    """Matplotlib wants distances from the point, not absolute bounds."""
    return [[v - lo for v, lo in zip(values, lows)],
            [hi - v for v, hi in zip(values, highs)]]


ERRBAR = dict(ecolor=INK_2, elinewidth=0.7, capsize=2.2, capthick=0.7)


def spread_str(lo, hi, fmt="%.1f"):
    return (fmt + "-" + fmt) % (lo, hi) if lo is not None else "n/a"


# --- system identity -------------------------------------------------------
# One colour, one hatch and one marker per system, fixed here and used by every
# figure, so a reader who has learnt the encoding once keeps it.  MetaTensor is
# the darkest hue and is always the first slot of a bar group.  The hatches and
# markers carry the identity in greyscale and under full CVD.
SYSTEM_IDENTITY = collections.OrderedDict([
    #  system                  colour     hatch  marker
    ("ours",                  ("#1a4fa0", "",    "o")),
    ("torch-compile",         ("#eb6834", "///", "s")),
    ("torch-compile-static",  ("#eb6834", "///", "s")),
    ("torch-compile-ro",      ("#1baf7a", "...", "D")),
    ("torch-compile-dynamic", ("#1baf7a", "...", "D")),
    ("torch-compile-mat",     ("#eda100", "xxx", "^")),
    ("torch-eager",           ("#8a5cd6", "\\\\", "v")),
    ("jax",                   ("#6b6b6b", "++",  "P")),
    ("iree",                  ("#c04a8a", "oo",  "*")),
    ("torch-tensorrt",        ("#4fa8e0", "--",  "X")),
    ("triton",                ("#a0522d", "||",  "h")),
    # not a baseline: the same code with the JIT off, the "before" end of the
    # fusion dumbbell.
    ("nojit",                 ("#9a9994", "",    "o")),
    # The three arms of the DAG-placement experiment.  virtual is our system
    # and keeps its colour; the other two are not baselines, they are the same
    # runtime with the DAG somewhere else.
    ("virtual",               ("#1a4fa0", "",    "o")),
    ("deferred",              ("#d94f70", "\\\\", "s")),
    ("eager",                 ("#9a9994", "",    "^")),
    ("jax-perlayer",          ("#6b6b6b", "++",  "P")),
    ("jax-while",             ("#3f3f3f", "..",  "X")),
])
MARKERS = "osD^vP*Xh"
# Machines are not systems; the compare_* figures colour by machine.
MACHINE_COLORS = ["#1a4fa0", "#eb6834", "#1baf7a"]
MACHINE_HATCH = ["", "///", "..."]


def style_of(system):
    return SYSTEM_IDENTITY.get(system, ("#6b6b6b", "", "o"))


def legend_below(fig, ax=None, ncol=2, handles=None, labels=None, fontsize=7):
    """One legend per figure, outside the axes, under the plot.  A legend
    inside the axes always ends up on top of a bar sooner or later."""
    if handles is None:
        handles, labels = ax.get_legend_handles_labels()
    if not handles:
        return
    # Fewer columns rather than a legend wider than the page: the label
    # lengths differ per figure, so the fit is measured, not guessed.
    for n in range(max(1, ncol), 0, -1):
        leg = fig.legend(handles, labels, loc="outside lower center", ncol=n,
                         frameon=False, columnspacing=1.1, handlelength=1.4,
                         handletextpad=0.5, fontsize=fontsize,
                         borderaxespad=0.2)
        if n == 1:
            return leg
        fig.canvas.draw()
        if leg.get_window_extent().width / fig.dpi <= fig.get_size_inches()[0]:
            return leg
        leg.remove()


def drawn(systems, args):
    """IREE runs one to two orders of magnitude behind XLA on this GPU and
    flattens every axis it shares; the tables keep it, the figures show it
    only with --iree (written with an _iree suffix)."""
    if getattr(args, "iree", False):
        return list(systems)
    return [s for s in systems if s != "iree"]


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
    g = micro_values(read_tsv(os.path.join(out, "micro.tsv")))
    pts = []
    for (dtype, v, k, n), m in g.items():
        if dtype != "float64" or not m.get("fused") or not m.get("torch-compile"):
            continue
        r, lo, hi = ratio_band(m["torch-compile"], m["fused"])
        pts.append((r, micro_label(v, k, n), lo, hi, m))
    if not pts:
        return None
    pts.sort()
    vals = [p[0] for p in pts]
    labels = [p[1] for p in pts]
    lows = [p[2] for p in pts]
    highs = [p[3] for p in pts]
    y = range(len(pts))
    # app rows only exist when the point was also run with --applevel; a
    # point without them just gets no marker.
    app_y, app_x = [], []
    for i, (_, _, _, _, m) in enumerate(pts):
        if m.get("app"):
            am, _, _ = ratio_band(m["torch-compile"], m["app"])
            if am is not None:
                app_y.append(i)
                app_x.append(am)

    fig, ax = plt.subplots(figsize=(args.width, 0.155 * len(pts) + 1.15))
    colors = [POS if v >= 1 else NEG for v in vals]
    # Bars grow from the 1.0 rule, which is the neutral midpoint of the scale.
    ax.barh(list(y), [v - 1 for v in vals], left=1, height=0.55,
            color=colors, linewidth=0,
            hatch=(HATCH[0] if not args.texture else None))
    ax.errorbar(vals, list(y), xerr=err(vals, lows, highs), fmt="none",
                zorder=4, **ERRBAR)
    if app_x:
        # No error bars here - the row is busy enough; the range is in the
        # table.
        # A white face is what makes the marker read on top of a saturated bar;
        # an unfilled one disappears into it.
        ax.scatter(app_x, app_y, s=18, facecolors="white", edgecolors=INK,
                   linewidth=0.9, zorder=6)
    ax.axvline(1, color=INK_2, linewidth=0.8, zorder=3)
    ax.set_xscale("log", base=2)
    ax.set_xticks([0.5, 1, 2, 4, 8, 16])
    ax.xaxis.set_major_formatter(FuncFormatter(
        lambda v, _: ("%g" % v).rstrip("0").rstrip(".") + "\u00d7"))
    ax.set_yticks(list(y), labels)
    ax.set_xlim(min(lows) / 1.5, max(highs) * 2.4)
    ax.set_xlabel("speedup over torch.compile\nlog2 scale; higher is better")
    ax.xaxis.grid(True, zorder=0)
    ax.set_axisbelow(True)
    despine(ax, keep=("left",))
    ax.tick_params(axis="y", length=0)
    # Direct-label the extremes only; the rest are in the table.
    for i in (0, len(pts) - 1):
        v = vals[i]
        # Always to the right of the row, past the whisker: a label placed to
        # the left of a sub-1x bar runs into the benchmark names.
        at = max(highs[i], 1.0)
        ax.annotate("%.2f\u00d7" % v, (at, i), xytext=(4, 0),
                    textcoords="offset points", va="center",
                    ha="left", fontsize=7, color=INK_2)
    legend_handles = [
        Line2D([], [], color=POS, lw=4, label="MetaTensor faster"),
        Line2D([], [], color=NEG, lw=4, label="torch.compile faster")]
    if app_x:
        legend_handles.append(Line2D(
            [], [], marker="o", linestyle="none", markerfacecolor="white",
            markeredgecolor=INK, markersize=5,
            label="app-level through PyPy"))
    legend_below(fig, handles=legend_handles,
                 labels=[h.get_label() for h in legend_handles], ncol=2)
    if args.titles:
        ax.set_title("Microbenchmark speedup, float64")
    rows = []
    header = ["benchmark", "MetaTensor", "range"]
    if app_x:
        header += ["app-level", "range"]
    header += ["torch.compile", "range", "speedup", "range"]
    if app_x:
        header += ["app speedup", "range"]
    for r, lab, lo, hi, m in pts:
        om, olo, ohi = band(m["fused"])
        cm, clo, chi = band(m["torch-compile"])
        row = [lab, "%.1f" % om, spread_str(olo, ohi)]
        if app_x:
            if m.get("app"):
                apm, aplo, aphi = band(m["app"])
                ar, arlo, arhi = ratio_band(m["torch-compile"], m["app"])
                row += ["%.1f" % apm, spread_str(aplo, aphi)]
            else:
                row += ["n/a", "n/a"]
        row += ["%.1f" % cm, spread_str(clo, chi), "%.2f" % r,
                spread_str(lo, hi, "%.2f")]
        if app_x:
            row += (["%.2f" % ar, spread_str(arlo, arhi, "%.2f")]
                    if m.get("app") else ["n/a", "n/a"])
        rows.append(row)
    write_table(os.path.join(args.outdir, "micro_speedup.tex"),
                "median us per iteration over the rounds, with the observed range",
                header, rows)
    return fig, "micro_speedup"


def fig_fusion(out, args):
    """The same benchmark before and after fusion: a dumbbell, one row each."""
    g = micro_values(read_tsv(os.path.join(out, "micro.tsv")))
    pts = []
    for (dtype, v, k, n), m in g.items():
        if dtype != "float64" or not m.get("fused") or not m.get("nojit"):
            continue
        fm, flo, fhi = band(m["fused"])
        sm, slo, shi = band(m["nojit"])
        pts.append((sm / fm, sm, fm, micro_label(v, k, n),
                    (flo, fhi), (slo, shi)))
    if not pts:
        return None
    pts.sort()
    y = range(len(pts))

    fig, ax = plt.subplots(figsize=(args.width, 0.155 * len(pts) + 1.05))
    for i, (_, slow, fast, _lab, _f, _s) in enumerate(pts):
        ax.plot([fast, slow], [i, i], color=GRID, linewidth=1.2, zorder=1,
                solid_capstyle="round")
    slow = [p[1] for p in pts]
    fast = [p[2] for p in pts]
    ax.errorbar(slow, list(y), xerr=err(slow, [p[5][0] for p in pts],
                                       [p[5][1] for p in pts]),
                fmt="none", zorder=2, **ERRBAR)
    ax.errorbar(fast, list(y), xerr=err(fast, [p[4][0] for p in pts],
                                        [p[4][1] for p in pts]),
                fmt="none", zorder=2, **ERRBAR)
    ax.scatter(slow, list(y), s=14, color=style_of("nojit")[0], marker="o",
               zorder=3, linewidth=0, label="interpreted (no JIT)")
    ax.scatter(fast, list(y), s=14, color=style_of("ours")[0], marker="o",
               zorder=3, linewidth=0, label="MetaTensor (fused)")
    ax.set_xscale("log")
    ax.set_yticks(list(y), [p[3] for p in pts])
    ax.set_xlabel("steady-state time per iteration (\u00b5s, log scale)\n"
                  "lower is better")
    ax.xaxis.grid(True, zorder=0)
    ax.set_axisbelow(True)
    despine(ax, keep=("left",))
    ax.tick_params(axis="y", length=0)
    best = pts[-1]
    # To the right of the slow end: to the left of the fast end it lands on
    # the benchmark name.
    ax.annotate("%.1f\u00d7" % best[0], (best[1], len(pts) - 1),
                xytext=(4, 0), textcoords="offset points",
                ha="left", va="center", fontsize=7, color=INK_2)
    legend_below(fig, ax, ncol=2)
    if args.titles:
        ax.set_title("Effect of kernel fusion")
    write_table(os.path.join(args.outdir, "fusion.tex"),
                "median us per iteration over the rounds, with the observed range",
                ["benchmark", "fused", "range", "nojit", "range", "gain"],
                [[p[3], "%.1f" % p[2], spread_str(*p[4]),
                  "%.1f" % p[1], spread_str(*p[5]), "%.2f" % p[0]]
                 for p in reversed(pts)])
    return fig, "fusion"


def fig_fusion_stats(out, args):
    """Mechanism accounting: how big the fusion regions are and what ended
    them.  The bars are the force reasons per model (see fusion_stats.py and
    the README); the table carries the sizes next to them."""
    rows = read_tsv(os.path.join(out, "fusion.tsv"))
    if not rows:
        return None
    reasons = [("forced_library", "library call"), ("forced_item", ".item()"),
               ("forced_loop", "loop/guard exit"),
               ("forced_leafcap", "leaf cap"), ("forced_assign", "assign"),
               ("forced_other", "other")]
    num = lambda r, k: float(r.get(k) or 0)
    fig, ax = plt.subplots(figsize=(args.width, 0.3 * len(rows) + 1.2))
    y = list(range(len(rows)))
    left = [0.0] * len(rows)
    for i, (key, label) in enumerate(reasons):
        vals = [num(r, key) for r in rows]
        if not any(vals):
            continue
        ax.barh(y, vals, left=left, height=0.7, linewidth=0,
                color=SERIES[i % len(SERIES)],
                hatch=HATCH[i % len(HATCH)] if args.texture else None,
                label=label)
        left = [a + b for a, b in zip(left, vals)]
    ax.set_yticks(y, [r["model"] for r in rows])
    ax.set_xlabel("fusion regions per forward,\nby what ended the region")
    ax.xaxis.grid(True, zorder=0)
    ax.set_axisbelow(True)
    despine(ax, keep=("left",))
    ax.tick_params(axis="y", length=0)
    legend_below(fig, ax, ncol=3)
    if args.titles:
        ax.set_title("Why each fusion region ends")
    write_table(os.path.join(args.outdir, "fusion_stats.tex"),
                "per model forward: distinct fused kernels, kernel launches "
                "per iteration, DAG nodes per kernel, extra outputs, and the "
                "force reason of every region in the compiled traces",
                ["model", "kernels", "launches/iter", "nodes min", "median",
                 "max", "extra outs", "library", ".item()", "loop", "leaf cap",
                 "assign", "other"],
                [[r["model"], r["kernels"], r["launches_per_iter"],
                  r["nodes_min"], r["nodes_median"], r["nodes_max"],
                  r["extra_outputs_total"]] +
                 [r[k] for k, _ in reasons] for r in rows])
    return fig, "fusion_stats"


# Out of the population, and so out of every figure, table and mean: two
# fixtures whose weights were never trained, and one mirror of
# weights an individual re-uploaded.  No sweep measures them any more; the
# names stay so that a result set recorded before the rule was applied
# still draws the population and nothing else.  select_models.py records
# which criterion each one failed.
PROBES = ("tiny-gpt2", "bert-tiny", "vit-tiny")

# Admitted although they fall under the usage floor, each as the only cover
# for something the population would otherwise lose: the small end of the
# encoder ladder, and the only attention-free architecture.  The headline
# mean is reported with and without them, so neither can carry a result.
C3_EXEMPT = ("bert-mini", "mixer_b16")


def _model_medians(out):
    """{model: {system: median us}} from models.tsv."""
    rows = read_tsv(os.path.join(out, "models.tsv"))
    g = collections.defaultdict(lambda: collections.defaultdict(list))
    for r in rows:
        try:
            g[r["model"]][r["system"]].append(float(r["steady_us"]))
        except (ValueError, KeyError):
            pass
    return dict((m, dict((s, med(v)) for s, v in d.items()))
                for m, d in g.items())


def _geomean(xs):
    return math.exp(sum(math.log(x) for x in xs) / len(xs)) if xs else None


# The baselines the distribution is drawn against, in the order they are drawn.
# "best-of-three" is the fastest of the three torch.compile modes per model,
# which is the strongest configuration a PyTorch user can reach without
# changing the model.
SPEEDUP_BASES = [("torch-eager", "PyTorch eager"),
                 ("best-of-three", "torch.compile, fastest of three modes"),
                 ("jax", "JAX/XLA"),
                 ("torch-tensorrt", "Torch-TensorRT")]
COMPILE_MODES = ("torch-compile", "torch-compile-ro", "torch-compile-mat")


def _base_us(d, base):
    if base == "best-of-three":
        cands = [d[s] for s in COMPILE_MODES if d.get(s)]
        return min(cands) if cands else None
    return d.get(base)


def fig_model_speedup(out, args):
    """The per-model speedup distribution, the way a compiler paper reports it.

    One sorted curve per baseline over the population, so the spread is
    visible rather than averaged away, with the geometric mean in the legend.
    A tally would hide that the same mean can come from one large win or from
    twelve small ones.
    """
    g = _model_medians(out)
    models = [m for m in g if m not in PROBES]
    if not models:
        return None
    fig, ax = plt.subplots(figsize=(args.width, 2.5))
    table_rows, curves = [], []
    for base, label in SPEEDUP_BASES:
        sp = []
        for m in models:
            ours, b = g[m].get("ours"), _base_us(g[m], base)
            if ours and b:
                sp.append(b / ours)
        if not sp:
            continue
        sp.sort()
        curves.append(sp)
        gm = _geomean(sp)
        color, _, marker = style_of("torch-compile" if base == "best-of-three" else base)
        ax.plot(range(1, len(sp) + 1), sp, marker=marker, markersize=3.5,
                linewidth=1.2, color=color,
                label="%s (gmean %.2f$\\times$)" % (label, gm))
        qual = [b / g[m]["ours"] for m in models
                if m not in C3_EXEMPT and g[m].get("ours")
                for b in [_base_us(g[m], base)] if b]
        table_rows.append([label, "%d" % len(sp), "%.2f" % gm,
                           "%.2f" % _geomean(qual) if qual else "n/a",
                           "%.2f" % sp[0], "%.2f" % sp[-1],
                           "%d" % sum(1 for x in sp if x > 1.0)])
    # Everything under the line is a loss; shading it means a reader does not
    # have to read the axis to see how much of each curve is below one.
    ax.axhspan(1e-3, 1.0, color="0.92", zorder=0)
    ax.axhline(1.0, color="0.35", linewidth=0.8, linestyle="--", zorder=1)
    ax.set_yscale("log")
    ax.set_xlabel("model configurations, sorted by speedup")
    ax.set_ylabel("speedup over baseline\n(>1 favours MetaTensor)")
    ax.set_xlim(0.5, max(2, len(models)) + 0.5)
    # A speedup axis reads as 0.8x and 2x, never as 8 times ten to the
    # minus one, so the decade ticks a log scale defaults to are replaced.
    lo = min(min(c) for c in curves) if curves else 0.5
    hi = max(max(c) for c in curves) if curves else 2.0
    ticks = [t for t in (0.25, 0.4, 0.5, 0.6, 0.8, 1, 1.5, 2, 3, 4, 6, 8)
             if lo / 1.15 <= t <= hi * 1.15]
    ax.set_yticks(ticks)
    ax.set_yticklabels(["%g$\\times$" % t for t in ticks])
    ax.yaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    ax.set_ylim(lo / 1.15, hi * 1.15)
    ax.yaxis.grid(True, zorder=0)
    ax.set_axisbelow(True)
    despine(ax)
    legend_below(fig, ax, ncol=1 if args.width < DOUBLE_COL else 2)
    if args.titles:
        ax.set_title("End-to-end speedup distribution")
    write_table(os.path.join(args.outdir, "model_speedup.tex"),
                "speedup of MetaTensor over each baseline across the "
                "population, medians of three rounds",
                ["baseline", "models", "geomean", "geomean, C3 only",
                 "min", "max", "faster on"],
                table_rows)
    return fig, "model_speedup"


def fig_models(out, args):
    """Three systems per model, all in the same unit: grouped bars from zero."""
    rows = read_tsv(os.path.join(out, "models.tsv"))
    if not rows:
        return None
    g = collections.defaultdict(lambda: collections.defaultdict(list))
    for r in rows:
        g[r["model"]][r["system"]].append(float(r["steady_us"]))
    systems = [s for s in ["ours", "torch-compile", "torch-compile-ro", "torch-compile-mat",
                           "torch-eager", "jax", "iree", "torch-tensorrt"]
               if any(s in d for d in g.values())]
    models = sorted((m for m in g if m not in PROBES),
                    key=lambda m: med(g[m].get("ours", [])) or 0)
    stats = {s: [band(g[m].get(s, [])) for m in models] for s in systems}
    vals = {s: [b[0] or 0 for b in stats[s]] for s in systems}

    shown = drawn(systems, args)
    # 3 systems keeps the original 0.26 half-height; more systems shrink to fit.
    h = 0.8 / len(shown)
    fig, ax = plt.subplots(figsize=(args.width, 0.42 * len(models) + 1.6))
    y = [i for i in range(len(models))]
    for si_, s in enumerate(shown):
        # MetaTensor first means topmost in every horizontal bar group.
        off = ((len(shown) - 1) / 2 - si_) * h
        color, hatch, _ = style_of(s)
        ax.barh([v + off for v in y], vals[s], height=h * 0.88,
                color=color, linewidth=0,
                hatch=hatch if args.texture else None,
                label=SYSTEM_LABEL[s])
        ax.errorbar(vals[s], [v + off for v in y],
                    xerr=err(vals[s], [b[1] or 0 for b in stats[s]],
                             [b[2] or 0 for b in stats[s]]),
                    fmt="none", zorder=4, **ERRBAR)
    ax.set_yticks(y, models)
    ax.set_xscale("log")
    ax.set_xlim(left=50)
    ax.set_xlabel("steady-state time per iteration (\u00b5s, log scale)\n"
                  "lower is better")
    ax.xaxis.grid(True, zorder=0)
    ax.set_axisbelow(True)
    despine(ax, keep=("left",))
    ax.tick_params(axis="y", length=0)
    legend_below(fig, ax, ncol=3 if args.width >= DOUBLE_COL else 2)
    if args.titles:
        ax.set_title("End-to-end inference")

    ov, ov_st = vals.get("ours", [0] * len(models)), stats.get("ours", [(None, None, None)] * len(models))
    tc, tc_st = vals.get("torch-compile", [0] * len(models)), stats.get("torch-compile", [(None, None, None)] * len(models))
    tcro, tcro_st = vals.get("torch-compile-ro", [0] * len(models)), stats.get("torch-compile-ro", [(None, None, None)] * len(models))
    tcmat, tcmat_st = vals.get("torch-compile-mat", [0] * len(models)), stats.get("torch-compile-mat", [(None, None, None)] * len(models))
    te = vals.get("torch-eager", [0] * len(models))
    jx = vals.get("jax", [0] * len(models))
    ir = vals.get("iree", [0] * len(models))
    trt = vals.get("torch-tensorrt", [0] * len(models))
    header = ["model", "MetaTensor", "range", "torch.compile", "range",
              "compile-ro", "range", "compile-mat", "range", "eager"]
    if "jax" in systems:
        header.append("JAX/XLA")
    if "iree" in systems:
        header.append("IREE")
    header.append("TensorRT")
    header.append("ratio")
    if "jax" in systems:
        header.append("jax/compile")
    if "torch-tensorrt" in systems:
        header.append("trt/compile")
    table_rows = []
    for i, m in enumerate(models):
        row = [m, "%.0f" % ov[i], spread_str(ov_st[i][1], ov_st[i][2], "%.0f"),
               "%.0f" % tc[i], spread_str(tc_st[i][1], tc_st[i][2], "%.0f"),
               "%.0f" % tcro[i] if tcro[i] else "n/a",
               spread_str(tcro_st[i][1], tcro_st[i][2], "%.0f"),
               "%.0f" % tcmat[i] if tcmat[i] else "n/a",
               spread_str(tcmat_st[i][1], tcmat_st[i][2], "%.0f"),
               "%.0f" % te[i]]
        if "jax" in systems:
            row.append("%.0f" % jx[i] if jx[i] else "n/a")
        if "iree" in systems:
            row.append("%.0f" % ir[i] if ir[i] else "n/a")
        row.append("%.0f" % trt[i] if trt[i] else "n/a")
        row.append("%.2f" % (ov[i] / tc[i]) if tc[i] else "n/a")
        if "jax" in systems:
            row.append("%.2f" % (jx[i] / tc[i]) if tc[i] and jx[i] else "n/a")
        if "torch-tensorrt" in systems:
            row.append("%.2f" % (trt[i] / tc[i]) if tc[i] and trt[i] else "n/a")
        table_rows.append(row)
    write_table(os.path.join(args.outdir, "models.tex"),
                "median us per iteration over the rounds, with the observed range",
                header, table_rows)
    return fig, "models"


PHASE_ORDER = {"first": 0, "revisit": 1, "new": 2, "1": 0, "2": 1}
PHASE_LABEL = {"first": "first visit", "revisit": "revisit",
               "new": "new length distribution", "1": "first visit",
               "2": "revisit"}


def fig_dynamic(out, args):
    """One panel per phase, because pooling them would put a cold visit and a
    warm one on the same line.  The line is the steady-state median per
    length; the compile events the sweep exists to expose are counters, not
    time, and live in the tables beside it."""
    rows = read_tsv(os.path.join(out, "dynamic_summary.tsv"))
    if not rows:
        return None
    systems = [s for s in ["ours", "torch-compile-static",
                           "torch-compile-dynamic", "torch-eager"]
               if any(r["system"] == s for r in rows)]
    phases = sorted({r.get("pass", "1") for r in rows},
                    key=lambda p: PHASE_ORDER.get(p, 9))
    per = collections.defaultdict(list)
    for r in rows:
        per[(r["system"], r.get("pass", "1"), int(r["length"]))].append(r)
    lengths = sorted({int(r["length"]) for r in rows})
    per_phase_lengths = {p: sorted({int(r["length"]) for r in rows
                                    if r.get("pass", "1") == p})
                         for p in phases}

    # At one column three panels side by side leave an inch each, which is not
    # a plot.  They stack, and the whole figure keeps to the column.
    if args.width >= DOUBLE_COL:
        fig, axes = plt.subplots(1, len(phases), squeeze=False, sharey=True,
                                 figsize=(args.width, args.width * 0.46))
        axes = list(axes[0])
    else:
        # Not sharex: each phase sweeps its own lengths, and a shared axis
        # would stretch every phase over the union of all three and leave the
        # tick values on the bottom panel only.
        fig, axes = plt.subplots(len(phases), 1, squeeze=False, sharey=True,
                                 figsize=(args.width,
                                          1.25 * len(phases) + 1.0))
        axes = [a[0] for a in axes]
    top = 0.0
    for ax, phase in zip(axes, phases):
        xs = per_phase_lengths[phase]
        for s in systems:
            bands = [band([float(r["median_us"]) for r in per.get((s, phase, L), [])])
                     for L in xs]
            if not any(b[0] for b in bands):
                continue
            ys = [b[0] or 0.0 for b in bands]
            top = max(top, max(b[2] or 0.0 for b in bands))
            color, _, marker = style_of(s)
            ax.errorbar(xs, ys, yerr=err(ys, [b[1] or 0.0 for b in bands],
                                         [b[2] or 0.0 for b in bands]),
                        fmt="none", zorder=2, **ERRBAR)
            ax.plot(xs, ys, color=color, linewidth=1.3, marker=marker,
                    markersize=3.0, markeredgewidth=0,
                    label=SYSTEM_LABEL[s], zorder=3)
        ax.set_xticks(xs)
        ax.set_title(PHASE_LABEL.get(phase, phase), fontsize=7)
        ax.yaxis.grid(True, zorder=0)
        ax.set_axisbelow(True)
        despine(ax)
    # One x label per figure when the panels are stacked, one per panel when
    # they are not: a shared axis carries its label once.
    if args.width >= DOUBLE_COL:
        for ax in axes:
            ax.set_xlabel("sequence length (tokens)")
    else:
        axes[-1].set_xlabel("sequence length (tokens)")
    axes[0].set_ylabel("median time per step (\u00b5s)\nlower is better")
    axes[0].set_ylim(0, top * 1.08)
    legend_below(fig, axes[0], ncol=2)
    if args.titles:
        fig.suptitle("Changing sequence length")

    DELTAS = ["loops", "bridges", "kernels", "cache_hits", "new_graphs"]

    def delta(key, col):
        vs = [float(r.get(col) or 0) for r in per.get(key, [])]
        return med(vs) if vs else 0.0

    time_rows = []
    for s in systems:
        for phase in phases:
            row = [SYSTEM_LABEL[s], PHASE_LABEL.get(phase, phase)]
            for L in lengths:
                vs = [float(r["median_us"]) for r in per.get((s, phase, L), [])]
                row.append("%.0f [%s]" % (band(vs)[0],
                                          spread_str(*band(vs)[1:], fmt="%.0f"))
                           if vs else "-")
            row += ["%.0f" % sum(delta((s, phase, L), c) for L in lengths)
                    for c in DELTAS]
            time_rows.append(row)
    write_table(os.path.join(args.outdir, "dynamic.tex"),
                "median us per step over the rounds (range in brackets), and "
                "the compile counters summed over the phase. first visit is "
                "the first time each length is seen, revisit is the same "
                "lengths again, and the last phase is five lengths the run "
                "has not seen",
                ["system", "phase"] + [str(L) for L in lengths]
                + ["loops", "bridges", "kernels", "hits", "new graphs"],
                time_rows)
    write_table(os.path.join(args.outdir, "dynamic_deltas.tex"),
                "per-length compile counters: the delta inside that length's "
                "window, median over the rounds. new graphs counts graphs "
                "compiled in the window, which on a first visit are first "
                "compilations and on a revisit are recompilations",
                ["system", "phase", "length", "loops", "bridges", "kernels",
                 "cache hits", "new graphs"],
                [[SYSTEM_LABEL[s], PHASE_LABEL.get(p, p), str(L)]
                 + ["%.0f" % delta((s, p, L), c) for c in DELTAS]
                 for s in systems for p in phases
                 for L in per_phase_lengths[p]])
    return fig, "dynamic"


def fig_precision(out, args):
    """Two panels with unrelated y-scales would invite a misread, so the panels
    are collapsed into one axis of speedup: a ratio is unitless and shares a
    scale no matter how far apart the absolute times are."""
    g = micro_values(read_tsv(os.path.join(out, "micro.tsv")))
    dtypes = ["float64", "float32", "float16"]
    variants = sorted({k[1] for k in g if k[0] != "float64"})
    if not variants:
        return None
    fig, ax = plt.subplots(figsize=(args.width, args.width * 0.85))
    x = range(len(dtypes))
    w = 0.8 / max(len(variants), 1)
    table = []
    for vi, v in enumerate(variants):
        n = max(k[3] for k in g if k[1] == v)
        ratios, lows, highs, absolute = [], [], [], []
        for d in dtypes:
            m = g.get((d, v, 1, n)) or {}
            r, lo, hi = ratio_band(m.get("torch-compile", []), m.get("fused", []))
            ratios.append(r or 0)
            lows.append(lo or 0)
            highs.append(hi or 0)
            absolute.append((band(m.get("fused", [])), band(m.get("torch-compile", []))))
        off = (vi - (len(variants) - 1) / 2) * w
        ax.bar([i + off for i in x], ratios, w * 0.86, color=SERIES[vi],
               linewidth=0, hatch=HATCH[vi] if args.texture else None,
               label="%s (n=%s)" % (VARIANT_LABEL.get(v, "v%d" % v), si(n)))
        ax.errorbar([i + off for i in x], ratios,
                    yerr=err(ratios, lows, highs), fmt="none", zorder=4,
                    **ERRBAR)
        for d, r, lo, hi, (ob, cb) in zip(dtypes, ratios, lows, highs, absolute):
            table.append([VARIANT_LABEL.get(v, "v%d" % v), d,
                          "%.1f" % ob[0] if ob[0] else "n/a",
                          "%.1f" % cb[0] if cb[0] else "n/a",
                          "%.2f" % r if r else "n/a",
                          spread_str(lo, hi, "%.2f") if r else "n/a"])
    ax.axhline(1, color=INK_2, linewidth=0.8, zorder=3)
    ax.set_xticks(list(x), ["float64", "float32", "float16"])
    ax.set_xlabel("element type")
    ax.set_ylabel("speedup over torch.compile (×)\nhigher is better")
    ax.yaxis.set_major_formatter(FuncFormatter(
        lambda v, _: ("%g" % v) + "\u00d7"))
    ax.yaxis.grid(True, zorder=0)
    ax.set_axisbelow(True)
    despine(ax)
    legend_below(fig, ax, ncol=3)
    if args.titles:
        ax.set_title("Precision sweep")
    write_table(os.path.join(args.outdir, "precision.tex"),
                "median us per iteration over the rounds, with the speedup range",
                ["variant", "dtype", "MetaTensor", "torch.compile", "speedup",
                 "range"], table)
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
    gl = collections.defaultdict(list)
    for r in rows:
        key = (r["experiment"], r["model"], r["variant"])
        if r.get("steady_us"):
            g[key].append(float(r["steady_us"]))
        if r.get("launches_per_iter"):
            gl[key].append(float(r["launches_per_iter"]))
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
        groups.setdefault((exp, model), []).append(
            (var,) + band(g[(exp, model, var)]))

    keys = list(groups)
    fig, ax = plt.subplots(figsize=(args.width, 0.52 * len(keys) + 0.8))
    height = 0.30
    ys, xs, lows, highs, texts, ticks = [], [], [], [], [], []
    for i, key in enumerate(keys):
        items = groups[key]
        for slot, (var, value, lo, hi) in enumerate(items):
            ys.append(i + ((len(items) - 1) / 2 - slot) * height)
            xs.append(value)
            lows.append(lo)
            highs.append(hi)
            texts.append(var)
        ticks.append("%s / %s" % (key[0].replace("_", " "), key[1]))
    ax.barh(ys, xs, height=height * 0.88, color=SERIES[0], linewidth=0,
            hatch=HATCH[0] if args.texture else None)
    ax.errorbar(xs, ys, xerr=err(xs, lows, highs), fmt="none", zorder=4,
                **ERRBAR)
    for xv, yv, t in zip(xs, ys, texts):
        ax.annotate(t, (xv, yv), xytext=(3, 0), textcoords="offset points",
                    va="center", fontsize=6.5, color=INK_2)
    ax.set_yticks(range(len(keys)), ticks)
    ax.set_xlabel("steady-state time per iteration (\u00b5s)\nlower is better")
    ax.set_xlim(0, max(highs) * 1.22)
    ax.xaxis.grid(True, zorder=0)
    ax.set_axisbelow(True)
    despine(ax, keep=("left",))
    ax.tick_params(axis="y", length=0)
    if args.titles:
        ax.set_title("Ablations")
    write_table(os.path.join(args.outdir, "ablation.tex"),
                "median us per iteration over the rounds, with the observed "
                "range, and launches/iter where the experiment records it",
                ["experiment", "model", "setting", "us", "range", "launches/iter"],
                [[e.replace("_", " "), m, v, "%.0f" % band(g[(e, m, v)])[0],
                  spread_str(*band(g[(e, m, v)])[1:], fmt="%.0f"),
                  "%.1f" % statistics.median(gl[(e, m, v)]) if gl.get((e, m, v)) else ""]
                 for (e, m, v) in sorted(g, key=lambda k: (k[0], k[1], setting_key(k[2])))])
    return fig, "ablation"


def fig_explain(out, args):
    """Table only: Dynamo graph count/break count/op count per model, one
    row each, no rounds to spread over."""
    rows = read_tsv(os.path.join(out, "explain.tsv"))
    if not rows:
        return None
    write_table(os.path.join(args.outdir, "explain.tex"),
                "torch._dynamo.explain graph structure per model",
                ["model", "graphs", "breaks", "ops"],
                [[r["model"], r.get("graphs", ""), r.get("breaks", ""),
                  r.get("ops", "")] for r in rows])
    fig, ax = plt.subplots(figsize=(args.width, 0.3))
    ax.axis("off")
    return fig, "explain"


def append_notes(path, paragraphs):
    """Add a notes block under a table written by write_table."""
    with open(path, "a") as f:
        f.write("\n%% notes\n\\begin{minipage}{\\linewidth}\\footnotesize\n")
        for para in paragraphs:
            f.write("\\par\\smallskip\n" + para + "\n")
        f.write("\\end{minipage}\n")


def fig_op_inventory(out, args):
    """Table only: every operator a forward runs, both systems in one
    vocabulary.

    The call-count table next to it only covers the GEMM-shaped work, which is
    where the time is but not where the difference in *what is executed* is.
    This one names every operator, so the reader can see that the two sides run
    the same architecture down to the activation, and read off exactly which
    cells do not line up and why."""
    rows = read_tsv(os.path.join(out, "op_inventory.tsv"))
    if not rows:
        return None
    words = [c for c in rows[0] if c not in ("model", "system", "notes")]
    # Drop a column no model uses, so the table fits the page.
    words = [w for w in words if any(int(r[w]) for r in rows)]
    short = {"layernorm": "LN", "rmsnorm": "RMS", "softmax": "smax",
             "residual_add": "res", "embedding_gather": "gather",
             "reshape_transpose": "shape", "reduce": "red", "matmul": "mm",
             "rotary": "rope"}
    table = [[r["model"], SYSTEM_LABEL.get(r["system"], r["system"])] +
             [r[w] for w in words] for r in rows]
    path = os.path.join(args.outdir, "op_inventory.tex")
    write_table(path,
                "operators per forward, both systems in one vocabulary",
                ["model", "system"] + [short.get(w, w) for w in words], table,
                align="ll" + "r" * len(words))
    append_notes(path, op_inventory_notes(out))
    fig, ax = plt.subplots(figsize=(args.width, 0.3))
    ax.axis("off")
    return fig, "op_inventory"


def op_inventory_notes(out):
    """The mapping table and the per-model reasons, from the tsv's companion
    notes file, turned into LaTeX paragraphs."""
    path = os.path.join(out, "op_inventory_notes.txt")
    if not os.path.exists(path):
        return []
    tex = lambda s: (str(s).replace("\\", "").replace("_", r"\_")
                     .replace("%", r"\%").replace("&", r"\&")
                     .replace("#", r"\#"))
    maps, alias, notes, diffs = [], [], [], []
    for line in open(path):
        parts = line.rstrip("\n").split("\t")
        if parts[0] == "map":
            maps.append(r"\texttt{%s}$\rightarrow$%s" % (tex(parts[1]),
                                                         tex(parts[2])))
        elif parts[0] == "alias":
            alias.append(r"\texttt{%s}" % tex(parts[1]))
        elif parts[0] == "note":
            notes.append(tex(parts[2]))
        elif parts[0] == "diff":
            diffs.append((parts[2], parts[1]))
    paras = []
    if maps:
        paras.append(r"\textbf{Mapping.} torch's rows are one eager forward "
                     r"under \texttt{torch.profiler}, with each aten name "
                     r"mapped to a vocabulary word: " + ", ".join(maps) + ".")
    if alias:
        paras.append(r"\textbf{Not counted.} These are the same call seen at "
                     r"an outer or inner dispatch level as one already "
                     r"counted, so counting them too would double-count: " +
                     ", ".join(alias) + ".")
    paras += [r"\textbf{Note.} " + n for n in notes]
    if diffs:
        # Grouped by reason, not by model: the same cause explains the same
        # cell in six models, and listing it six times is unreadable.
        by = collections.OrderedDict()
        for why, cell in diffs:
            by.setdefault(why, []).append(cell)
        paras.append(r"\textbf{Cells that differ.} Every cell where the two "
                     r"rows disagree, grouped by the one reason it does.")
        for why, cells in by.items():
            paras.append(r"\emph{%s} %s." % (tex(why), "; ".join(
                tex(c) for c in cells)))
    return paras


def _role(model):
    """Where a row stands under the selection rule: population, in it
    under the usage exemption, or out of it."""
    if model in PROBES:
        return "excluded"
    if model in C3_EXEMPT:
        return "population, C3 exemption"
    return "population"


WEIGHT_LABEL = {"trained": "trained",
                "head-random": "encoder trained, head random",
                "random": "randomly initialised fixture"}


def fig_model_provenance(out, args):
    """Table only: which checkpoint each row is, and whether it was trained.

    A number cannot be checked against the wrong checkpoint, and a model whose
    weights were never trained is a property to declare rather than to leave
    for a reader to discover.  One row per configuration: the HuggingFace id,
    the snapshot the export read, the parameter count and the weight
    provenance."""
    rows = [r for r in read_tsv(os.path.join(out, "model_inventory.tsv"))
            if r.get("system") == "ours" and r.get("hf_id")]
    if not rows:
        return None
    table = []
    for r in rows:
        repo, _, rev = r["revision"].rpartition("@")
        table.append([r["model"], r["hf_id"], rev,
                      si(int(r["params_only"])),
                      WEIGHT_LABEL.get(r["weights"], r["weights"]),
                      _role(r["model"])])
    path = os.path.join(args.outdir, "model_provenance.tex")
    write_table(path,
                "checkpoint, snapshot and weight provenance per configuration",
                ["model", "HuggingFace id", "snapshot", "params", "weights",
                 "role"], table, align="lllrll")
    fig, ax = plt.subplots(figsize=(args.width, 0.3))
    ax.axis("off")
    return fig, "model_provenance"


def fig_model_inventory(out, args):
    """Table only: what each model computes, ours against torch.

    The models figure is a timing comparison, and a reviewer's first question
    about one is whether the two systems ran the same workload.  The logits
    check answers that from the output; this answers it from the structure -
    parameters, multiply-add work, and the cuBLAS/cuDNN calls a forward costs
    - with every remaining difference named in the notes column."""
    rows = read_tsv(os.path.join(out, "model_inventory.tsv"))
    if not rows:
        return None
    table = [[r["model"], SYSTEM_LABEL.get(r["system"], r["system"]),
              si(int(r["params_only"])),
              ("+" + si(int(r["params_plus_buffers"]) -
                        int(r["params_only"])))
              if int(r["params_plus_buffers"]) > int(r["params_only"])
              else "--",
              r["gflops"], r["gemm_calls"],
              r["bmm_calls"], r["conv_calls"], r["sdpa_calls"],
              r["notes"]] for r in rows]
    path = os.path.join(args.outdir, "model_inventory.tex")
    write_table(path,
                "per forward: parameters, GEMM/bmm/conv FLOPs and the "
                "cuBLAS/cuDNN calls that carry them, with the remaining "
                "differences named",
                ["model", "system", "params", "buffers", "GFLOP", "GEMM",
                 "bmm", "conv", "SDPA", "notes"], table,
                align="ll" + "r" * 7 + "p{0.30\\textwidth}")
    append_notes(path, [
        r"\emph{params} counts the learned weights; \emph{buffers} is what "
        r"the registered non-learned tensors that are still part of the "
        r"trained model add on top, which in these nine models is the "
        r"batch-norm running mean and variance of ResNet-18 (9600 values) "
        r"and nothing else. Both sides are counted the same way and agree "
        r"exactly on both columns for ResNet-18.",
        r"Excluded from both columns on both sides, as inputs or as tables "
        r"rebuilt from the config rather than anything training produced: "
        r"ours \texttt{image}, \texttt{rope.cos}, \texttt{rope.sin}, "
        r"\texttt{rope.p}; torch \texttt{position\_ids}, "
        r"\texttt{token\_type\_ids}, \texttt{inv\_freq}, "
        r"\texttt{original\_inv\_freq}, \texttt{num\_batches\_tracked} "
        r"(a counter, not data), and a module's causal-mask buffer where it "
        r"keeps one. The input token ids live in the config, not in the "
        r"checkpoint index, so they are outside our count by construction.",
    ])
    fig, ax = plt.subplots(figsize=(args.width, 0.3))
    ax.axis("off")
    return fig, "model_inventory"


DEOPT_PATTERNS = ["never", "alternate", "both-hot", "fresh", "probe-a",
                  "probe-e"]
DEOPT_LABEL = {
    "never": "guard holds",
    "alternate": "guard fails every iter",
    "both-hot": "both paths hot",
    "fresh": "fresh value every iter",
    "probe-a": "probe (a) shared branch",
    "probe-e": "probe (e) shape change",
}


TRANSITION_ARM = {"direct": ("#1a4fa0", "", "o"),
                  "drain": ("#8a5cd6", "\\\\", "v")}
TRANSITION_LABEL = {"direct": "descriptor kept",
                    "drain": "drained to canonical, re-recorded"}
# The three ways the interior consumer can be present, in the order that makes
# the story readable: never, always, and arriving after a guard.
READ_ORDER = ["none", "every", "half"]
READ_LABEL = {"none": "no interior\nconsumer",
              "every": "consumer from\nthe first step",
              "half": "consumer after\na guard fails"}


def fig_transition(out, args):
    """What a fused value costs to carry across a guard.

    Two panels, because the two halves of the result are of different kinds.
    Above: launches per iteration, which is structure and is identical across
    rounds and cache states - it shows that reusing a launched region is what
    keeps the chain one kernel at all, and that the advantage does not survive
    the guard.  Below: cumulative time in each cache state, which is where the
    extra kernel's compile shows up and then stops showing up.
    """
    phases = read_tsv(os.path.join(out, "transition_phases.tsv"))
    rows = read_tsv(os.path.join(out, "transition.tsv"))
    if not phases and not rows:
        return None
    fig, axes = plt.subplots(2, 1, figsize=(args.width, 4.1),
                            gridspec_kw={"height_ratios": [1.0, 0.85]})

    ax = axes[0]
    per = {(r["arm"], r["interior_reads"]): float(r["launches_per_iter"])
           for r in phases}
    reads = [k for k in READ_ORDER if any((a, k) in per
                                          for a in TRANSITION_ARM)]
    x = list(range(len(reads)))
    width = 0.38
    for i, arm in enumerate(("direct", "drain")):
        vals = [per.get((arm, k), 0.0) for k in reads]
        colour, hatch, _ = TRANSITION_ARM[arm]
        ax.bar([v + (i - 0.5) * width for v in x], vals, width * 0.9,
               color=colour, linewidth=0,
               hatch=hatch if args.texture else None,
               label=TRANSITION_LABEL[arm])
        for xi, v in zip(x, vals):
            if v:
                ax.text(xi + (i - 0.5) * width, v + 0.12, "%.2f" % v,
                        ha="center", fontsize=6)
    ax.set_xticks(x, [READ_LABEL.get(k, k) for k in reads])
    ax.set_ylabel("kernel launches\nper iteration (lower is better)")
    ax.set_ylim(0, max([v for v in per.values()] or [1]) * 1.22)
    ax.yaxis.grid(True, zorder=0)
    ax.set_axisbelow(True)
    despine(ax)

    ax = axes[1]
    g = collections.defaultdict(list)
    for r in rows:
        try:
            g[(r["cache"], r["arm"])].append(float(r["total_ms"]))
        except (KeyError, ValueError):
            pass
    caches = [c for c in ("cold", "warm") if any(k[0] == c for k in g)]
    x = list(range(len(caches)))
    for i, arm in enumerate(("direct", "drain")):
        vals = [med(g.get((c, arm), [])) or 0.0 for c in caches]
        colour, hatch, _ = TRANSITION_ARM[arm]
        ax.bar([v + (i - 0.5) * width for v in x], vals, width * 0.9,
               color=colour, linewidth=0,
               hatch=hatch if args.texture else None)
        lo = [min(g.get((c, arm), [0])) for c in caches]
        hi = [max(g.get((c, arm), [0])) for c in caches]
        ax.errorbar([v + (i - 0.5) * width for v in x], vals,
                    yerr=err(vals, lo, hi), fmt="none", zorder=4, **ERRBAR)
    ax.set_xticks(x, ["cold kernel cache", "warm kernel cache"][:len(caches)])
    ax.set_yscale("log")
    ax.set_ylabel("cumulative ms over the run\n(log scale, lower is better)")
    ax.yaxis.grid(True, zorder=0)
    ax.set_axisbelow(True)
    despine(ax)

    legend_below(fig, axes[0], ncol=1 if args.width < DOUBLE_COL else 2)
    if args.titles:
        fig.suptitle("Carrying a fused value across a guard")

    table = []
    for k in reads:
        row = [READ_LABEL.get(k, k).replace("\n", " ")]
        for arm in ("direct", "drain"):
            row.append("%.2f" % per[(arm, k)] if (arm, k) in per else "n/a")
        if ("direct", k) in per and ("drain", k) in per:
            row.append("%.2f" % (per[("drain", k)] / per[("direct", k)]))
        else:
            row.append("n/a")
        table.append(row)
    write_table(os.path.join(args.outdir, "transition.tex"),
                "kernel launches per iteration, 400 iterations, identical "
                "across rounds and cache states",
                ["interior consumer", "descriptor kept", "drained",
                 "drained / kept"], table)
    return fig, "transition"


MOTION_ARMS = ["derived", "handwritten", "direct", "drain"]
MOTION_LABEL = {"derived": "derived (rule specialised in place)",
                "handwritten": "canonicalised first",
                "direct": "hand-written adapter, descriptor kept",
                "drain": "hand-written adapter, drained"}
MOTION_CASE = {"axis": "vector axis", "layout": "blocked layout"}


def _motion_rows(out):
    rows = read_tsv(os.path.join(out, "motion.tsv"))
    for r in rows:
        r["cache"] = r.get("cache") or "cold"
    return rows


def _sign_interval(xs):
    s = sorted(xs)
    n = len(s)
    if n < 6:
        return (s[0], s[-1]) if s else (0.0, 0.0)
    k = 2 if n >= 10 else 1
    return s[k - 1], s[n - k]


def fig_motion(out, args):
    """The MOTION demonstration, one change class per group.  Above: kernel
    launches per step after the transition, which is the evidence that the
    rule was specialised into the region (it does not depend on a clock).
    Below: the step latency those launches cost, medians with sign-test
    intervals over fresh processes."""
    rows = [r for r in _motion_rows(out) if r["cache"] == "cold"]
    if not rows:
        return None
    cases = [c for c in MOTION_CASE if any(r["case"] == c for r in rows)]
    arms = [a for a in MOTION_ARMS if any(r["arm"] == a for r in rows)]
    g = collections.defaultdict(list)
    for r in rows:
        g[(r["case"], r["arm"])].append(r)
    fig, axes = plt.subplots(2, 1, figsize=(args.width, 4.3), sharex=True)
    width = 0.8 / max(len(arms), 1)
    x = list(range(len(cases)))
    table = []
    for panel, key, ylabel in ((0, "launches_after",
                                "kernel launches per step\nafter the change"),
                               (1, "step_us", "step latency (us)")):
        ax = axes[panel]
        for i, arm in enumerate(arms):
            vals, lo, hi = [], [], []
            for c in cases:
                xs = [float(r[key]) for r in g.get((c, arm), [])]
                m = med(xs) if xs else 0.0
                a, b = _sign_interval(xs) if xs else (0.0, 0.0)
                vals.append(m)
                lo.append(a)
                hi.append(b)
            pos = [v + (i - (len(arms) - 1) / 2.0) * width for v in x]
            ax.bar(pos, vals, width * 0.9, color=SERIES[i], linewidth=0,
                   hatch=HATCH[i] if args.texture else None,
                   label=MOTION_LABEL[arm])
            if panel == 1:
                ax.errorbar(pos, vals, yerr=err(vals, lo, hi), fmt="none",
                            zorder=4, **ERRBAR)
            for p_, v in zip(pos, vals):
                ax.text(p_, v * 1.02 + 0.02, ("%.2f" if panel == 0 else
                                              "%.0f") % v,
                        ha="center", va="bottom", fontsize=5.5)
        ax.set_ylabel(ylabel)
        ax.yaxis.grid(True, zorder=0)
        ax.set_axisbelow(True)
        ax.set_ylim(0, ax.get_ylim()[1] * 1.12)
        despine(ax)
    axes[1].set_xticks(x, [MOTION_CASE[c] for c in cases])
    legend_below(fig, axes[1], ncol=1 if args.width < DOUBLE_COL else 2)
    if args.titles:
        fig.suptitle("A change rule derived, against hand-written recovery")
    for c in cases:
        for arm in arms:
            rs = g.get((c, arm), [])
            if not rs:
                continue
            st = [float(r["step_us"]) for r in rs]
            at = [float(r["at_us"]) for r in rs]
            a, b = _sign_interval(st)
            table.append([MOTION_CASE[c], MOTION_LABEL[arm],
                          "%.2f" % med([float(r["launches_after"]) for r in rs]),
                          "%.1f [%.1f, %.1f]" % (med(st), a, b),
                          "%.0f" % med(at),
                          "%.0f" % med([float(r["retained_bytes"]) for r in rs]),
                          str(len(rs))])
    write_table(os.path.join(args.outdir, "motion.tex"),
                "the MOTION demonstration: medians over fresh processes, "
                "sign-test intervals (97.9%% at n=10), cold kernel cache",
                ["change class", "arm", "launches/step", "step us",
                 "transition us", "retained bytes", "n"], table)
    return fig, "motion"


def fig_motion_retention(out, args):
    """Device bytes still held at the end of a run, against the run's length.
    The comparison changes direction with length, so it is drawn as a curve
    rather than quoted at one length."""
    rows = [r for r in _motion_rows(out) if r["cache"] == "warm"]
    if not rows:
        return None
    cases = [c for c in MOTION_CASE if any(r["case"] == c for r in rows)]
    fig, axes = plt.subplots(len(cases), 1,
                             figsize=(args.width, 2.1 * len(cases) + 0.5),
                             squeeze=False)
    for ci, c in enumerate(cases):
        ax = axes[ci][0]
        for i, arm in enumerate(MOTION_ARMS):
            pts = collections.defaultdict(list)
            for r in rows:
                if r["case"] == c and r["arm"] == arm:
                    pts[int(r["steps"])].append(float(r["retained_bytes"]))
            if not pts:
                continue
            ls = sorted(pts)
            ax.plot(ls, [med(pts[L]) / 1e6 for L in ls], marker=MARKERS[i],
                    color=SERIES[i], markersize=3.5, linewidth=1.2,
                    label=MOTION_LABEL[arm] if ci == 0 else None)
        ax.set_xscale("log")
        ax.set_ylabel("retained MB\n(%s)" % MOTION_CASE[c])
        ax.yaxis.grid(True, zorder=0)
        ax.set_axisbelow(True)
        despine(ax)
    axes[-1][0].set_xlabel("steps in the run (change at the midpoint)")
    legend_below(fig, axes[-1][0], ncol=1 if args.width < DOUBLE_COL else 2)
    return fig, "motion_retention"


DECODE_SYSTEMS = ["ours", "torch-eager", "torch-compile", "torch-compile-ro"]


def fig_decode(out, args):
    """Next-token latency with a key/value cache, per model, medians with
    sign-test intervals over fresh processes.  Stacked one model per row so
    it fits a column."""
    rows = read_tsv(os.path.join(out, "decode.tsv"))
    if not rows:
        return None
    models = []
    for r in rows:
        if r["model"] not in models:
            models.append(r["model"])
    systems = [s for s in DECODE_SYSTEMS if any(r["system"] == s for r in rows)]
    g = collections.defaultdict(list)
    for r in rows:
        try:
            g[(r["model"], r["system"])].append(float(r["token_us"]))
        except ValueError:
            pass
    fig, ax = plt.subplots(figsize=(args.width, 0.55 + 0.62 * len(models)))
    height = 0.8 / max(len(systems), 1)
    table = []
    for i, sysname in enumerate(systems):
        vals, lo, hi, ys = [], [], [], []
        for mi, m in enumerate(models):
            xs = g.get((m, sysname), [])
            v = med(xs) / 1e3 if xs else 0.0
            a, b = _sign_interval(xs) if xs else (0.0, 0.0)
            vals.append(v)
            lo.append(a / 1e3)
            hi.append(b / 1e3)
            ys.append(mi + (i - (len(systems) - 1) / 2.0) * height)
        colour, hatch, _ = style_of(sysname)
        ax.barh(ys, vals, height * 0.9, color=colour, linewidth=0,
                hatch=hatch if args.texture else None,
                label=SYSTEM_LABEL.get(sysname, sysname))
        ax.errorbar(vals, ys, xerr=err(vals, lo, hi), fmt="none", zorder=4,
                    **ERRBAR)
        for y, v in zip(ys, vals):
            if v:
                ax.text(v * 1.02, y, "%.2f" % v, va="center", fontsize=5.5)
    ax.set_yticks(range(len(models)), models)
    ax.invert_yaxis()
    ax.set_xlabel("next-token latency, ms (lower is better)")
    ax.xaxis.grid(True, zorder=0)
    ax.set_axisbelow(True)
    ax.set_xlim(0, ax.get_xlim()[1] * 1.12)
    despine(ax)
    legend_below(fig, ax, ncol=1 if args.width < DOUBLE_COL else 2)
    for m in models:
        base = med(g.get((m, "ours"), [])) if g.get((m, "ours")) else 0.0
        for sysname in systems:
            xs = g.get((m, sysname), [])
            if not xs:
                continue
            a, b = _sign_interval(xs)
            rs = [r for r in rows if r["model"] == m and r["system"] == sysname]
            table.append([m, SYSTEM_LABEL.get(sysname, sysname),
                          "%.0f [%.0f, %.0f]" % (med(xs), a, b),
                          "%.2f" % (med(xs) / base) if base else "n/a",
                          "%s" % (rs[0].get("launches_per_token") or "-"),
                          "%d/%d" % (sum(r["tokens_match"] == "1" for r in rs),
                                     len(rs))])
    write_table(os.path.join(args.outdir, "decode.tex"),
                "next-token latency (us), greedy decode with a key/value "
                "cache, medians and sign-test intervals over fresh processes",
                ["model", "system", "token us", "vs ours", "launches/token",
                 "tokens match"], table)
    return fig, "decode"


def fig_deopt(out, args):
    """Two questions, two panels, same rows.  Left: what an iteration costs
    once the dust settles, which is where a bridge either did or did not
    recover the steady state.  Right: the most expensive single iteration of
    the run - the deoptimization itself.  That one spans four orders of
    magnitude between a bridge compile and a dynamo recompilation, so it is
    markers on a log axis, not bar length."""
    rows = read_tsv(os.path.join(out, "deopt.tsv"))
    if not rows:
        return None
    g = collections.defaultdict(lambda: collections.defaultdict(list))
    for r in rows:
        for col in ("steady_us", "first_fail_us", "after_fail_us", "peak_us",
                    "launches_per_iter", "loops", "bridges", "cold_us"):
            v = r.get(col)
            if v not in (None, ""):
                g[(r["pattern"], r["system"])][col].append(float(v))
    patterns = [p for p in DEOPT_PATTERNS if any(k[0] == p for k in g)]
    systems = drawn([s for s in ("ours", "torch-eager", "torch-compile",
                                 "torch-compile-ro")
                     if any(k[1] == s for k in g)], args)
    if not patterns:
        return None

    if args.width >= DOUBLE_COL:
        fig, (ax, ax2) = plt.subplots(
            1, 2, figsize=(args.width, 0.58 * len(patterns) + 1.2),
            gridspec_kw={"width_ratios": [1.25, 1]})
    else:
        # One column is too narrow for two panels side by side once the
        # pattern names are on the left, so they stack.
        fig, (ax, ax2) = plt.subplots(
            2, 1, figsize=(args.width, 0.66 * len(patterns) + 1.5),
            sharey=True)
    height = 0.8 / len(systems)
    for si, system in enumerate(systems):
        colour, hatch, marker = style_of(system)
        ys, xs, lows, highs, pys, pxs = [], [], [], [], [], []
        for i, pattern in enumerate(patterns):
            d = g.get((pattern, system))
            if not d:
                continue
            y = i + ((len(systems) - 1) / 2.0 - si) * height
            m, lo, hi = band(d.get("steady_us", []))
            if m is not None:
                ys.append(y)
                xs.append(m)
                lows.append(lo)
                highs.append(hi)
            pm, _, _ = band(d.get("peak_us", []))
            if pm is not None:
                pys.append(y)
                pxs.append(pm)
        ax.barh(ys, xs, height=height * 0.85, color=colour, linewidth=0,
                label=SYSTEM_LABEL.get(system, system),
                hatch=hatch if args.texture else None)
        if xs:
            ax.errorbar(xs, ys, xerr=err(xs, lows, highs), fmt="none",
                        zorder=4, **ERRBAR)
        ax2.plot(pxs, pys, marker, color=colour, linestyle="none",
                 markersize=4, markeredgewidth=0)
    ticks = [DEOPT_LABEL.get(p, p) for p in patterns]
    ax.set_yticks(range(len(patterns)), ticks)
    ax.set_xlabel("steady-state per iteration (\u00b5s), lower is better")
    ax.xaxis.grid(True, zorder=0)
    ax.set_axisbelow(True)
    ax.set_ylim(-0.6, len(patterns) - 0.4)
    despine(ax, keep=("left",))
    ax.tick_params(axis="y", length=0)
    ax2.set_xscale("log")
    if args.width >= DOUBLE_COL:
        ax2.set_yticks(range(len(patterns)), ["" for _ in patterns])
    else:
        ax2.set_yticks(range(len(patterns)), ticks)
    ax2.set_xlabel("most expensive iteration (\u00b5s, log scale)\n"
                   "lower is better")
    ax2.xaxis.grid(True, zorder=0)
    ax2.set_axisbelow(True)
    ax2.set_ylim(-0.6, len(patterns) - 0.4)
    despine(ax2, keep=("left",))
    ax2.tick_params(axis="y", length=0)
    legend_below(fig, ax, ncol=2)
    if args.titles:
        ax.set_title("Cost of a guard failure")

    def cell(d, col, fmt="%.0f"):
        m, _, _ = band(d.get(col, []))
        return fmt % m if m is not None and m >= 0 else "n/a"

    table = []
    for pattern in patterns:
        for system in ("ours", "torch-eager", "torch-compile",
                       "torch-compile-ro"):
            d = g.get((pattern, system))
            if not d:
                continue
            table.append([SYSTEM_LABEL.get(system, system),
                          DEOPT_LABEL.get(pattern, pattern),
                          cell(d, "steady_us", "%.1f"),
                          cell(d, "first_fail_us"), cell(d, "after_fail_us"),
                          cell(d, "peak_us"), cell(d, "cold_us"),
                          cell(d, "launches_per_iter", "%.2f"),
                          cell(d, "loops"), cell(d, "bridges")])
    write_table(os.path.join(args.outdir, "deopt.tex"),
                "median over the rounds; first/after/peak are single "
                "iterations, peak is the deoptimization itself",
                ["system", "pattern", "steady us", "first fail", "after fail",
                 "peak", "cold", "launch/it", "loops", "bridges"], table)
    return fig, "deopt"


def fig_micro_baselines(out, args):
    """Every backend against the same fused baseline: grouped diverging bars,
    one row per point, same log-ratio convention as fig_micro_speedup."""
    g = micro_values(read_tsv(os.path.join(out, "micro.tsv")))
    systems = ["torch-eager", "torch-compile", "torch-compile-ro", "torch-compile-mat",
               "jax", "iree", "triton", "torch-tensorrt"]
    pts = []
    for (dtype, v, k, n), m in g.items():
        if dtype != "float64" or not m.get("fused"):
            continue
        r = None
        if m.get("torch-compile"):
            r, _, _ = ratio_band(m["torch-compile"], m["fused"])
        pts.append((r, micro_label(v, k, n), m))
    if not pts:
        return None
    pts.sort(key=lambda p: (p[0] is None, p[0]))
    labels = [p[1] for p in pts]
    used = [s for s in systems if any(m.get(s) for _, _, m in pts)]
    if not used:
        return None

    # One marker per system at its ratio to MetaTensor, same convention as the
    # app-level markers in fig_micro_speedup: 28 points x up to 6 systems as
    # grouped bars made the figure absurdly tall, and the ratio is the number
    # that matters here, not the absolute width of a bar.
    fig, ax = plt.subplots(figsize=(args.width, 0.16 * len(pts) + 1.25))
    y = list(range(len(pts)))
    for si_, s in enumerate(drawn(used, args)):
        ys, xs = [], []
        for i, (_, _, m) in enumerate(pts):
            fm = med(m.get("fused", []))
            sm = med(m.get(s, []))
            if fm and sm:
                ys.append(i)
                xs.append(sm / fm)
        if not xs:
            continue
        color, _, marker = style_of(s)
        ax.scatter(xs, ys, s=20, marker=marker,
                   facecolors=color, edgecolors="white",
                   linewidth=0.5, zorder=4, label=SYSTEM_LABEL.get(s, s))
    ax.axvline(1, color=INK_2, linewidth=0.8, zorder=3)
    ax.set_xscale("log", base=2)
    ax.xaxis.set_major_formatter(FuncFormatter(
        lambda v, _: ("%g" % v).rstrip("0").rstrip(".") + "×"))
    ax.set_yticks(y, labels)
    ax.set_ylim(-0.6, len(pts) - 0.4)
    ax.set_xlabel("time relative to MetaTensor\n"
                  "log2 scale; right of 1x is slower")
    ax.xaxis.grid(True, zorder=0)
    ax.set_axisbelow(True)
    despine(ax, keep=("left",))
    ax.tick_params(axis="y", length=0)
    legend_below(fig, ax, ncol=3)
    if args.titles:
        ax.set_title("Baselines relative to MetaTensor")

    header = ["benchmark", "MetaTensor (us)", "eager", "torch.compile"]
    if "jax" in used:
        header.append("JAX/XLA")
    if "iree" in used:
        header.append("IREE")
    if "triton" in used:
        header.append("Triton")
    header.append("MT/Triton")
    table_rows = []
    for _, lab, m in pts:
        fm = med(m.get("fused", []))
        te = med(m.get("torch-eager", []))
        tc = med(m.get("torch-compile", []))
        row = [lab, "%.0f" % fm if fm else "n/a",
               "%.0f" % te if te else "n/a", "%.0f" % tc if tc else "n/a"]
        if "jax" in used:
            jv = med(m.get("jax", []))
            row.append("%.0f" % jv if jv else "n/a")
        if "iree" in used:
            iv = med(m.get("iree", []))
            row.append("%.0f" % iv if iv else "n/a")
        tv = med(m.get("triton", []))
        if "triton" in used:
            row.append("%.0f" % tv if tv else "n/a")
        row.append("%.2f" % (fm / tv) if (fm and tv) else "n/a")
        table_rows.append(row)
    write_table(os.path.join(args.outdir, "micro_baselines.tex"),
                "median us per iteration, MetaTensor and baselines, float64",
                header, table_rows)
    return fig, "micro_baselines"


def fig_compile_overhead(out, args):
    """Compile and first-run cost per backend, and the break-even iteration
    count against eager, read from results.jsonl."""
    records = read_jsonl(os.path.join(out, "results.jsonl"))
    if not records:
        return None
    systems = ["torch-compile", "torch-compile-ro", "torch-compile-mat",
               "torch-tensorrt", "jax", "iree", "triton", "ours"]

    def workload_key(r):
        if r.get("kind") == "micro" and r.get("dtype") == "float64":
            return ("micro", r.get("variant"), r.get("k"), r.get("n"))
        if r.get("kind") == "models":
            return ("models", r.get("model"))
        return None

    def workload_label(key):
        if key[0] == "micro":
            return micro_label(key[1], key[2], key[3])
        return key[1]

    g = collections.defaultdict(lambda: collections.defaultdict(list))
    for r in records:
        key = workload_key(r)
        if key is None:
            continue
        sysname = r.get("mode") if key[0] == "micro" else r.get("system")
        if not sysname:
            continue
        g[key][sysname].append(r)

    def medf(rs, field):
        vals = [r[field] for r in rs
                if isinstance(r.get(field), (int, float)) and r[field] >= 0]
        return statistics.median(vals) if vals else None

    rows = []
    for key in g:
        eager_rs = g[key].get("torch-eager", [])
        eager_first = medf(eager_rs, "first_run_ms")
        eager_steady = medf(eager_rs, "steady_us")
        for s in systems:
            rs = g[key].get(s)
            if not rs:
                continue
            compile_ms = medf(rs, "compile_ms")
            first_ms = medf(rs, "first_run_ms")
            steady = medf(rs, "steady_us")
            be = "n/a"
            if (first_ms is not None and eager_first is not None
                    and eager_steady is not None and steady is not None
                    and eager_steady - steady > 0):
                # compile_ms is paid before the first run where it is
                # reported; torch.compile folds it into first_run_ms.
                be = "%.0f" % ((first_ms + (compile_ms or 0) - eager_first)
                                * 1000 / (eager_steady - steady))
            rows.append((key, s, compile_ms, first_ms, steady, be))
    if not rows:
        return None

    write_table(os.path.join(args.outdir, "compile_overhead.tex"),
                "median compile / first-run cost and break-even iteration "
                "count against torch eager",
                ["workload", "system", "compile (ms)", "first run (ms)",
                 "steady (us)", "break-even (iters)"],
                [[workload_label(key), SYSTEM_LABEL.get(s, s),
                  "%.0f" % c if (c is not None and c >= 0) else "n/a",
                  "%.0f" % f if (f is not None and f >= 0) else "n/a",
                  "%.0f" % st if st is not None else "n/a", be]
                 for key, s, c, f, st, be in rows])

    # The figure is model workloads only - all 28 micro points plus 9 models
    # in one grouped-bar figure makes the page ~4x taller than it needs to be,
    # and the micro/table pairing already covers the micro side in
    # micro_speedup and micro_baselines.
    model_keys = [key for key in dict.fromkeys(k for k, s, c, f, st, be in rows)
                  if key[0] == "models"]
    fallback = not model_keys
    if fallback:
        # No model rows in this run: fall back to the micro rows, capped to
        # the largest workloads so the figure still fits a page.
        keys_order = list(dict.fromkeys(k for k, s, c, f, st, be in rows))
        best = {}
        for key, s, c, f, st, be in rows:
            if f is not None and f > 0:
                best[key] = max(best.get(key, 0), f)
        keys_order = sorted(keys_order, key=lambda k: -best.get(k, 0))[:12]
    else:
        keys_order = model_keys
    plot_rows = [(key, s, f) for key, s, c, f, st, be in rows
                 if f is not None and f > 0 and key in keys_order]
    systems_used = [s for s in drawn(systems, args)
                    if any(pr[1] == s for pr in plot_rows)]
    if not plot_rows or not systems_used:
        return None

    h = 0.8 / len(systems_used)
    fig, ax = plt.subplots(figsize=(args.width, 0.42 * len(keys_order) + 1.3))
    y = list(range(len(keys_order)))
    for si_, s in enumerate(systems_used):
        off = ((len(systems_used) - 1) / 2 - si_) * h
        ys, xs = [], []
        for i, key in enumerate(keys_order):
            match = [f for k, ss, f in plot_rows if k == key and ss == s]
            if match:
                ys.append(i + off)
                xs.append(match[0])
        if not xs:
            continue
        ax.barh(ys, xs, height=h * 0.86, color=style_of(s)[0],
                linewidth=0,
                hatch=style_of(s)[1] if args.texture else None,
                label=SYSTEM_LABEL.get(s, s))
    ax.set_xscale("log")
    ax.set_yticks(y, [workload_label(k) for k in keys_order])
    ax.set_xlabel("first-run latency (ms, log scale)\nlower is better")
    ax.xaxis.grid(True, zorder=0)
    ax.set_axisbelow(True)
    despine(ax, keep=("left",))
    ax.tick_params(axis="y", length=0)
    legend_below(fig, ax, ncol=3)
    if args.titles:
        ax.set_title("Compile / first-run overhead")
    return fig, "compile_overhead"


# --- machines --------------------------------------------------------------


def fig_gap(out, args):
    """Two things explain the same gap and they are in different units, so they
    get one panel each against a shared list of models: how many kernels a
    forward launches, and how much of the wall clock those kernels occupy."""
    write_gemm_count(out, args)
    rows = read_tsv(os.path.join(out, "gap.tsv"))
    rows = [r for r in rows if r.get("launches_per_iter")]
    if not rows:
        return None
    g = collections.defaultdict(dict)
    for r in rows:
        g[r["model"]][r["system"]] = r
    systems = [s for s in ["ours", "torch-compile", "jax"]
               if any(s in d for d in g.values())]
    models = sorted(g, key=lambda m: float(g[m].get("ours", {}).get("steady_us") or 0))

    def col(sys_, model, key, default=0.0):
        v = g[model].get(sys_, {}).get(key)
        return float(v) if v else default

    wide = args.width >= DOUBLE_COL
    if wide:
        fig, axes = plt.subplots(1, 2, sharey=True,
                                 figsize=(args.width,
                                          0.42 * len(models) + 1.2))
    else:
        fig, axes = plt.subplots(2, 1, sharey=True,
                                 figsize=(args.width,
                                          0.60 * len(models) + 1.3))
    h = 0.8 / len(systems)
    y = list(range(len(models)))
    panels = [(axes[0], "launches_per_iter",
               "kernel launches per forward, lower is better"),
              (axes[1], "gpu_util", "GPU busy / wall time, higher is better")]
    for ax, key, xlabel in panels:
        for si_, s in enumerate(systems):
            off = ((len(systems) - 1) / 2 - si_) * h
            colour, hatch, _ = style_of(s)
            ax.barh([v + off for v in y], [col(s, m, key) for m in models],
                    height=h * 0.88, color=colour,
                    linewidth=0,
                    hatch=hatch if args.texture else None,
                    label=SYSTEM_LABEL[s])
        ax.set_xlabel(xlabel)
        ax.xaxis.grid(True, zorder=0)
        ax.set_axisbelow(True)
        despine(ax, keep=("left",))
        ax.tick_params(axis="y", length=0)
    axes[1].set_xlim(0, 1)
    axes[0].set_yticks(y, models)
    if not wide:
        axes[1].set_yticks(y, models)
    legend_below(fig, axes[0], ncol=3)
    if args.titles:
        fig.suptitle("Where the gap comes from")

    header = ["model", "system", "us/iter", "launches", "kernels", "GPU us",
              "GPU util", "gemm", "split-K", "bmm", "gen", "copy", "other"]
    table_rows = []
    for m in models:
        for s in systems:
            r = g[m].get(s)
            if not r:
                continue
            mix = gap_mix(r)
            table_rows.append([m, SYSTEM_LABEL[s], r["steady_us"],
                               r["launches_per_iter"], r["kernels"],
                               r["gpu_busy_us"], r["gpu_util"]] +
                              [gap_num(mix[k])
                               for k in ("gemm", "splitk", "bmm", "gen",
                                         "copy", "other")])
    write_table(os.path.join(args.outdir, "gap.tex"),
                "per forward: nsys launch counts and GPU busy time against the "
                "measured wall time, and the launches attributed kernel by "
                "kernel - cuBLAS GEMMs, the cuBLASLt split-K reduce/epilogue "
                "kernels that finish some of them, generated "
                "(Triton/Inductor/XLA) kernels, copies, everything else; the "
                "five classes sum to the launch count (checked before "
                "rounding - these are per-forward averages over a differenced "
                "trace window and are fractional when cuBLAS does not pick "
                "the same kernel every iteration), and bmm is the batched "
                "(attention) subset of gemm, attributed by launch "
                "multiplicity - two products per layer, so the GEMM kernel "
                "launched 2L times on an L-layer model - and name-based where "
                "the kernel name says batched or attention",
                header, table_rows)
    return fig, "gap"


def gap_mix(r):
    """A gap.tsv row's launch attribution, as counts.

    Since the trace-based breakdown was added the classes are their own
    numeric columns; a row written before that carries them only inside the
    free-text `notes` mix ("gemm=43 gen=24"), which is parsed as the
    fallback so an older result set still renders.  `bmm` (the batched
    attention products inside `gemm`) exists only in the new rows."""
    mix = {}
    for part in (r.get("notes") or "").split():
        key, sep, value = part.partition("=")
        if sep and value.isdigit():
            mix[key] = int(value)
    out = {}
    for key in ("gemm", "splitk", "gen", "copy", "other", "bmm"):
        v = r.get(key)
        out[key] = float(v) if v not in (None, "") else float(mix.get(key, 0))
    return out


def gap_num(v):
    """A class count as the table prints it.  These are per-forward averages
    over a differenced trace window, so they are genuinely fractional when
    cuBLAS picks a different kernel on some iterations; one decimal, and the
    sum check that backs the caption is done on the unrounded values."""
    return "%.1f" % v


def gap_rows(out):
    """gap.tsv for this run, or the newest earlier run on the same machine.

    `bench.sh gap` needs nsys, which is not installed everywhere; a run
    without it has no gap.tsv at all, and the launch attribution does not
    change from run to run the way a timing does.  Returns (rows, source),
    source being None when the rows are this run's own."""
    path = os.path.join(out, "gap.tsv")
    rows = [r for r in read_tsv(path) if r.get("launches_per_iter")]
    if rows:
        return rows, None
    parent = os.path.dirname(os.path.abspath(out))
    for sib in sorted(glob.glob(os.path.join(parent, "*", "gap.tsv")),
                      reverse=True):
        rows = [r for r in read_tsv(sib) if r.get("launches_per_iter")]
        if rows:
            return rows, os.path.basename(os.path.dirname(sib))
    return [], None


def write_gemm_count(out, args):
    """figures/gemm_count.tex: how a forward's launches split into matrix
    products, generated kernels and everything else.

    The reviewer's question is whether the systems are running the same
    architecture, and the launch total alone cannot answer it - a system that
    issues three GEMMs where another issues one fused GEMM has the same
    total if it also fuses one elementwise kernel more.  The split comes from
    gap.tsv's mix column ("gemm=43 gen=24 copy=6 other=2"), which
    gap_analysis attributes kernel by kernel from the nsys trace."""
    rows, source = gap_rows(out)
    # gap.tsv is appended to, so one model can be re-measured without losing
    # the rest; the last row for a (model, system) is the current one, which
    # is how fig_gap reads it too.
    latest = collections.OrderedDict()
    for r in rows:
        latest[(r["model"], r["system"])] = r
    header = ["model", "system", "gemm", "split-K", "bmm", "gen", "copy",
              "other", "total", "launches"]
    table = []
    for r in latest.values():
        mix = gap_mix(r)
        if not any(mix.values()):
            continue
        total = sum(mix[k] for k in ("gemm", "splitk", "gen", "copy", "other"))
        table.append([r["model"], SYSTEM_LABEL.get(r["system"], r["system"]),
                      gap_num(mix["gemm"]), gap_num(mix["splitk"]),
                      gap_num(mix["bmm"]), gap_num(mix["gen"]),
                      gap_num(mix["copy"]), gap_num(mix["other"]),
                      gap_num(total), r.get("launches_per_iter") or "-"])
    if not table:
        return
    caption = ("kernel launches per forward, attributed kernel by kernel: "
               "matrix products, the cuBLAS split-K reduce/epilogue kernels "
               "that finish some of them (kernels, not calls; a run traced "
               "before gap\\_analysis separated them reports 0 and folds them "
               "into gemm), the batched attention products inside gemm "
               "(bmm), generated elementwise/reduction kernels, copies, "
               "everything else. total is the five classes added up and "
               "equals the measured launch count, checked before rounding: "
               "the counts are per-forward averages over a differenced trace "
               "window and are fractional where cuBLAS does not pick the same "
               "kernel every iteration. bmm is a subset of gemm and is not "
               "added in; attention products are attributed by launch "
               "multiplicity, two per layer, so the GEMM kernel launched 2L "
               "times per forward on an L-layer model is the attention pair "
               "whatever cuBLAS named it, and name-based where the kernel "
               "name says batched or attention. gap\\_kernels.tsv is where a "
               "forward's launches are attributed kernel by kernel")
    if source:
        caption += " (from %s: this run has no nsys trace)" % source
    write_table(os.path.join(args.outdir, "gemm_count.tex"), caption,
                header, table)


def machine_label(path):
    """The accelerator, never the host: a published figure should not name
    somebody's server. Runs predating machine.txt fall back to the directory."""
    gpu = None
    txt = os.path.join(path, "machine.txt")
    if os.path.exists(txt):
        for line in open(txt):
            key, _, value = line.partition(" ")
            if key == "gpu" and value.strip():
                gpu = value.strip()
    if not gpu:
        parent = os.path.basename(os.path.dirname(os.path.abspath(path)))
        if "-" in parent:
            gpu = parent.partition("-")[2]
    return pretty_gpu(gpu or "unknown GPU")


def machine_labels(paths):
    """Distinct labels without falling back to a host; same-model runs are
    separated by their date."""
    labels = [machine_label(p) for p in paths]
    if len(set(labels)) == len(labels):
        return labels
    out = []
    for path, label in zip(paths, labels):
        if labels.count(label) > 1:
            out.append("%s, %s" % (label, run_date(path)))
        else:
            out.append(label)
    return out


def run_date(path):
    txt = os.path.join(path, "machine.txt")
    if os.path.exists(txt):
        for line in open(txt):
            key, _, value = line.partition(" ")
            if key == "date" and value.strip():
                return value.strip()[:10]
    base = os.path.basename(os.path.abspath(path))
    m = re.search(r"\d{4}-\d{2}-\d{2}", base)
    return m.group(0) if m else base


def pretty_gpu(name):
    name = name.replace("NVIDIA ", "").replace("GeForce ", "").strip()
    m = re.match(r"^rtx(\d{3,4})(ti|super)?$", name, re.I)
    if m:
        return "RTX %s%s" % (m.group(1), " Ti" if (m.group(2) or "").lower() == "ti"
                             else " SUPER" if m.group(2) else "")
    return name


def _speedup_rows(out):
    g = micro_values(read_tsv(os.path.join(out, "micro.tsv")))
    rows = {}
    for (dtype, v, k, n), m in g.items():
        if dtype != "float64" or not m.get("fused") or not m.get("torch-compile"):
            continue
        rows[(v, k, n)] = (micro_label(v, k, n),) + \
            ratio_band(m["torch-compile"], m["fused"])
    return rows


def _grouped_ratio_bars(ax, keys, per_machine, labels, args, xlabel, bands=None):
    y = list(range(len(keys)))
    h = 0.8 / len(labels)
    for mi, label in enumerate(labels):
        vals = [per_machine[mi][k] for k in keys]
        off = ((len(labels) - 1) / 2 - mi) * h
        ax.barh([i + off for i in y], [v - 1 for v in vals], left=1,
                height=h * 0.86, color=MACHINE_COLORS[mi], linewidth=0,
                hatch=MACHINE_HATCH[mi] if args.texture else None,
                label=label)
        if bands:
            ax.errorbar(vals, [i + off for i in y],
                        xerr=err(vals, [bands[mi][k][0] for k in keys],
                                 [bands[mi][k][1] for k in keys]),
                        fmt="none", zorder=4, **ERRBAR)
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
    labels = machine_labels(outs)
    common = set(rows[0])
    for r in rows[1:]:
        common &= set(r)
    dropped = sorted(set().union(*(set(r) for r in rows)) - common)
    if not common:
        return None
    keys = sorted(common, key=lambda k: rows[0][k][1])
    per = [{k: r[k][1] for k in keys} for r in rows]
    bands = [{k: (r[k][2], r[k][3]) for k in keys} for r in rows]

    fig, ax = plt.subplots(figsize=(args.width, 0.20 * len(keys) + 1.2))
    _grouped_ratio_bars(ax, keys, per, labels, args,
                        "speedup over torch.compile\n"
                        "log2 scale; higher is better", bands)
    ax.set_yticks(list(range(len(keys))), [rows[0][k][0] for k in keys])
    legend_below(fig, ax, ncol=2)
    if args.titles:
        ax.set_title("Microbenchmark speedup across machines")
    if dropped:
        print("compare_speedup: %d point(s) not measured at the same size on "
              "every machine, left out: %s"
              % (len(dropped), ", ".join(micro_label(*k) for k in dropped)),
              file=sys.stderr)
    write_table(os.path.join(args.outdir, "compare_speedup.tex"),
                "fused speedup over torch.compile, float64, with the range "
                "over the rounds",
                ["benchmark"] + [c for lab in labels for c in (lab, "range")],
                [[rows[0][k][0]]
                 + [c for p, b in zip(per, bands)
                    for c in ("%.2f" % p[k], spread_str(*b[k], fmt="%.2f"))]
                 for k in keys])
    return fig, "compare_speedup"


def fig_compare_models(out, args):
    """End-to-end ratio to torch.compile, per model, per machine."""
    outs = [out] + args.compare
    labels = machine_labels(outs)
    per, bands = [], []
    for o in outs:
        g = collections.defaultdict(lambda: collections.defaultdict(list))
        for r in read_tsv(os.path.join(o, "models.tsv")):
            g[r["model"]][r["system"]].append(float(r["steady_us"]))
        usable = {m: ratio_band(d["torch-compile"], d["ours"]) for m, d in g.items()
                  if d.get("ours") and d.get("torch-compile")}
        per.append({m: b[0] for m, b in usable.items()})
        bands.append({m: (b[1], b[2]) for m, b in usable.items()})
    common = set(per[0])
    for p in per[1:]:
        common &= set(p)
    if not common:
        return None
    keys = sorted(common, key=lambda m: per[0][m])

    fig, ax = plt.subplots(figsize=(args.width, 0.30 * len(keys) + 1.25))
    _grouped_ratio_bars(ax, keys, per, labels, args,
                        "speedup over torch.compile\n"
                        "log2 scale; higher is better", bands)
    ax.set_yticks(list(range(len(keys))), keys)
    legend_below(fig, ax, ncol=2)
    if args.titles:
        ax.set_title("End-to-end speedup across machines")
    write_table(os.path.join(args.outdir, "compare_models.tex"),
                "end-to-end speedup over torch.compile, with the range over "
                "the rounds",
                ["model"] + [c for lab in labels for c in (lab, "range")],
                [[m] + [c for p, b in zip(per, bands)
                        for c in ("%.2f" % p[m], spread_str(*b[m], fmt="%.2f"))]
                 for m in keys])
    return fig, "compare_models"


def fig_compare_dynamic(out, args):
    """Cost against sequence length, as a ratio so two GPUs share one axis."""
    outs = [out] + args.compare
    labels = machine_labels(outs)
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

    fig, ax = plt.subplots(figsize=(args.width, args.width * 0.72))
    markers = ["o", "s", "^", "D"]
    for mi, (g, label) in enumerate(zip(series, labels)):
        ys = []
        for L in lengths:
            ours = med(g.get(("ours", L), []))
            comp = med(g.get(("torch-compile-dynamic", L), []))
            ys.append(comp / ours if ours and comp else float("nan"))
        ax.plot(lengths, ys, color=MACHINE_COLORS[mi], linewidth=1.4,
                marker=markers[mi], markersize=3.4, markeredgewidth=0,
                label=label, clip_on=False)
    ax.axhline(1, color=INK_2, linewidth=0.8, zorder=3)
    ax.set_xticks(lengths)
    ax.set_xlim(lengths[0] - 3, lengths[-1] + 3)
    ax.set_xlabel("sequence length (tokens)")
    ax.set_ylabel("speedup over torch.compile (\u00d7)\nhigher is better")
    ax.yaxis.set_major_formatter(FuncFormatter(
        lambda v, _: ("%g" % v) + "\u00d7"))
    ax.yaxis.grid(True, zorder=0)
    ax.set_axisbelow(True)
    despine(ax)
    legend_below(fig, ax, ncol=2)
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


def fig_integration(out, args):
    """Language-integration cost: the five control-flow micro variants (guard,
    force, value guard, exception, side effect) at k=4 n=1M float64, one row
    per system.  torch's graphs/breaks land in the launches_per_iter/graphs
    tsv columns (see the README note on micro.tsv's shifted torch columns),
    read here by that same shift rather than by their header name."""
    rows = read_tsv(os.path.join(out, "micro.tsv"))
    variants, k, n = [1, 2, 3, 4, 5], 4, 1000000
    dtype_of = lambda r: r.get("breaks") or r.get("graphs") or "float64"
    g = collections.defaultdict(lambda: collections.defaultdict(list))
    for r in rows:
        try:
            v, rk, rn = int(r["variant"]), int(r["k"]), int(r["n"])
        except (KeyError, ValueError, TypeError):
            continue
        if v in variants and rk == k and rn == n and dtype_of(r) == "float64":
            g[v][r["mode"]].append(r)
    if not g:
        return None

    def fnum(recs, field):
        xs = [float(x[field]) for x in recs
              if x.get(field) not in (None, "", "-1")]
        return statistics.median(xs) if xs else None

    table, bars = [], []
    for v in variants:
        m = g.get(v, {})
        ours_us = fnum(m.get("fused", []), "steady_us")
        ours_launches = fnum(m.get("fused", []), "launches_per_iter")
        tc_us = fnum(m.get("torch-compile", []), "steady_us")
        tc_graphs = fnum(m.get("torch-compile", []), "launches_per_iter")
        tc_breaks = fnum(m.get("torch-compile", []), "graphs")
        eager_us = fnum(m.get("torch-eager", []), "steady_us")
        jax_us = fnum(m.get("jax", []), "steady_us")
        ratio = tc_us / ours_us if tc_us and ours_us else None
        label = VARIANT_LABEL.get(v, "v%d" % v)
        table.append([
            label,
            "%.1f" % ours_us if ours_us else "n/a",
            "%.2f" % ours_launches if ours_launches is not None else "n/a",
            "%.1f" % tc_us if tc_us else "n/a",
            "%d" % tc_graphs if tc_graphs is not None else "n/a",
            "%d" % tc_breaks if tc_breaks is not None else "n/a",
            "%.1f" % eager_us if eager_us else "n/a",
            "%.1f" % jax_us if jax_us else "n/a",
            "%.2f" % ratio if ratio else "n/a",
        ])
        bars.append((label, {"ours": ours_us, "torch-compile": tc_us,
                             "torch-eager": eager_us, "jax": jax_us}))

    systems = [s for s in ["ours", "torch-compile", "torch-eager", "jax"]
               if any(b[1].get(s) for b in bars)]
    h = 0.8 / len(systems)
    fig, ax = plt.subplots(figsize=(args.width, 0.42 * len(bars) + 1.2))
    y = list(range(len(bars)))
    for si_, s in enumerate(systems):
        off = ((len(systems) - 1) / 2 - si_) * h
        vals = [b[1].get(s) or 0 for b in bars]
        color, hatch, _ = style_of(s)
        ax.barh([yv + off for yv in y], vals, height=h * 0.88,
                color=color, linewidth=0,
                hatch=hatch if args.texture else None,
                label=SYSTEM_LABEL.get(s, s))
    ax.set_yticks(y, [b[0] for b in bars])
    ax.set_xlabel("steady-state time per iteration (µs)\nlower is better")
    ax.xaxis.grid(True, zorder=0)
    ax.set_axisbelow(True)
    despine(ax, keep=("left",))
    ax.tick_params(axis="y", length=0)
    legend_below(fig, ax, ncol=2)
    if args.titles:
        ax.set_title("Language integration cost")

    write_table(os.path.join(args.outdir, "integration.tex"),
                "control-flow variants at k=4 n=1M float64: steady us, "
                "MetaTensor launches/iter, torch.compile graphs/breaks, "
                "ratio torch.compile/ours",
                ["variant", "MetaTensor us", "launches/iter",
                 "torch.compile us", "graphs", "breaks", "eager us",
                 "JAX us", "compile/ours"],
                table)
    return fig, "integration"


def fig_warmup(out, args):
    """Per-forward latency from a fresh process until steady state: one panel
    per model, cumulative time (log-log) so the crossover between systems and
    the tail flattening into steady state both read directly off the line."""
    rows = read_tsv(os.path.join(out, "warmup.tsv"))
    if not rows:
        return None
    series = collections.defaultdict(dict)  # (model,system,cache,round) -> {iter: us}
    for r in rows:
        key = (r["model"], r["system"], r["cache"], r["round"])
        series[key][int(r["iter"])] = float(r["us"])
    models = sorted({k[0] for k in series})
    systems = [s for s in ["torch-eager", "ours", "torch-compile",
                           "torch-compile-ro", "jax"]
               if any(k[1] == s for k in series)]

    if args.width >= DOUBLE_COL:
        fig, axes = plt.subplots(1, len(models), sharey=True, squeeze=False,
                                 figsize=(args.width, args.width * 0.78))
        axes = list(axes[0])
    else:
        fig, axes = plt.subplots(len(models), 1, sharex=True, sharey=True,
                                 squeeze=False,
                                 figsize=(args.width,
                                          1.35 * len(models) + 0.9))
        axes = [a[0] for a in axes]
    for ax, model in zip(axes, models):
        for si_, s in enumerate(systems):
            colour, _, _ = style_of(s)
            for cache, ls in (("warm", "-"), ("cold", "--")):
                keys = [k for k in series
                        if k[0] == model and k[1] == s and k[2] == cache]
                if not keys:
                    continue
                curves = []
                for k in keys:
                    d = series[k]
                    n = max(d) + 1
                    acc, cum = 0.0, []
                    for it in range(n):
                        acc += d.get(it, 0.0)
                        cum.append(acc / 1000.0)  # ms
                    curves.append(cum)
                n = min(len(c) for c in curves)
                med_curve = [statistics.median(c[i] for c in curves)
                            for i in range(n)]
                ax.plot(range(1, n + 1), med_curve, color=colour,
                        linestyle=ls, linewidth=1.2)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_title(model, fontsize=8)
        despine(ax)
    if args.width >= DOUBLE_COL:
        for ax in axes:
            ax.set_xlabel("forward index (log scale)")
    else:
        axes[-1].set_xlabel("forward index (log scale)")
    axes[0].set_ylabel("cumulative ms (log)\nlower is better")
    # Colour is the system, dash is the kernel cache: ten curves need seven
    # legend entries, not ten.
    handles = [Line2D([], [], color=style_of(s)[0], lw=1.6,
                      label=SYSTEM_LABEL.get(s, s)) for s in systems]
    handles += [Line2D([], [], color=INK_2, lw=1.2, linestyle="-",
                       label="warm kernel cache"),
                Line2D([], [], color=INK_2, lw=1.2, linestyle="--",
                       label="cold kernel cache")]
    legend_below(fig, handles=handles,
                 labels=[h.get_label() for h in handles], ncol=3)
    if args.titles:
        fig.suptitle("Warm-up: cumulative time from a fresh process")

    jrows = [r for r in read_jsonl(os.path.join(out, "results.jsonl"))
            if r.get("kind") == "warmup"]
    g = collections.defaultdict(lambda: collections.defaultdict(list))
    for r in jrows:
        g[(r["model"], r["system"], r["cache"])]["first_us"].append(r.get("first_us"))
        g[(r["model"], r["system"], r["cache"])]["steady_us"].append(r.get("steady_us"))
        g[(r["model"], r["system"], r["cache"])]["steady_at"].append(r.get("steady_at"))
        g[(r["model"], r["system"], r["cache"])]["crossover"].append(r.get("crossover"))
    table = []
    for model in models:
        for s in systems:
            for cache in ("warm", "cold"):
                d = g.get((model, s, cache))
                if not d:
                    continue
                first = med(d["first_us"])
                steady = med(d["steady_us"])
                sa = [x for x in d["steady_at"] if isinstance(x, (int, float))]
                sa_str = "%.0f" % statistics.median(sa) if sa else "none"
                co = [x for x in d["crossover"] if isinstance(x, (int, float))]
                co_str = "%.0f" % statistics.median(co) if co else "none"
                table.append([model, SYSTEM_LABEL.get(s, s), cache,
                             "%.2f" % (first / 1000.0) if first else "n/a",
                             sa_str, "%.1f" % steady if steady else "n/a", co_str])
    write_table(os.path.join(args.outdir, "warmup.tex"),
               "first forward and steady-state onset from a fresh process, "
               "median over the rounds; crossover is the first forward at "
               "which cumulative time drops below torch-eager's",
               ["model", "system", "cache", "first forward (ms)",
                "steady_at (iters)", "steady (us)", "crossover vs eager (iters)"],
               table)
    return fig, "warmup"


def fig_batch(out, args):
    """Batch-size sweep: one panel per model, x = batch (log2),
    y = per-sequence microseconds (log), one line per system.

    Per sequence rather than per forward, because per forward every line
    climbs and the figure would only say "more work takes longer".  What the
    sweep is for is where the lines *meet*: at batch 1 the systems are
    separated by per-launch overhead, and as the GEMMs grow that overhead is
    amortised until only the arithmetic is left.
    """
    rows = read_tsv(os.path.join(out, "batch.tsv"))
    if not rows:
        return None
    per_seq = collections.defaultdict(list)
    steady = collections.defaultdict(list)
    ident = collections.defaultdict(list)
    failed = collections.defaultdict(list)
    for r in rows:
        key = (r["model"], r["system"], int(r["batch"]))
        if (r.get("status") or "ok") != "ok" or not r.get("per_seq_us"):
            failed[key].append(1)
            continue
        per_seq[key].append(float(r["per_seq_us"]))
        steady[key].append(float(r["steady_us"]))
        if r.get("batch_rows_identical") not in (None, ""):
            ident[key].append(r["batch_rows_identical"])
    if not per_seq:
        return None
    models = sorted({k[0] for k in per_seq})
    systems = [s for s in ["ours", "torch-compile", "torch-compile-ro",
                           "torch-eager", "jax"]
               if any(k[1] == s for k in per_seq)]
    batches = sorted({k[2] for k in per_seq} | {k[2] for k in failed})

    if args.width >= DOUBLE_COL:
        fig, axes = plt.subplots(1, len(models), squeeze=False, sharey=True,
                                 figsize=(args.width, args.width * 0.42 + 0.5))
        axes = axes[0]
    else:
        # At one column two panels side by side leave 1.2in each; they stack.
        fig, axes = plt.subplots(len(models), 1, squeeze=False, sharex=True,
                                 figsize=(args.width,
                                          1.35 * len(models) + 0.95))
        axes = [a[0] for a in axes]
    for mi, model in enumerate(models):
        ax = axes[mi]
        for s in drawn(systems, args):
            pts = [(b, band(per_seq[(model, s, b)])) for b in batches
                   if (model, s, b) in per_seq]
            if not pts:
                continue
            xs = [p[0] for p in pts]
            ys = [p[1][0] for p in pts]
            color, _, marker = style_of(s)
            ax.errorbar(xs, ys, yerr=err(ys, [p[1][1] for p in pts],
                                         [p[1][2] for p in pts]),
                        fmt="none", zorder=2, **ERRBAR)
            ax.plot(xs, ys, color=color, linewidth=1.4, marker=marker,
                    markersize=3.4, markeredgewidth=0, zorder=3,
                    label=SYSTEM_LABEL[s] if mi == 0 else None)
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_xticks(batches)
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: "%d" % v))
        if ax is axes[-1] or args.width >= DOUBLE_COL:
            ax.set_xlabel("batch size (sequences per forward)")
        ax.set_title(model, fontsize=8)
        ax.yaxis.grid(True, zorder=0)
        ax.set_axisbelow(True)
        despine(ax)
    for ax in (axes if args.width < DOUBLE_COL else axes[:1]):
        ax.set_ylabel("\u00b5s per sequence\n(log, lower is better)")
    legend_below(fig, axes[0], ncol=2)

    def cell(key, table, spec="%.0f"):
        v = med(table.get(key, []))
        return (spec % v) if v else "n/a"

    header = ["model", "batch"] + [SYSTEM_LABEL[s] for s in systems] + [
        "ours/compile", "ours/jax", "rows identical"]
    table_rows = []
    for model in models:
        for b in batches:
            row = [model, str(b)]
            for s in systems:
                key = (model, s, b)
                row.append("failed" if key in failed and key not in steady
                           else cell(key, steady))
            ours = med(steady.get((model, "ours", b), []))
            tc = med(steady.get((model, "torch-compile", b), []))
            jx = med(steady.get((model, "jax", b), []))
            row.append("%.2f" % (ours / tc) if ours and tc else "n/a")
            row.append("%.2f" % (ours / jx) if ours and jx else "n/a")
            marks = [v for s in systems for v in ident.get((model, s, b), [])]
            row.append("n/a" if not marks
                       else ("yes" if all(v == "1" for v in marks) else "NO"))
            table_rows.append(row)
    write_table(os.path.join(args.outdir, "batch.tex"),
                "steady-state us per forward (the whole batch), median over "
                "the rounds, and the ours/torch.compile and ours/JAX ratios; "
                "\"rows identical\" is the batching check - every sequence in "
                "the batch is the same token ids, so every output row block "
                "must match the first",
                header, table_rows)
    return fig, "batch"


def fig_correctness(out, args):
    """Correctness on more than one input: the worst case over the five
    derived inputs per model (see run_correctness.sh), not the average - a
    system that is right four times out of five is wrong."""
    rows = read_tsv(os.path.join(out, "correctness.tsv"))
    rows = [r for r in rows if r.get("system") != "ours" and r.get("maxabsdiff")]
    if not rows:
        return None
    g = collections.defaultdict(lambda: collections.defaultdict(list))
    for r in rows:
        g[r["model"]][r["system"]].append(r)
    systems = [s for s in ["torch-eager", "torch-compile", "jax"]
               if any(s in d for d in g.values())]
    # A result set recorded before the selection rule was applied still
    # holds rows for models that are no longer measured; the figure shows
    # the population.
    models = sorted(m for m in g if m not in PROBES)

    def worst(model, sys_):
        rs = g[model].get(sys_) or []
        if not rs:
            return None
        diffs = [float(r["maxabsdiff"]) for r in rs if r["maxabsdiff"]]
        matches = [float(r["argmax_match"]) for r in rs if r.get("argmax_match")]
        passes = [r.get("pass") for r in rs if r.get("pass") not in (None, "")]
        tol = next((r["tol"] for r in rs if r.get("tol")), "")
        return (max(diffs) if diffs else None,
                min(matches) if matches else None,
                tol, sum(1 for x in passes if x == "1"), len(rs))

    fig, ax = plt.subplots(figsize=(args.width, 0.36 * len(models) + 1.3))
    h = 0.8 / len(systems)
    y = list(range(len(models)))
    for si_, s in enumerate(systems):
        vals = []
        for m in models:
            w = worst(m, s)
            # A zero difference has no place on a log axis and no meaning
            # here either; floor it at the smallest float32 step.
            vals.append(max(w[0], 1e-9) if w and w[0] is not None else 0.0)
        colour, hatch, _ = style_of(s)
        ax.barh([v + ((len(systems) - 1) / 2 - si_) * h for v in y], vals,
                height=h * 0.88, color=colour, linewidth=0,
                hatch=hatch if args.texture else None,
                label=SYSTEM_LABEL[s])
    tols = sorted({float(worst(m, s)[2]) for m in models for s in systems
                   if worst(m, s) and worst(m, s)[2]})
    for t in tols:
        ax.axvline(t, color="0.3", linewidth=0.8, linestyle="--")
    ax.set_xscale("log")
    ax.set_xlabel("max |diff| vs MetaTensor over 5 inputs (log scale)\n"
                  "dashed: tolerance; lower is better")
    ax.set_yticks(y, models)
    ax.xaxis.grid(True, zorder=0)
    ax.set_axisbelow(True)
    despine(ax, keep=("left",))
    ax.tick_params(axis="y", length=0)
    legend_below(fig, ax, ncol=3)
    if args.titles:
        ax.set_title("Correctness on five derived inputs")

    header = ["model", "system", "max |diff|", "min argmax match", "tol",
              "pass"]
    table = []
    for m in models:
        for s in systems:
            w = worst(m, s)
            if not w:
                continue
            diff, match, tol, npass, n = w
            table.append([m, SYSTEM_LABEL[s],
                          "%.3g" % diff if diff is not None else "-",
                          "%.3f" % match if match is not None else "-",
                          tol or "-", "%d/%d" % (npass, n)])
    write_table(os.path.join(args.outdir, "correctness.tex"),
                "each system against MetaTensor on five inputs per model "
                "derived from the stored one (INPUT\\_SEED=1..5): the worst "
                "of the five maximum absolute logit differences, the smallest "
                "argmax agreement over the five (fraction of token positions, "
                "top-1 for the vision models), the tolerance for that "
                "(workload class, dtype), and how many of the five inputs "
                "passed it",
                header, table)
    return fig, "correctness"


# --- where the DAG lives ---------------------------------------------------

LAZY_ARMS = ["virtual", "deferred", "eager"]


def _lazy_group(rows, key):
    g = collections.defaultdict(lambda: collections.defaultdict(list))
    for r in rows:
        g[key(r)][r["system"]].append(r)
    return g


def _lazy_ratio_panel(ax, keys, groups, labels, args, field="steady_us"):
    """Bars per arm, relative to the virtual arm, with the range over rounds."""
    x = list(range(len(keys)))
    width = 0.38
    for i, arm in enumerate(["deferred", "eager"]):
        vals, lows, highs = [], [], []
        for k in keys:
            num = [r[field] for r in groups[k].get(arm, [])]
            den = [r[field] for r in groups[k].get("virtual", [])]
            m, lo, hi = ratio_band(num, den)
            vals.append(m or 0.0)
            lows.append(lo or 0.0)
            highs.append(hi or 0.0)
        colour, hatch, _ = style_of(arm)
        xs = [v + (i - 0.5) * width for v in x]
        ax.bar(xs, vals, width * 0.92, color=colour, linewidth=0,
               hatch=hatch if args.texture else None,
               label=SYSTEM_LABEL.get(arm, arm))
        ax.errorbar(xs, vals, yerr=err(vals, lows, highs), fmt="none",
                    zorder=4, **ERRBAR)
    ax.axhline(1.0, color=INK_2, linewidth=0.8, linestyle="--")
    ax.set_xticks(x, labels, rotation=30, ha="right")
    ax.set_yscale("log")
    ax.yaxis.grid(True, zorder=0)
    ax.set_axisbelow(True)
    despine(ax)


def fig_lazy(out, args):
    """The experiment the whole deferred layer exists for: one binary, one
    kernel emitter, one cache, one allocator, one set of forcing boundaries,
    and the operation DAG in three different places."""
    rows = read_tsv(os.path.join(out, "lazy.tsv"))
    if not rows:
        return None
    groups = _lazy_group(rows, lambda r: r["model"])
    keys = [k for k in collections.OrderedDict.fromkeys(r["model"] for r in rows)]
    fig, axes = plt.subplots(1, 2, figsize=(args.width, 2.5))
    _lazy_ratio_panel(axes[0], keys, groups, keys, args)
    axes[0].set_ylabel("latency relative to the DAG\nin the JIT (lower is better)")

    # Allocation is the other half of the story: the deferred arm rebuilds the
    # DAG on every iteration and the bytes say so.
    x = list(range(len(keys)))
    width = 0.27
    for i, arm in enumerate(LAZY_ARMS):
        vals = []
        for k in keys:
            vals.append(med([r["gc_ms"] for r in groups[k].get(arm, [])]) or 0.0)
        colour, hatch, _ = style_of(arm)
        axes[1].bar([v + (i - 1) * width for v in x], vals, width * 0.92,
                    color=colour, linewidth=0,
                    hatch=hatch if args.texture else None,
                    label=SYSTEM_LABEL.get(arm, arm))
    axes[1].set_ylabel("ms in the collector over the\ntimed loop (lower is better)")
    axes[1].set_xticks(x, keys, rotation=30, ha="right")
    axes[1].yaxis.grid(True, zorder=0)
    axes[1].set_axisbelow(True)
    despine(axes[1])
    handles, labels = axes[1].get_legend_handles_labels()
    legend_below(fig, handles=handles, labels=labels, ncol=3)
    if args.titles:
        axes[0].set_title("Where the operation DAG lives")

    table = []
    for k in keys:
        for arm in LAZY_ARMS:
            rs = groups[k].get(arm, [])
            if not rs:
                continue
            m, lo, hi = band([r["steady_us"] for r in rs])
            table.append([k, SYSTEM_LABEL.get(arm, arm), "%.1f" % m,
                          spread_str(lo, hi),
                          "%.1f" % (med([r["launches_per_iter"] for r in rs]) or 0),
                          "%d" % (med([r["kernels"] for r in rs]) or 0),
                          "%d" % (med([r["compiles"] for r in rs]) or 0),
                          "%d" % (med([r["gc_ms"] for r in rs]) or 0),
                          si(int(med([r.get("peak_bytes") for r in rs]) or 0)),
                          "%d" % (med([r["lazy_nodes"] for r in rs]) or 0),
                          "%d" % (med([r.get("loops") for r in rs]) or 0),
                          "%d" % (med([r.get("bridges") for r in rs]) or 0),
                          rs[0].get("argmax_match", "")])
    write_table(os.path.join(args.outdir, "lazy.tex"),
                "one binary, three places to keep the operation DAG. "
                "kernels and Triton compiles are cumulative process counters "
                "and are cardinalities: they say how many distinct kernels an "
                "arm compiled, not which, and do not identify kernel keys or "
                "order",
                ["model", "DAG lives in", "us/iter", "range", "launches/iter",
                 "kernels", "compiles", "gc ms", "peak bytes",
                 "nodes deferred", "loops", "bridges", "same argmax"], table)
    return fig, "lazy"


def fig_lazy_col(out, args):
    """The body's version: the latency panel alone, at one column, because the
    collector numbers are in the prose and in the appendix table."""
    rows = read_tsv(os.path.join(out, "lazy.tsv"))
    if not rows:
        return None
    groups = _lazy_group(rows, lambda r: r["model"])
    keys = [k for k in collections.OrderedDict.fromkeys(r["model"] for r in rows)]
    fig, ax = plt.subplots(figsize=(args.width, 2.4))
    _lazy_ratio_panel(ax, keys, groups, keys, args)
    ax.set_ylabel("latency relative to the DAG\nin the JIT (lower is better)")
    legend_below(fig, ax, ncol=2)
    return fig, "lazy_col"


def fig_lazy_micro(out, args):
    """The same three arms across the size range, where the host cost of
    rebuilding the DAG is the whole story at one end and invisible at the
    other."""
    rows = read_tsv(os.path.join(out, "lazy_micro.tsv"))
    if not rows:
        return None
    key = lambda r: (int(r["variant"]), int(r["k"]), int(r["n"]))
    groups = _lazy_group(rows, key)
    keys = list(collections.OrderedDict.fromkeys(key(r) for r in rows))
    labels = [micro_label(*k) for k in keys]
    fig, ax = plt.subplots(figsize=(args.width, 2.6))
    _lazy_ratio_panel(ax, keys, groups, labels, args)
    ax.set_ylabel("latency relative to\nthe DAG in the JIT\n(lower is better)")
    legend_below(fig, ax, ncol=2)
    write_table(os.path.join(args.outdir, "lazy_micro.tex"),
                "median us per iteration over the rounds, with launches per "
                "iteration",
                ["benchmark", "DAG lives in", "us/iter", "range",
                 "launches/iter"],
                [[labels[i], SYSTEM_LABEL.get(arm, arm),
                  "%.1f" % band([r["steady_us"] for r in groups[k][arm]])[0],
                  spread_str(*band([r["steady_us"] for r in groups[k][arm]])[1:]),
                  "%.2f" % (med([r["launches_per_iter"] for r in groups[k][arm]]) or 0)]
                 for i, k in enumerate(keys) for arm in LAZY_ARMS
                 if groups[k].get(arm)])
    return fig, "lazy_micro"


def fig_lazy_warmup(out, args):
    """Start-up, where the deferred arm should win: it fuses on the first
    forward, while the trace has to be recorded and optimized first."""
    rows = read_tsv(os.path.join(out, "lazy_warmup.tsv"))
    if not rows:
        return None
    models = list(collections.OrderedDict.fromkeys(r["model"] for r in rows))
    fig, axes = plt.subplots(len(models), 1, figsize=(args.width, 2.0 * len(models)),
                             sharex=True, squeeze=False)
    for ax, model in zip([a[0] for a in axes], models):
        for arm in LAZY_ARMS:
            series = collections.defaultdict(list)
            for r in rows:
                if r["model"] == model and r["system"] == arm:
                    series[int(r["iter"])].append(float(r["us"]))
            if not series:
                continue
            xs = sorted(series)
            colour, _, marker = style_of(arm)
            ax.plot(xs, [med(series[i]) for i in xs], color=colour,
                    linewidth=1.0, marker=marker, markersize=2.2,
                    markevery=max(1, len(xs) // 12),
                    label=SYSTEM_LABEL.get(arm, arm))
        ax.set_yscale("log")
        ax.set_ylabel("%s\nus per forward\n(lower is better)" % model)
        ax.yaxis.grid(True, zorder=0)
        ax.set_axisbelow(True)
        despine(ax)
    axes[-1][0].set_xlabel("forward, from a fresh process with an empty kernel cache")
    legend_below(fig, axes[0][0], ncol=3)
    return fig, "lazy_warmup"


# --- host-dependent control flow in a real model ---------------------------

def fig_control(out, args):
    """Early exit on distilgpt2: the exit layer is decided on the host from a
    value the model just computed."""
    rows = read_tsv(os.path.join(out, "control.tsv"))
    if not rows:
        return None
    regimes = list(collections.OrderedDict.fromkeys(r["regime"] for r in rows))
    systems = list(collections.OrderedDict.fromkeys(r["system"] for r in rows))
    fig, axes = plt.subplots(len(regimes), 1, figsize=(args.width, 1.5 * len(regimes) + 0.8),
                             squeeze=False, sharex=True)
    axes = axes.reshape(1, -1)
    for ax, regime in zip(axes[0], regimes):
        vals, p95s, colours, hatches, names = [], [], [], [], []
        for s in systems:
            rs = [r for r in rows if r["regime"] == regime and r["system"] == s]
            if not rs:
                continue
            vals.append(med([r["p50_us"] for r in rs]) or 0.0)
            p95s.append(med([r["p95_us"] for r in rs]) or 0.0)
            c, h, _ = style_of(s)
            colours.append(c)
            hatches.append(h)
            names.append(SYSTEM_LABEL.get(s, s))
        y = list(range(len(vals)))
        ax.barh(y, vals, 0.62, color=colours, linewidth=0,
                hatch=hatches if args.texture else None)
        for i, (v, p) in enumerate(zip(vals, p95s)):
            ax.plot([p], [i], marker="|", markersize=9, color=INK_2)
        ax.set_yticks(y, names)
        ax.invert_yaxis()
        ax.set_xlabel("us per iteration, median\nand p95 tick (lower is better)")
        ax.xaxis.grid(True, zorder=0)
        ax.set_axisbelow(True)
        despine(ax, keep=("left",))
        ax.tick_params(axis="y", length=0)
        if args.titles or len(regimes) > 1:
            ax.set_title(regime, fontsize=7)
    table = []
    for regime in regimes:
        for s in systems:
            rs = [r for r in rows if r["regime"] == regime and r["system"] == s]
            if not rs:
                continue
            col = lambda c, spec="%.1f": (spec % med([r[c] for r in rs
                                                     if r.get(c) not in (None, "")])
                                          if any(r.get(c) not in (None, "") for r in rs)
                                          else "")
            table.append([regime, SYSTEM_LABEL.get(s, s), col("total_ms", "%.0f"),
                          col("p50_us"), col("p95_us"), col("max_us", "%.0f"),
                          col("bridges", "%.0f"), col("frame_compiles", "%.0f"),
                          rs[0].get("exit_seq", ""),
                          "yes" if all(r.get("pass") == "1" for r in rs) else "NO"])
    write_table(os.path.join(args.outdir, "control.tex"),
                "early exit on distilgpt2, median over three rounds. total ms "
                "is the whole phase including every compilation; frame "
                "compiles counts every frame Dynamo compiled inside that "
                "phase, first compilations included; the exit layers are the "
                "per-iteration sequence of exit depths, which every system has "
                "to reproduce or the row is not comparable",
                ["regime", "system", "total ms", "p50 us", "p95 us", "max us",
                 "bridges", "frame compiles", "exit layers", "pass"],
                table)
    return fig, "control"


def fig_control_cost(out, args):
    """Cumulative time from the first iteration, so every compilation is
    inside the number rather than warmed away before it starts."""
    rows = read_tsv(os.path.join(out, "control_series.tsv"))
    if not rows:
        return None
    regimes = list(collections.OrderedDict.fromkeys(r["regime"] for r in rows))
    systems = list(collections.OrderedDict.fromkeys(r["system"] for r in rows))
    if args.width >= DOUBLE_COL:
        fig, axes = plt.subplots(1, len(regimes), figsize=(args.width, 2.6),
                                 squeeze=False, sharey=True)
        axes = list(axes[0])
    else:
        fig, axes = plt.subplots(len(regimes), 1, squeeze=False,
                                 sharex=True, sharey=True,
                                 figsize=(args.width,
                                          1.4 * len(regimes) + 0.9))
        axes = [a[0] for a in axes]
    for ax, regime in zip(axes, regimes):
        for sysname in systems:
            series = collections.defaultdict(list)
            for r in rows:
                if r["regime"] == regime and r["system"] == sysname:
                    series[int(r["iter"])].append(float(r["us"]))
            if not series:
                continue
            xs = sorted(series)
            total, ys = 0.0, []
            for i in xs:
                total += med(series[i]) / 1000.0
                ys.append(total)
            colour, _, marker = style_of(sysname)
            ax.plot(xs, ys, color=colour, linewidth=1.0, marker=marker,
                    markersize=2.2, markevery=max(1, len(xs) // 10),
                    label=SYSTEM_LABEL.get(sysname, sysname))
        ax.set_title(regime, fontsize=7)
        ax.yaxis.grid(True, zorder=0)
        ax.set_axisbelow(True)
        despine(ax)
    if args.width >= DOUBLE_COL:
        for ax in axes:
            ax.set_xlabel("iteration")
    else:
        axes[-1].set_xlabel("iteration")
    # Stacked, the label belongs to the figure rather than to the top panel:
    # on one axes constrained_layout sizes it against that panel alone and
    # clips the longer line.
    label = "cumulative ms, compilation\nincluded (lower is better)"
    if args.width >= DOUBLE_COL:
        axes[0].set_ylabel(label)
    else:
        fig.supylabel(label, fontsize=plt.rcParams["axes.labelsize"])
    legend_below(fig, axes[0], ncol=3)
    return fig, "control_cost"


FIGURES = collections.OrderedDict([
    ("micro_speedup", fig_micro_speedup),
    ("integration", fig_integration),
    ("fusion", fig_fusion),
    ("fusion_stats", fig_fusion_stats),
    ("models", fig_models),
    ("model_speedup", fig_model_speedup),
    ("dynamic", fig_dynamic),
    ("batch", fig_batch),
    ("precision", fig_precision),
    ("ablation", fig_ablation),
    ("explain", fig_explain),
    ("model_inventory", fig_model_inventory),
    ("model_provenance", fig_model_provenance),
    ("op_inventory", fig_op_inventory),
    ("deopt", fig_deopt),
    ("transition", fig_transition),
    ("motion", fig_motion),
    ("motion_retention", fig_motion_retention),
    ("decode", fig_decode),
    ("micro_baselines", fig_micro_baselines),
    ("compile_overhead", fig_compile_overhead),
    ("gap", fig_gap),
    ("correctness", fig_correctness),
    ("lazy", fig_lazy),
    ("lazy_col", fig_lazy_col),
    ("lazy_micro", fig_lazy_micro),
    ("lazy_warmup", fig_lazy_warmup),
    ("control", fig_control),
    ("control_cost", fig_control_cost),
    ("warmup", fig_warmup),
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
    p.add_argument("--iree", action="store_true",
                   help="include IREE in the figures (files get an _iree suffix); "
                        "the tables always carry it")
    args = p.parse_args(argv)

    if not args.out:
        p.error("no result directory given and $OUT is unset")
    if not os.path.isdir(args.out):
        p.error("%s is not a directory" % args.out)
    for d in args.compare:
        if not os.path.isdir(d):
            p.error("%s is not a directory" % d)
    # The categorical palette is never cycled, so a fourth machine would need a
    # hue that does not exist.
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
            args.width = PAGE_WIDTH.get(
                name, DOUBLE_COL if name in WIDE else SINGLE_COL)
        made = table[name](args.out, args)
        if made is None:
            print("skip %s (no rows in %s)" % (name, args.out), file=sys.stderr)
            continue
        fig, stem = made
        if args.iree and stem in ("models", "micro_baselines", "compile_overhead"):
            stem += "_iree"
        for ext in args.format.split(","):
            path = os.path.join(args.outdir, "%s.%s" % (stem, ext.strip()))
            fig.savefig(path)
            written.append(path)
        plt.close(fig)
        # models and micro_speedup also get a single-column variant, always,
        # regardless of --column: a teammate laying out the paper needs both
        # widths and re-running with --column single would otherwise clobber
        # the double-column default.
        if name in ("models", "micro_speedup"):
            saved_width = args.width
            args.width = PAGE_WIDTH.get(name + "_col", SINGLE_COL)
            made_col = table[name](args.out, args)
            args.width = saved_width
            if made_col is not None:
                fig_col, stem_col = made_col
                for ext in args.format.split(","):
                    path = os.path.join(args.outdir,
                                         "%s_col.%s" % (stem_col, ext.strip()))
                    fig_col.savefig(path)
                    written.append(path)
                plt.close(fig_col)
    for path in written:
        print(path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
