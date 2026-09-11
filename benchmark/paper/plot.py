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
import json
import os
import re
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

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
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#8a5cd6", "#6b6b6b",
          "#c04a8a"]
POS, NEG = "#2a78d6", "#e34948"      # diverging poles, neutral midpoint is the rule line
INK, INK_2, GRID = "#0b0b0b", "#52514e", "#d9d8d4"
# Opt-in only: texture is for print and full CVD, never decoration.
HATCH = ["", "///", "...", "xxx", "\\\\", "++", "oo"]

SINGLE_COL, DOUBLE_COL = 3.25, 6.75   # MLSys column and text widths, inches
# Figures whose row labels are long enough that a single column would leave no
# room for the plot itself.  --column overrides this.
WIDE = {"micro_speedup", "fusion", "models", "ablation", "micro_baselines",
        "compile_overhead",
        "gap",
        "compare_speedup", "compare_models"}

SYSTEM_LABEL = {
    "ours": "MetaTensor",
    "torch-compile": "torch.compile",
    "torch-eager": "PyTorch eager",
    "torch-compile-dynamic": "torch.compile (dynamic)",
    "torch-compile-static": "torch.compile (static)",
    "jax": "JAX/XLA",
    "iree": "IREE",
    "torch-tensorrt": "Torch-TensorRT",
    "triton": "Triton (handwritten)",
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


ERRBAR = dict(ecolor=INK_2, elinewidth=0.7, capsize=1.4, capthick=0.7)


def spread_str(lo, hi, fmt="%.1f"):
    return (fmt + "-" + fmt) % (lo, hi) if lo is not None else "n/a"


# One colour, hatch and marker per system, whatever subset a figure shows.
SYSTEM_STYLE = {s: i for i, s in enumerate(
    ["ours", "torch-compile", "torch-eager", "jax", "iree", "torch-tensorrt",
     "triton"])}


def style_of(system):
    i = SYSTEM_STYLE.get(system, 0)
    return SERIES[i % len(SERIES)], HATCH[i % len(HATCH)], "osD^vP*"[i % 7]


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

    fig, ax = plt.subplots(figsize=(args.width, 0.16 * len(pts) + 0.75))
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
        ax.scatter(app_x, app_y, s=30, facecolors="white", edgecolors=INK,
                   linewidth=1.1, zorder=6)
    ax.axvline(1, color=INK_2, linewidth=0.8, zorder=3)
    ax.set_xscale("log", base=2)
    ax.set_xticks([0.5, 1, 2, 4, 8, 16])
    ax.xaxis.set_major_formatter(FuncFormatter(
        lambda v, _: ("%g" % v).rstrip("0").rstrip(".") + "\u00d7"))
    ax.set_yticks(list(y), labels)
    ax.set_xlim(min(lows) / 1.5, max(highs) * 1.45)
    ax.set_xlabel("speedup over torch.compile")
    ax.xaxis.grid(True, zorder=0)
    ax.set_axisbelow(True)
    despine(ax, keep=("left",))
    ax.tick_params(axis="y", length=0)
    # Direct-label the extremes only; the rest are in the table.
    for i in (0, len(pts) - 1):
        v = vals[i]
        # Anchored past the whisker, not at the bar end, so the label never
        # sits on top of the error bar.
        at = highs[i] if v >= 1 else lows[i]
        ax.annotate("%.2f\u00d7" % v, (at, i), xytext=(5 if v >= 1 else -5, 0),
                    textcoords="offset points", va="center",
                    ha="left" if v >= 1 else "right", fontsize=7, color=INK_2)
    legend_handles = [
        Line2D([], [], color=POS, lw=4, label="MetaTensor faster"),
        Line2D([], [], color=NEG, lw=4, label="torch.compile faster")]
    if app_x:
        legend_handles.append(Line2D(
            [], [], marker="o", linestyle="none", markerfacecolor="none",
            markeredgecolor=INK_2, markersize=5,
            label="same run, app-level through PyPy"))
    ax.legend(handles=legend_handles, loc="lower right",
              bbox_to_anchor=(1.0, 0.0))
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

    fig, ax = plt.subplots(figsize=(args.width, 0.16 * len(pts) + 0.75))
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
    ax.scatter(slow, list(y), s=14, color=SERIES[1],
               zorder=3, linewidth=0, label="interpreted (nojit)")
    ax.scatter(fast, list(y), s=14, color=SERIES[0],
               zorder=3, linewidth=0, label="fused (ours)")
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
                "median us per iteration over the rounds, with the observed range",
                ["benchmark", "fused", "range", "nojit", "range", "gain"],
                [[p[3], "%.1f" % p[2], spread_str(*p[4]),
                  "%.1f" % p[1], spread_str(*p[5]), "%.2f" % p[0]]
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
    systems = [s for s in ["ours", "torch-compile", "torch-eager", "jax", "iree", "torch-tensorrt"]
               if any(s in d for d in g.values())]
    models = sorted(g, key=lambda m: med(g[m].get("ours", [])) or 0)
    stats = {s: [band(g[m].get(s, [])) for m in models] for s in systems}
    vals = {s: [b[0] or 0 for b in stats[s]] for s in systems}

    shown = drawn(systems, args)
    # 3 systems keeps the original 0.26 half-height; more systems shrink to fit.
    h = 0.8 / len(shown)
    fig, ax = plt.subplots(figsize=(args.width, 0.42 * len(models) + 1.3))
    y = [i for i in range(len(models))]
    for si_, s in enumerate(shown):
        off = (si_ - (len(shown) - 1) / 2) * h
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
    ax.set_xlabel("steady-state time per iteration (\u00b5s), log scale")
    ax.xaxis.grid(True, zorder=0)
    ax.set_axisbelow(True)
    despine(ax, keep=("left",))
    ax.tick_params(axis="y", length=0)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=3,
              frameon=False, columnspacing=1.2, handlelength=1.4)
    if args.titles:
        ax.set_title("End-to-end inference")

    ov, ov_st = vals.get("ours", [0] * len(models)), stats.get("ours", [(None, None, None)] * len(models))
    tc, tc_st = vals.get("torch-compile", [0] * len(models)), stats.get("torch-compile", [(None, None, None)] * len(models))
    te = vals.get("torch-eager", [0] * len(models))
    jx = vals.get("jax", [0] * len(models))
    ir = vals.get("iree", [0] * len(models))
    trt = vals.get("torch-tensorrt", [0] * len(models))
    header = ["model", "MetaTensor", "range", "torch.compile", "range", "eager"]
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
    top = 0.0
    for i, s in enumerate(systems):
        bands = [band(g[(s, L)]) for L in lengths]
        ys = [b[0] for b in bands]
        top = max(top, max(b[2] for b in bands))
        ax.errorbar(lengths, ys,
                    yerr=err(ys, [b[1] for b in bands], [b[2] for b in bands]),
                    fmt="none", zorder=2, **ERRBAR)
        ax.plot(lengths, ys, color=SERIES[i], linewidth=1.4, marker=markers[i],
                markersize=3.4, markeredgewidth=0, label=SYSTEM_LABEL[s],
                clip_on=False, zorder=3)
    ax.set_xticks(lengths)
    ax.set_xlim(lengths[0] - 3, lengths[-1] + 3)
    ax.set_xlabel("sequence length (tokens)")
    ax.set_ylabel("median time per step (\u00b5s)")
    # Headroom so the legend has somewhere to go: the systems separate by an
    # order of magnitude, and every corner is occupied at the natural ceiling.
    ax.set_ylim(0, top * 1.2)
    ax.yaxis.grid(True, zorder=0)
    ax.set_axisbelow(True)
    despine(ax)
    # Which corner is free depends on where the recompile climb lands, which
    # depends on the data - let matplotlib measure it rather than guessing.
    ax.legend(loc="best", labelspacing=0.3, borderaxespad=0.3)
    if args.titles:
        ax.set_title("Changing sequence length")
    write_table(os.path.join(args.outdir, "dynamic.tex"),
                "median us per step over the rounds (range in brackets); "
                "recompiles over the whole sweep",
                ["system"] + [str(L) for L in lengths] + ["recompiles"],
                [[SYSTEM_LABEL[s]]
                 + ["%.0f [%s]" % (band(g[(s, L)])[0],
                                   spread_str(*band(g[(s, L)])[1:], fmt="%.0f"))
                    for L in lengths]
                 + [next((r["recompiles"] for r in rows if r["system"] == s), "")]
                 for s in systems])
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
    fig, ax = plt.subplots(figsize=(args.width, args.width * 0.62))
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
    ax.set_xlabel("steady-state time per iteration (\u00b5s)")
    ax.set_xlim(0, max(highs) * 1.18)
    ax.xaxis.grid(True, zorder=0)
    ax.set_axisbelow(True)
    despine(ax, keep=("left",))
    ax.tick_params(axis="y", length=0)
    if args.titles:
        ax.set_title("Ablations")
    write_table(os.path.join(args.outdir, "ablation.tex"),
                "median us per iteration over the rounds, with the observed range",
                ["experiment", "model", "setting", "us", "range"],
                [[e.replace("_", " "), m, v, "%.0f" % band(g[(e, m, v)])[0],
                  spread_str(*band(g[(e, m, v)])[1:], fmt="%.0f")]
                 for (e, m, v) in sorted(g, key=lambda k: (k[0], k[1], setting_key(k[2])))])
    return fig, "ablation"


def fig_micro_baselines(out, args):
    """Every backend against the same fused baseline: grouped diverging bars,
    one row per point, same log-ratio convention as fig_micro_speedup."""
    g = micro_values(read_tsv(os.path.join(out, "micro.tsv")))
    systems = ["torch-eager", "torch-compile", "jax", "iree", "triton", "torch-tensorrt"]
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
    fig, ax = plt.subplots(figsize=(args.width, 0.16 * len(pts) + 0.9))
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
    ax.set_xlabel("time relative to MetaTensor (fused), log scale")
    ax.xaxis.grid(True, zorder=0)
    ax.set_axisbelow(True)
    despine(ax, keep=("left",))
    ax.tick_params(axis="y", length=0)
    ax.legend(loc="lower right")
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
    systems = ["torch-compile", "torch-tensorrt", "jax", "iree", "triton", "ours"]

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
        off = (si_ - (len(systems_used) - 1) / 2) * h
        ys, xs = [], []
        for i, key in enumerate(keys_order):
            match = [f for k, ss, f in plot_rows if k == key and ss == s]
            if match:
                ys.append(i + off)
                xs.append(match[0])
        if not xs:
            continue
        ax.barh(ys, xs, height=h * 0.86, color=style_of(s)[0],
                linewidth=0, hatch=HATCH[si_ % len(HATCH)] if args.texture else None,
                label=SYSTEM_LABEL.get(s, s))
    ax.set_xscale("log")
    ax.set_yticks(y, [workload_label(k) for k in keys_order])
    ax.set_xlabel("first-run latency (ms), log scale")
    ax.xaxis.grid(True, zorder=0)
    ax.set_axisbelow(True)
    despine(ax, keep=("left",))
    ax.tick_params(axis="y", length=0)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=3,
              frameon=False, columnspacing=1.2, handlelength=1.4)
    if args.titles:
        ax.set_title("Compile / first-run overhead")
    return fig, "compile_overhead"


# --- machines --------------------------------------------------------------


def fig_gap(out, args):
    """Two things explain the same gap and they are in different units, so they
    get one panel each against a shared list of models: how many kernels a
    forward launches, and how much of the wall clock those kernels occupy."""
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

    fig, axes = plt.subplots(1, 2, figsize=(args.width, 0.42 * len(models) + 0.9),
                             sharey=True)
    h = 0.8 / len(systems)
    y = list(range(len(models)))
    panels = [(axes[0], "launches_per_iter", "kernel launches per forward"),
              (axes[1], "gpu_util", "GPU busy / wall time")]
    for ax, key, xlabel in panels:
        for si_, s in enumerate(systems):
            off = (si_ - (len(systems) - 1) / 2) * h
            ax.barh([v + off for v in y], [col(s, m, key) for m in models],
                    height=h * 0.88, color=SERIES[si_ % len(SERIES)],
                    linewidth=0,
                    hatch=HATCH[si_ % len(HATCH)] if args.texture else None,
                    label=SYSTEM_LABEL[s])
        ax.set_xlabel(xlabel)
        ax.xaxis.grid(True, zorder=0)
        ax.set_axisbelow(True)
        despine(ax, keep=("left",))
        ax.tick_params(axis="y", length=0)
    axes[1].set_xlim(0, 1)
    axes[0].set_yticks(y, models)
    axes[0].legend(loc="lower right", ncol=1)
    if args.titles:
        fig.suptitle("Where the gap comes from")

    header = ["model", "system", "us/iter", "launches", "kernels", "GPU us",
              "GPU util", "mix"]
    table_rows = []
    for m in models:
        for s in systems:
            r = g[m].get(s)
            if not r:
                continue
            table_rows.append([m, SYSTEM_LABEL[s], r["steady_us"],
                               r["launches_per_iter"], r["kernels"],
                               r["gpu_busy_us"], r["gpu_util"],
                               r["notes"] or "-"])
    write_table(os.path.join(args.outdir, "gap.tex"),
                "per forward: nsys launch counts and GPU busy time against the "
                "measured wall time; mix is gemm/generated/copy launches",
                header, table_rows)
    return fig, "gap"


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
                height=h * 0.86, color=SERIES[mi], linewidth=0,
                hatch=HATCH[mi] if args.texture else None, label=label)
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

    fig, ax = plt.subplots(figsize=(args.width, 0.22 * len(keys) + 0.9))
    _grouped_ratio_bars(ax, keys, per, labels, args,
                        "speedup over torch.compile", bands)
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

    fig, ax = plt.subplots(figsize=(args.width, 0.30 * len(keys) + 0.9))
    _grouped_ratio_bars(ax, keys, per, labels, args,
                        "speedup over torch.compile", bands)
    ax.set_yticks(list(range(len(keys))), keys)
    ax.legend(loc="lower right")
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
    fig, ax = plt.subplots(figsize=(args.width, 0.42 * len(bars) + 0.8))
    y = list(range(len(bars)))
    for si_, s in enumerate(systems):
        off = (si_ - (len(systems) - 1) / 2) * h
        vals = [b[1].get(s) or 0 for b in bars]
        ax.barh([yv + off for yv in y], vals, height=h * 0.88,
                color=SERIES[si_ % len(SERIES)], linewidth=0,
                hatch=HATCH[si_ % len(HATCH)] if args.texture else None,
                label=SYSTEM_LABEL.get(s, s))
    ax.set_yticks(y, [b[0] for b in bars])
    ax.set_xlabel("steady-state time per iteration (µs)")
    ax.xaxis.grid(True, zorder=0)
    ax.set_axisbelow(True)
    despine(ax, keep=("left",))
    ax.tick_params(axis="y", length=0)
    ax.legend(loc="lower right")
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


FIGURES = collections.OrderedDict([
    ("micro_speedup", fig_micro_speedup),
    ("integration", fig_integration),
    ("fusion", fig_fusion),
    ("models", fig_models),
    ("dynamic", fig_dynamic),
    ("precision", fig_precision),
    ("ablation", fig_ablation),
    ("micro_baselines", fig_micro_baselines),
    ("compile_overhead", fig_compile_overhead),
    ("gap", fig_gap),
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
            args.width = DOUBLE_COL if name in WIDE else SINGLE_COL
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
    for path in written:
        print(path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
