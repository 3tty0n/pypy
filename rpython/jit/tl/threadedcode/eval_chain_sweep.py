#!/usr/bin/env python3
"""Sweep the arithmetic-chain length M for int (monomorphic) vs poly (mixed-type)
workloads, measuring tier2 (residual) vs tier3 (inlined) code size, compile cost
and throughput.  Emits a CSV and a multi-panel PDF.

The story: tier3 inlines type-specialised arithmetic, so under *polymorphism* its
compiled code grows ~2x faster (it specialises both element types and bridges the
off-type) while tier2's residual, type-agnostic ops keep code independent of the
element-type mix -- they cross over at small M.  tier3 still wins throughput,
so the merit is a code-size / compile-cost tradeoff, not raw speed.

Usage: python3 eval_chain_sweep.py            # measure -> CSV (+ PDF if mpl)
       python3 eval_chain_sweep.py --plot-only # just (re)draw PDF from CSV
"""
import os, re, csv, sys, subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
BIN  = os.path.join(HERE, "targettla-c")
LANG = os.path.join(HERE, "lang")
CSV  = os.path.join(HERE, "chain_sweep.csv")
PDF  = os.path.join(HERE, "chain_sweep.pdf")
MS   = [0, 1, 2, 4, 8, 16]
SIZE, NOUT = 200, 300

def _g(text, pat, cast=float, default=0):
    m = re.search(pat, text, re.M)
    return cast(m.group(1)) if m else default

def build(m, kind):
    tla = os.path.join(LANG, "chain_%s_%d.tla" % (kind, m))
    tlc = os.path.join(LANG, "chain_%s_%d.tlc" % (kind, m))
    src = subprocess.check_output(["python2", os.path.join(HERE, "gen_chain.py"),
                                   str(m), kind], text=True)
    open(tla, "w").write(src)
    subprocess.check_call(["python2", os.path.join(HERE, "compile_tla.py"),
                           tla, "-o", tlc], stdout=subprocess.DEVNULL)
    return tlc

def measure(tlc, tier):
    log = "/tmp/sweep.jitlog"
    env = dict(os.environ, PYPYLOG="jit-summary:" + log)
    out = subprocess.run([BIN, "--tier", str(tier), tlc, str(SIZE), str(NOUT)],
                         env=env, capture_output=True, text=True, timeout=120)
    lines = [l for l in out.stdout.splitlines() if l.strip()]
    result = lines[-1] if lines else "?"
    total = sum(float(l) for l in lines[:-1] if _is_float(l))
    s = open(log).read() if os.path.exists(log) else ""
    return {
        "opt_ops":   _g(s, r"^opt ops:\s+(\d+)", int),
        "backend_ms": _g(s, r"Backend:\s+\d+\s+([0-9.]+)") * 1e3,
        "loops":     _g(s, r"Total # of loops:\s+(\d+)", int),
        "bridges":   _g(s, r"Total # of bridges:\s+(\d+)", int),
        "guards":    _g(s, r"^opt guards:\s+(\d+)", int),
        "total_s":   total,
        "result":    result,
    }

def _is_float(s):
    try: float(s); return True
    except ValueError: return False

def run_sweep():
    rows = []
    for m in MS:
        for kind in ("int", "poly"):
            tlc = build(m, kind)
            for tier in (2, 3, 4):
                r = measure(tlc, tier)
                r.update(M=m, kind=kind, tier=tier)
                rows.append(r)
                print("M=%-3d %-5s t%d  optOps=%-5d back=%.2fms  brdg=%d  tot=%.4fs  -> %s"
                      % (m, kind, tier, r["opt_ops"], r["backend_ms"],
                         r["bridges"], r["total_s"], r["result"]))
    cols = ["M", "kind", "tier", "opt_ops", "backend_ms", "loops", "bridges",
            "guards", "total_s", "result"]
    with open(CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols); w.writeheader()
        for r in rows: w.writerow({k: r[k] for k in cols})
    print("wrote", CSV)
    return rows

def plot():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rows = list(csv.DictReader(open(CSV)))
    def series(kind, tier, key):
        d = {int(r["M"]): float(r[key]) for r in rows
             if r["kind"] == kind and int(r["tier"]) == tier}
        return [d[m] for m in MS]
    sty = {("int",2):("C0","o","-","tier2 residual / int"),
           ("int",3):("C1","o","--","tier3 inlined / int"),
           ("int",4):("C2","o",":","tier4 hybrid / int"),
           ("poly",2):("C0","s","-","tier2 residual / poly"),
           ("poly",3):("C3","s","--","tier3 inlined / poly"),
           ("poly",4):("C4","s",":","tier4 hybrid / poly")}
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.5))
    panels = [("opt_ops","optimised resops (code-size proxy)"),
              ("backend_ms","backend / compile time (ms)"),
              ("total_s","total run time, %d x size %d (s)  -- lower=faster" % (NOUT, SIZE))]
    for a,(key,title) in zip(ax, panels):
        for (kind,tier),(c,mk,ls,lab) in sty.items():
            a.plot(MS, series(kind,tier,key), color=c, marker=mk, ls=ls, label=lab)
        a.set_xlabel("M  (arithmetic ops per element)"); a.set_title(title)
        a.grid(alpha=.3)
    ax[0].axvline(4, color="gray", ls=":", lw=1)
    ax[0].annotate("poly crossover\n(tier3 code > tier2)", (4, series("poly",3,"opt_ops")[3]),
                   xytext=(6,300), arrowprops=dict(arrowstyle="->", color="gray"), fontsize=8)
    ax[0].legend(fontsize=8, loc="upper left")
    fig.suptitle("tier2 residual vs tier3 inlined vs tier4 hybrid (selective inliner).  "
                 "Left: tier3-poly code GROWS, tier4-poly stays smallest (residualises poly sites, "
                 "no extra bridge).  Right: tier4-int keeps tier3 speed; tier4-poly pays tier2 speed.",
                 fontsize=10)
    fig.tight_layout(rect=[0,0,1,0.96]); fig.savefig(PDF)
    print("wrote", PDF)

if __name__ == "__main__":
    if "--plot-only" not in sys.argv:
        run_sweep()
    try:
        plot()
    except ImportError as e:
        print("(matplotlib unavailable: %s -- CSV written, skip PDF)" % e)
