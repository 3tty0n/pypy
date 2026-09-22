#!/usr/bin/env python3
"""Medians and distribution-free intervals for the motion comparison.

Ten runs is too few for a normal interval to mean anything, so the interval
is the sign-test one taken straight from the order statistics: for n=10,
[x(2), x(9)] covers the median with probability 1 - 2*P(Bin(10,1/2) <= 1) =
97.9%.  No distributional assumption, and the coverage is stated rather than
implied.

    motion_stats.py RESULT_DIR
"""
import collections
import csv
import os
import sys
from math import comb


def interval(xs):
    """(low, high, coverage) for the median, from the order statistics."""
    s = sorted(xs)
    n = len(s)
    if n < 6:
        return (s[0], s[-1], 0.0) if s else (0.0, 0.0, 0.0)
    best = None
    for k in range(1, n // 2 + 1):
        cov = 1.0 - 2.0 * sum(comb(n, i) for i in range(k)) / 2.0 ** n
        if cov >= 0.95:
            best = (s[k - 1], s[n - k], cov)
    return best or (s[0], s[-1], 1.0)


def median(xs):
    s = sorted(xs)
    n = len(s)
    return s[n // 2] if n % 2 else 0.5 * (s[n // 2 - 1] + s[n // 2])


FIELDS = [("step_us", "step latency (us)", "%.1f"),
          ("at_us", "transition step (us)", "%.1f"),
          ("retained_bytes", "retained bytes", "%.0f"),
          ("launches_after", "launches / step after", "%.2f"),
          ("kernels_after", "kernels after", "%.0f")]


ARMS = ("derived", "handwritten", "direct", "drain")


def cell(rows, key, fmt):
    try:
        xs = [float(r[key]) for r in rows if r.get(key)]
    except ValueError:
        xs = []
    if not xs:
        return "n/a"
    lo, hi, cov = interval(xs)
    return ("%s [%s, %s]" % (fmt, fmt, fmt)) % (median(xs), lo, hi)


def main(argv):
    if len(argv) < 2:
        print(__doc__.strip())
        return 2
    path = os.path.join(argv[1], "motion.tsv")
    rows = list(csv.DictReader(open(path), delimiter="\t"))
    for r in rows:
        r.setdefault("cache", "cold")
        r["cache"] = r["cache"] or "cold"
    groups = collections.defaultdict(list)
    for r in rows:
        groups[(r["case"], r["cache"], int(r["steps"]))].append(r)
    for (case, cache, steps), grp in sorted(groups.items()):
        by = collections.defaultdict(list)
        for r in grp:
            by[r["arm"]].append(r)
        arms = [a for a in ARMS if a in by]
        n = min(len(by[a]) for a in arms)
        if cache == "warm" and n < 6:
            continue
        print("== %s, %s cache, %d steps: %d runs per arm, all pass=%s"
              % (case, cache, steps, n, all(r["pass"] == "1" for r in grp)))
        print("%-24s" % "" + "".join("%-30s" % a for a in arms))
        for key, label, fmt in FIELDS:
            print("%-24s" % label +
                  "".join("%-30s" % cell(by[a], key, fmt) for a in arms))
        if "derived" in by:
            d = median([float(r["step_us"]) for r in by["derived"]])
            for a in arms[1:]:
                o = median([float(r["step_us"]) for r in by[a]])
                print("derived / %s step latency: %.3fx" % (a, d / o))
        print()
    curve = collections.defaultdict(list)
    for r in rows:
        if r["cache"] == "warm":
            curve[(r["case"], int(r["steps"]), r["arm"])].append(r)
    if curve:
        cases = sorted(set(k[0] for k in curve))
        for case in cases:
            lengths = sorted(set(k[1] for k in curve if k[0] == case))
            arms = [a for a in ARMS if any(k[2] == a for k in curve
                                           if k[0] == case)]
            print("== %s, retained bytes by loop length (warm cache, medians)"
                  % case)
            print("%-8s" % "steps" + "".join("%-14s" % a for a in arms) +
                  "derived/handwritten")
            for L in lengths:
                meds = {}
                for a in arms:
                    xs = [float(r["retained_bytes"])
                          for r in curve.get((case, L, a), [])]
                    meds[a] = median(xs) if xs else float("nan")
                ratio = (meds.get("derived", 0) / meds["handwritten"]
                         if meds.get("handwritten") else float("nan"))
                print("%-8d" % L + "".join("%-14.0f" % meds[a] for a in arms)
                      + "%.2f" % ratio)
            print()
    print("intervals: distribution-free sign-test, 97.9% coverage at n=10")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
