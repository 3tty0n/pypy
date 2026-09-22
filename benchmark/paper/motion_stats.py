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


def main(argv):
    if len(argv) < 2:
        print(__doc__.strip())
        return 2
    path = os.path.join(argv[1], "motion.tsv")
    rows = list(csv.DictReader(open(path), delimiter="\t"))
    by = collections.defaultdict(list)
    for r in rows:
        by[r["arm"]].append(r)
    arms = [a for a in ("derived", "handwritten") if a in by]
    n = min(len(by[a]) for a in arms) if arms else 0
    print("motion: %d runs per arm, %d arms, all pass=%s\n"
          % (n, len(arms), all(r["pass"] == "1" for r in rows)))
    print("%-22s %-28s %-28s" % ("", arms[0] if arms else "",
                                 arms[1] if len(arms) > 1 else ""))
    for key, label, fmt in FIELDS:
        cells = []
        for a in arms:
            try:
                xs = [float(r[key]) for r in by[a] if r.get(key)]
            except ValueError:
                xs = []
            if not xs:
                cells.append("n/a")
                continue
            lo, hi, cov = interval(xs)
            cells.append(("%s  [%s, %s]" % (fmt, fmt, fmt))
                         % (median(xs), lo, hi))
        print("%-22s %-28s %-28s" % (label, cells[0],
                                     cells[1] if len(cells) > 1 else ""))
    if arms:
        xs = [float(r["step_us"]) for r in by[arms[0]]]
        ys = [float(r["step_us"]) for r in by[arms[-1]]]
        lo, hi, cov = interval(xs)
        print("\nintervals are distribution-free, coverage %.1f%%" % (cov * 100))
        print("derived / handwritten step latency: %.3fx"
              % (median(xs) / median(ys)) if median(ys) else "")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
