#!/usr/bin/env python2
"""Generate skewed-polymorphism benchmarks for the tier-4 adaptive-specialization
study.

A list is filled so that element k is a *float* iff ``k % mod == 0`` and an int
otherwise -- so the loop's arithmetic sites see a float operand a controllable
``1/mod`` fraction of the time (mod=2 -> 50%, 8 -> 12.5%, 32 -> ~3%).  The hot
loop computes an M-add chain on l[k], so several arithmetic sites all carry the
same operand-type skew and arithmetic dominates the loop body.

This is exactly the regime where a binary "saw two types -> residual forever"
policy (old tier 4) over-residualises: a 3%-float site is kept as a slow residual
call for the dominant 97% int.  A frequency-aware policy inlines the dominant
type (tier-3 speed) and bridges the rare float.

Usage:  python2 gen_skew.py M MOD   -> prints .tla source
"""
import sys

def gen(m, mod):
    chain = " + ".join(["aref l k"] * (m + 1))
    return """\
# skewed-poly chain: M=%d adds/elem, float every %d-th element (~%.1f%% float)
let rec fill l k n =
  if k < n then fill (aset (if k %% %d == 0 then tofloat (k + 1) else k + 1) l k) (k + 1) n
  else l
;;
let rec loop l k n =
  if k < n then loop (aset (%s) l k) (k + 1) n
  else aref l 0
;;
let rec main n =
  let l = mklist 0 n in
  let lf = fill l 0 n in
  loop lf 0 n
;;
""" % (m, mod, 100.0 / mod, mod, chain)

if __name__ == "__main__":
    m = int(sys.argv[1])
    mod = int(sys.argv[2])
    sys.stdout.write(gen(m, mod))
