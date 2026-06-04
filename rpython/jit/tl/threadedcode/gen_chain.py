#!/usr/bin/env python2
"""Generate arithmetic-chain benchmarks to sweep tier2-vs-tier3 code growth.

The hot loop reads list element e = l[k] and computes a left-assoc chain of M
additions  e + e + ... + e  (M adds, all operands the same element so the chain
is monomorphic *per iteration* but its type alternates by DATA across iterations
in the _poly variant).  Arithmetic dominates the loop body, so the per-op
tier2(residual)-vs-tier3(inlined,type-specialised) difference is exposed:

  poly variant: tier3 must specialise+guard the whole chain per element type and
                bridge the off-type -> ~2*M inlined ops + M guards.
  poly variant: tier2's residual adds carry no type guard -> ONE loop, M residual
                calls, no type bridge -> code independent of element-type mix.

Usage:  python2 gen_chain.py M kind   (kind = int | poly)  -> prints .tla source
"""
import sys

def gen(m, kind):
    # element value at index k: even -> int(k+1); odd -> int (kind=int) or
    # float (kind=poly).  The loop add-chain reads l[k] (m+1 times).
    if kind == "poly":
        fill_odd = "tofloat (k + 1)"
    else:
        fill_odd = "k + 1"
    chain = " + ".join(["aref l k"] * (m + 1))
    return """\
# arithmetic-chain benchmark: M=%d adds per element, kind=%s
let rec fill l k n =
  if k < n then fill (aset (if k %% 2 == 0 then k + 1 else %s) l k) (k + 1) n
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
""" % (m, kind, fill_odd, chain)

if __name__ == "__main__":
    m = int(sys.argv[1])
    kind = sys.argv[2]
    sys.stdout.write(gen(m, kind))
