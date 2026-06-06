# Harmonic series partial sum (shootout: partial-sums / harmonic): sum_{k=1..n}
# 1/k, a float reduction over a tail loop, scaled to an int so the result is
# comparable across tiers.
let rec hsum k n acc =
  if k > n then acc
  else hsum (k + 1) n (acc + (tofloat 1 / tofloat k))
;;
let rec main n = toint ((hsum 1 n (tofloat 0)) * (tofloat 1000000)) ;;
