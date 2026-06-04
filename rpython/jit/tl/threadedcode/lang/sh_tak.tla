# Takeuchi tak.  main n = tak n 0 1.
let rec tak x y z =
  if y >= x then z
  else tak (tak (x - 1) y z) (tak (y - 1) z x) (tak (z - 1) x y)
;;
let rec main n = tak n 0 1 ;;
