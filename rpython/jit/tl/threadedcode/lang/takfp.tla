# Takeuchi function on floats (shootout: takfp).  Same control structure as the
# integer `tak`, but exercises float arithmetic + comparison instead of int.
let rec tak x y z =
  if y < x then
    tak (tak (x - tofloat 1) y z) (tak (y - tofloat 1) z x) (tak (z - tofloat 1) x y)
  else z
;;
let rec main n =
  toint (tak (tofloat (n * 3)) (tofloat (n * 2)) (tofloat n))
;;
