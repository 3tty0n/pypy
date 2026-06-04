# Takeuchi function.  main n = tak(n, 0, 1).  Outer self-call is a FRAME_RESET
# tail loop (arity 3); the three argument tak(...) calls are nested
# CALL_ASSEMBLER recursions.
let rec tak x y z =
  if y >= x then z
  else tak (tak (x - 1) y z) (tak (y - 1) z x) (tak (z - 1) x y)
;;
let rec main n = tak n 0 1 ;;
