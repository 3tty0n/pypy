# sum 1..n via a tail-recursive accumulator.  main x = sum_to(x, 0).
# The self-call is in tail position -> FRAME_RESET + JUMP in-frame loop
# (the same JIT-friendly shape as the hand-written lang/sum-tail.tla.py).
let rec sum_to n acc =
  if n < 1 then acc
  else sum_to (n - 1) (acc + n)
;;
let rec main x = sum_to x 0 ;;
