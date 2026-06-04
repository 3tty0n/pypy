# fib(n) via a tail-recursive two-accumulator loop.  main x = fibt(x, 0, 1).
# The self-call is in tail position -> FRAME_RESET + JUMP in-frame loop.
let rec fibt n a b =
  if n < 1 then a
  else fibt (n - 1) b (a + b)
;;
let rec main x = fibt x 0 1 ;;
