# Minimal tail-recursive countdown.  main x = countdown x ; returns 0.
# No accumulator and no arithmetic on the returned value, so it isolates the
# FRAME_RESET / tail-call plumbing from shallow-traced arithmetic.  The self
# call is in tail position, so it lowers to the in-frame FRAME_RESET + JUMP loop.
let rec countdown n =
  if n > 1 then countdown (n - 1)
  else 0
;;
let rec main x = countdown x ;;
