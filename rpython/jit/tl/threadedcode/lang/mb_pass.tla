# Tail-recursive passthrough -- the accumulator k is never modified in the loop.
# main x = pass x 42 ; returns 42.  A two-argument tail loop with no arithmetic
# on the carried value, isolating the frame reshape from any data dependency.
let rec pass n k =
  if n >= 1 then pass (n - 1) k
  else k
;;
let rec main x = pass x 42 ;;
