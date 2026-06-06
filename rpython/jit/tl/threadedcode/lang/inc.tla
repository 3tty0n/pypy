# Tail-recursive counter: the accumulator is incremented by 1 each step.
# main x = inc x 0 ; returns N.  Like mb_sum but with a constant `+ 1` instead of
# `+ n`, separating the carried-arithmetic cost from the loop-variable read.
let rec inc n acc =
  if n < 1 then acc
  else inc (n - 1) (acc + 1)
;;
let rec main x = inc x 0 ;;
