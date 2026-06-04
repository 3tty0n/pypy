# Tail-recursive countdown to zero.  main x = down x ; returns 0.
let rec down n =
  if n < 1 then 0
  else down (n - 1)
;;
let rec main x = down x ;;
