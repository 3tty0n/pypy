# Count down to zero (tail-recursive loop), main x = loop x -> 0.
let rec loop n =
  if n < 1 then 0
  else loop (n - 1)
;;
let rec main x = loop x ;;
