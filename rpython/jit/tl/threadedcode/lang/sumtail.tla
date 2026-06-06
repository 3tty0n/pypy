# Tail-recursive sum 1..n.  main x = sum_to x 0 ; returns N*(N+1)/2.
let rec sum_to n acc =
  if n < 1 then acc
  else sum_to (n - 1) (acc + n)
;;
let rec main x = sum_to x 0 ;;
