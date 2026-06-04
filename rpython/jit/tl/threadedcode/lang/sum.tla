# sum 1..n  (linear recursion, via CALL_ASSEMBLER).  main x = sum x.
let rec sum n =
  if n < 1 then 0
  else n + sum (n - 1)
;;
let rec main x = sum x ;;
