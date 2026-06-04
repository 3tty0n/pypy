# sum of squares 1..n  (linear recursion, via CALL_ASSEMBLER).  main x = sqsum x.
let rec sqsum n =
  if n < 1 then 0
  else n * n + sqsum (n - 1)
;;
let rec main x = sqsum x ;;
