# Fibonacci (tree recursion).  main x = fib x.
let rec fib n =
  if n < 2 then n
  else fib (n - 1) + fib (n - 2)
;;
let rec main x = fib x ;;
