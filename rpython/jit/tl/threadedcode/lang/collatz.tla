# Sum of Collatz step counts for 1..n.  main n = sum_steps 1 n 0.
let rec steps n acc =
  if n < 2 then acc
  else if n % 2 == 0 then steps (n / 2) (acc + 1)
  else steps (3 * n + 1) (acc + 1)
;;
let rec sum_steps i n acc =
  if i > n then acc
  else sum_steps (i + 1) n (acc + steps i 0)
;;
let rec main n = sum_steps 1 n 0 ;;
