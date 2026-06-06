# Count primes in 2..n by trial division.  main n = count 2 n 0.
let rec divides n d =
  if d * d > n then 0
  else if n % d == 0 then 1
  else divides n (d + 1)
;;
let rec count i n acc =
  if i > n then acc
  else if divides i 2 == 0 then count (i + 1) n (acc + 1)
  else count (i + 1) n acc
;;
let rec main n = count 2 n 0 ;;
