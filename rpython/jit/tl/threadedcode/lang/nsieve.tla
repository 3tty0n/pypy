let rec mark f i k n = if k >= n then f else mark (f[k] <- 0) i (k + i) n ;;
let rec sieve f i n acc =
  if i >= n then acc
  else if f[i] == 1 then sieve (mark f i (i + i) n) (i + 1) n (acc + 1)
  else sieve f (i + 1) n acc
;;
let rec fill1 a i n = if i < n then fill1 (a[i] <- 1) (i + 1) n else a ;;
let rec main n = sieve (fill1 (array n 0) 0 n) 2 n 0 ;;
