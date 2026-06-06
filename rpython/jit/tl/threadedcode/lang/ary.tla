# Array access (shootout: ary).  x[i] = i+1; then for `iters` rounds add the
# reversed x into accumulator y; return y[0].  Array-load/store heavy.
let rec fillx x i n = if i >= n then x else fillx (x[i] <- i + 1) (i + 1) n ;;
let rec addrev x y i n =
  if i >= n then y else addrev x (y[i] <- aref y i + aref x (n - 1 - i)) (i + 1) n
;;
let rec drive x y k iters n =
  if k >= iters then y else drive x (addrev x y 0 n) (k + 1) iters n
;;
let rec main n =
  let x = fillx (array n 0) 0 n in
  let y = drive x (array n 0) 0 1000 n in
  aref y 0
;;
