let rec madd a i j k n acc = if k < n then madd a i j (k + 1) n (acc + a[i * n + k]) else acc ;;
let rec mcol a c i j n = if j < n then mcol a (c[i * n + j] <- madd a i j 0 n 0) i (j + 1) n else c ;;
let rec mrow a c i n = if i < n then mrow a (mcol a c i 0 n) (i + 1) n else c ;;
let rec asum c i n acc = if i < n then asum c (i + 1) n (acc + c[i]) else acc ;;
let rec fill a i n = if i < n then fill (a[i] <- 1) (i + 1) n else a ;;
let rec main n = let s = n * n in asum (mrow (fill (array s 0) 0 s) (array s 0) 0 n) 0 s 0 ;;
