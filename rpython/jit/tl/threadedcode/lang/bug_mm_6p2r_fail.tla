let rec madd a b i k n acc = if k < n then madd a b i (k + 1) n (acc + a[i * n + k] * b[i * n + k]) else acc ;;
let rec mcol a b c i j n = if j < n then mcol a b (c[i * n + j] <- madd a b i 0 n 0) i (j + 1) n else c ;;
let rec mrow a b c i n = if i < n then mrow a b (mcol a b c i 0 n) (i + 1) n else c ;;
let rec asum a i n acc = if i < n then asum a (i + 1) n (acc + a[i]) else acc ;;
let rec fill a i n = if i < n then fill (a[i] <- 2) (i + 1) n else a ;;
let rec main n = let s = n * n in asum (mrow (fill (array s 0) 0 s) (array s 3) (array s 0) 0 n) 0 s 0 ;;
