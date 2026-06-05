let rec madd a b i j k n acc =
  if k < n then madd a b i j (k + 1) n (acc + a[i * n + k] * b[k * n + j]) else acc ;;
let rec mcol a b c i j n z =
  if j < n then mcol a b (c[i * n + j] <- madd a b i j 0 n z) i (j + 1) n z else c ;;
let rec mrow a b c i n z = if i < n then mrow a b (mcol a b c i 0 n z) (i + 1) n z else c ;;
let rec asum a i n acc = if i < n then asum a (i + 1) n (acc + a[i]) else acc ;;
let rec fillI a i n = if i < n then fillI (a[i] <- i % 7 + 1) (i + 1) n else a ;;
let rec fillF a i n = if i < n then fillF (a[i] <- tofloat (i % 7 + 1)) (i + 1) n else a ;;
let rec main n = let s = n * n in asum (mrow (fillI (array s 0) 0 s) (fillI (array s 0) 0 s) (array s 0) 0 n 0) 0 s 0 ;;
