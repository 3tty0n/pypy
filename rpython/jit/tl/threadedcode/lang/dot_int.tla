let rec dot a b i n acc = if i < n then dot a b (i + 1) n (acc + a[i] * b[i]) else acc ;;
let rec fillI a i n = if i < n then fillI (a[i] <- i % 9 + 1) (i + 1) n else a ;;
let rec fillF a i n = if i < n then fillF (a[i] <- tofloat (i % 9 + 1)) (i + 1) n else a ;;
let rec drive v k iters m acc = if k >= iters then acc else drive v (k + 1) iters m (acc + dot v v 0 m 0) ;;
let rec main m = drive (fillI (array m 0) 0 m) 0 600 m 0 ;;
