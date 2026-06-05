let rec swap a i j = let t = a[i] in (a[i] <- a[j])[j] <- t ;;
let rec siftdown a i n =
  if (2 * i + 1) >= n then a
  else
    let l = 2 * i + 1 in
    let c = (if (l + 1) < n then (if a[l] < a[l + 1] then (l + 1) else l) else l) in
    if a[i] < a[c] then siftdown (swap a i c) c n else a
;;
let rec heapify a i n = if i < 0 then a else heapify (siftdown a i n) (i - 1) n ;;
let rec sortloop a e = if e < 1 then a else sortloop (siftdown (swap a 0 e) 0 e) (e - 1) ;;
let rec hsort a n = sortloop (heapify a (n / 2 - 1) n) (n - 1) ;;
let rec asum a i n acc = if i < n then asum a (i + 1) n (acc + a[i]) else acc ;;
let rec fillI a i n = if i < n then fillI (a[i] <- (i * 7919 + 13) % n) (i + 1) n else a ;;
let rec fillF a i n = if i < n then fillF (a[i] <- tofloat ((i * 7919 + 13) % n)) (i + 1) n else a ;;
let rec main n =
  let si = asum (hsort (fillI (array n 0) 0 n) n) 0 n 0 in
  let sf = asum (hsort (fillF (array n 0) 0 n) n) 0 n (tofloat 0) in
  si + toint sf
;;
