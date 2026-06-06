# Reverse-complement over encoded DNA.  A=0,C=1,G=2,T=3, complement is 3-x.
# Returns a checksum of the reversed complemented sequence.
let rec gen a i n seed =
  if i >= n then a
  else
    let seed = (seed * 3877 + 29573) % 139968 in
    gen (a[i] <- (seed % 4)) (i + 1) n seed
;;
let rec rev s out i n =
  if i >= n then out else rev s (out[i] <- (3 - s[n - 1 - i])) (i + 1) n
;;
let rec checksum a i n acc =
  if i >= n then acc else checksum a (i + 1) n (acc + a[i] * (i + 1))
;;
let rec main n =
  let s = gen (array n 0) 0 n 42 in
  checksum (rev s (array n 0) 0 n) 0 n 0
;;
