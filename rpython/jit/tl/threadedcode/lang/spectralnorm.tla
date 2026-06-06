# Spectral norm (shootout: spectralnorm).  Approximates the largest singular
# value of the infinite matrix A(i,j) = 1/((i+j)(i+j+1)/2 + i + 1) by 10 power
# iterations of A^T A on the all-ones vector, returning sqrt(uv/vv).  Exercises
# float arrays, a 1/x kernel, and sqrt.  Result ~ 1.274224 (scaled by 1e6).
let rec aij i j = tofloat 1 / tofloat (((i + j) * (i + j + 1)) / 2 + i + 1) ;;

let rec ones u i n = if i >= n then u else ones (u[i] <- tofloat 1) (i + 1) n ;;

let rec au_row u i j n acc =
  if j >= n then acc else au_row u i (j + 1) n (acc + (aij i j * aref u j))
;;
let rec au u out i n =
  if i >= n then out else au u (out[i] <- au_row u i 0 n (tofloat 0)) (i + 1) n
;;
let rec atu_row u i j n acc =
  if j >= n then acc else atu_row u i (j + 1) n (acc + (aij j i * aref u j))
;;
let rec atu u out i n =
  if i >= n then out else atu u (out[i] <- atu_row u i 0 n (tofloat 0)) (i + 1) n
;;
# out = A^T A u, using s as scratch (does not alias u or out).
let rec atau u s out n = atu (au u s 0 n) out 0 n ;;

let rec dot2 u v i n acc =
  if i >= n then acc else dot2 u v (i + 1) n (acc + (aref u i * aref v i))
;;

# 10 power iterations; at the end u and v are the final pair, return the norm.
let rec power u v s k n =
  if k >= 10 then
    sqrt ((dot2 u v 0 n (tofloat 0)) / (dot2 v v 0 n (tofloat 0)))
  else
    let v2 = atau u s v n in
    let u2 = atau v2 s u n in
    power u2 v2 s (k + 1) n
;;

let rec main n =
  let u = ones (array n (tofloat 0)) 0 n in
  let v = array n (tofloat 0) in
  let s = array n (tofloat 0) in
  toint ((power u v s 0 n) * (tofloat 1000000))
;;
