# Fannkuch-redux kernel.  Enumerates permutations by recursive swapping, counts
# prefix reversals on a copied permutation, and returns the maximum flip count.
let rec fill a i n = if i >= n then a else fill (a[i] <- i) (i + 1) n ;;
let rec copy src dst i n = if i >= n then dst else copy src (dst[i] <- src[i]) (i + 1) n ;;
let rec swap a i j = let t = a[i] in (a[i] <- a[j])[j] <- t ;;
let rec flip a i j =
  if i >= j then a else flip (swap a i j) (i + 1) (j - 1)
;;
let rec flips a acc =
  let k = a[0] in
  if k == 0 then acc else flips (flip a 0 k) (acc + 1)
;;
let rec leaf a n = flips (copy a (array n 0) 0 n) 0 ;;
let rec max2 a b = if a < b then b else a ;;
let rec perm_i a pos i n best =
  if i >= n then best
  else
    let b = perm (swap a pos i) (pos + 1) n best in
    perm_i (swap a pos i) pos (i + 1) n b
;;
let rec perm a pos n best =
  if pos >= n then max2 best (leaf a n)
  else perm_i a pos pos n best
;;
let rec main n = perm (fill (array n 0) 0 n) 0 n 0 ;;
