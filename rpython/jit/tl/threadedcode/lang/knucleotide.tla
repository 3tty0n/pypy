# K-nucleotide over an encoded DNA stream.  Instead of printing frequency
# tables, count 6-mers into 4096 buckets and return a weighted checksum.
let rec gen a i n seed =
  if i >= n then a
  else
    let seed = (seed * 3877 + 29573) % 139968 in
    gen (a[i] <- (seed % 4)) (i + 1) n seed
;;
let rec code6 s i k acc =
  if k >= 6 then acc else code6 s i (k + 1) ((acc * 4) + s[i + k])
;;
let rec count s tab i n =
  if i + 6 >= n then tab
  else
    let c = code6 s i 0 0 in
    count s (tab[c] <- (tab[c] + 1)) (i + 1) n
;;
let rec sumtab tab i n acc =
  if i >= n then acc else sumtab tab (i + 1) n (acc + tab[i] * (i + 1))
;;
let rec main n =
  let s = gen (array n 0) 0 n 42 in
  sumtab (count s (array 4096 0) 0 n) 0 4096 0
;;
