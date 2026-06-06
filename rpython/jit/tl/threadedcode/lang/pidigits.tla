# Pidigits-style spigot.  The real Benchmark Game version depends on bignums;
# this bounded Rabinowitz-Wagon spigot keeps the carry/update shape and returns
# a checksum of n carry states for sizes that fit the translated ints.
let rec fill a i n =
  if i >= n then a else fill (a[i] <- 2) (i + 1) n
;;
let rec inner a i carry =
  if i < 0 then carry
  else
    let x = a[i] * 10 + carry in
    let k = i + 1 in
    let d = 2 * k - 1 in
    let a = a[i] <- (x % d) in
    inner a (i - 1) ((x / d) * k)
;;
let rec digit_loop a k n len carry acc =
  if k >= n then acc
  else
    let q = inner a (len - 1) 0 in
    digit_loop a (k + 1) n len (q % 10) (acc + q * (k + 1))
;;
let rec main n =
  let len = n * 10 / 3 + 1 in
  digit_loop (fill (array len 0) 0 len) 0 n len 0 0
;;
