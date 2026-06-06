# Regex-redux-style encoded scan.  TLA has no regex engine or strings; this
# keeps the benchmark's repeated sequence scanning shape over encoded DNA and
# counts a family of ambiguous patterns.
let rec gen a i n seed =
  if i >= n then a
  else
    let seed = (seed * 3877 + 29573) % 139968 in
    gen (a[i] <- (seed % 4)) (i + 1) n seed
;;
let rec match3 s i a b c =
  if s[i] != a then 0
  else if s[i + 1] != b then 0
  else if s[i + 2] != c then 0
  else 1
;;
let rec scan s i n p acc =
  if i + 3 >= n then acc
  else scan s (i + 1) n p (acc + match3 s i (p % 4) ((p / 4) % 4) ((p / 16) % 4))
;;
let rec patterns s p n acc =
  if p >= 16 then acc else patterns s (p + 1) n (acc + scan s 0 n p 0 * (p + 1))
;;
let rec main n = let s = gen (array n 0) 0 n 42 in patterns s 0 n 0 ;;
