# skewed-poly chain: M=8 adds/elem, float every 16-th element (~6.2% float)
let rec fill l k n =
  if k < n then fill (aset (if k % 16 == 0 then tofloat (k + 1) else k + 1) l k) (k + 1) n
  else l
;;
let rec loop l k n =
  if k < n then loop (aset (aref l k + aref l k + aref l k + aref l k + aref l k + aref l k + aref l k + aref l k + aref l k) l k) (k + 1) n
  else aref l 0
;;
let rec main n =
  let l = mklist 0 n in
  let lf = fill l 0 n in
  loop lf 0 n
;;
