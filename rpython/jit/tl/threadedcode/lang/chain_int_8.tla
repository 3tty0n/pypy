# arithmetic-chain benchmark: M=8 adds per element, kind=int
let rec fill l k n =
  if k < n then fill (aset (if k % 2 == 0 then k + 1 else k + 1) l k) (k + 1) n
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
