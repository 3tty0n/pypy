# mixed list: even idx -> int, odd idx -> float.  The loop add site reads list
# data, so its operand type alternates by DATA with no source branch -> a single
# polymorphic trace.  tier3 must guard+bridge on element type; tier2 (residual
# type-agnostic add) needs one trace.
let rec fill l k n =
  if k < n then fill (aset (if k % 2 == 0 then k + 1 else tofloat (k + 1)) l k) (k + 1) n
  else l
;;

let rec loop l k n =
  if k < n then loop (aset (aref l k + aref l k) l k) (k + 1) n
  else aref l 0
;;
let rec main n =
  let l = mklist 0 n in
  let lf = fill l 0 n in
  loop lf 0 n
;;
