# genadd is a single arithmetic site called with float,float (even i) and
# int,int (odd i) -> type-polymorphic.  loop recurses (depth <= x), so keep x
# small and rely on many outer runs to reach JIT hotness.  Result = x*(x+1).
let rec genadd a b = a + b ;;
let rec loop i acc =
  if i < 1 then acc
  else
    let v = (if i % 2 == 0 then tofloat i else i) in
    let s = genadd v v in
    let si = (if i % 2 == 0 then toint s else s) in
    loop (i - 1) (acc + si)
;;
let rec main x = loop x 0 ;;
