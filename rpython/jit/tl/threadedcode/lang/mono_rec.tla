# Identical shape/control flow, but genadd always sees int,int (monomorphic).
let rec genadd a b = a + b ;;
let rec loop i acc =
  if i < 1 then acc
  else
    let v = (if i % 2 == 0 then i else i) in
    let s = genadd v v in
    let si = (if i % 2 == 0 then s else s) in
    loop (i - 1) (acc + si)
;;
let rec main x = loop x 0 ;;
