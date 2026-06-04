# Monomorphic FLOAT: genadd always sees float,float.  facc stays float.
let rec genadd a b = a + b ;;
let rec loop i facc =
  if i < 1 then facc
  else
    let s = genadd (tofloat i) (tofloat i) in
    loop (i - 1) (facc + s)
;;
let rec main x = loop x (tofloat 0) ;;
