# Takeuchi's tarai (the original; tak's sibling).  The hand-written
# lang/tarai.tla.py hardcodes tarai(5, 3, 1) and ignores the CLI x, so this
# definition does the same (main's parameter is unused) to stay faithful.
let rec tarai x y z =
  if x <= y then y
  else tarai (tarai (x - 1) y z) (tarai (y - 1) z x) (tarai (z - 1) x y)
;;
let rec main x = tarai 5 3 1 ;;
