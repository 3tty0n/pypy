# Count nodes of a full binary tree of depth d.  main x = tree x.
let rec tree d =
  if d < 1 then 1
  else 1 + tree (d - 1) + tree (d - 1)
;;
let rec main x = tree x ;;
