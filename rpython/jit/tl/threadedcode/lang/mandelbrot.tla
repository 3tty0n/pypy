# Mandelbrot escape-iteration sum over an res x res grid.  Monomorphic float.
let rec iter zr zi cr ci k maxit =
  if k >= maxit then maxit
  else if ((zr * zr) + (zi * zi)) > (tofloat 4) then k
  else iter (((zr * zr) - (zi * zi)) + cr) ((((tofloat 2) * zr) * zi) + ci) cr ci (k + 1) maxit
;;
let rec row px py res maxit acc =
  if px >= res then acc
  else row (px + 1) py res maxit (acc + iter (tofloat 0) (tofloat 0) ((((tofloat px) / (tofloat res)) * (tofloat 4)) - (tofloat 2)) ((((tofloat py) / (tofloat res)) * (tofloat 4)) - (tofloat 2)) 0 maxit)
;;
let rec col py res maxit acc = if py >= res then acc else col (py + 1) res maxit (row 0 py res maxit acc) ;;
let rec main res = col 0 res 100 0 ;;
