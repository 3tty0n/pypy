# Tail-recursive sum 1..n via an accumulator.  main x = sum_to x 0 ;
# returns N*(N+1)/2.  Stresses the shallow-traced `acc + n` arithmetic carried
# across the tail loop (the case the old hand-written mb_sum had to neuter to a
# closed form while the tier-1 FRAME_RESET tracing bug was unfixed).
let rec sum_to n acc =
  if n < 1 then acc
  else sum_to (n - 1) (acc + n)
;;
let rec main x = sum_to x 0 ;;
