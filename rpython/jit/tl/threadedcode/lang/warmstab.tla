# warmstab: adaptive-specialization / warmup-stability stress.  A tail-recursive
# sweep whose shared leaf `genadd` is fed int operands in the first half and
# float operands in the second, so a tier-4 per-site operand-type (poly)
# decision faces a genuine within-run phase change -- this stresses how fast the
# adaptive policy converges (the warmup curve) rather than steady-state
# throughput.  Array-free, and only tail/leaf recursion, so it avoids both the
# array path and the deep non-tail (tak/tarai) recursion classes.
# main m  ->  sum_{k=0..m-1} 2k  =  m*(m-1).
let rec genadd a b = a + b ;;
let rec work k m acc =
  if k >= m then acc
  else if k < m / 2 then work (k + 1) m (acc + genadd k k)
  else work (k + 1) m (acc + toint (genadd (tofloat k) (tofloat k)))
;;
let rec main m = work 0 m 0 ;;
