# Euclid's gcd, main x = gcd(x, 48).  The self-call gcd b (a % b) is in tail
# position, so it compiles to the FRAME_RESET + JUMP in-frame loop.
let rec gcd a b =
  if b == 0 then a
  else gcd b (a % b)
;;
let rec main x = gcd x 48 ;;
