# Euclid's gcd.  main x = gcd x 48.
let rec gcd a b =
  if b == 0 then a
  else gcd b (a % b)
;;
let rec main x = gcd x 48 ;;
