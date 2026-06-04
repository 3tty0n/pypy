# Ackermann.  main x = ack(2, x) = 2x + 3.  Outer self-calls become a
# FRAME_RESET tail loop; the inner ack(m, n-1) is a nested CALL_ASSEMBLER.
let rec ack m n =
  if m == 0 then n + 1
  else if n == 0 then ack (m - 1) 1
  else ack (m - 1) (ack m (n - 1))
;;
let rec main x = ack 2 x ;;
