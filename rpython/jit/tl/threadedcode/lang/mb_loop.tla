# Countdown to zero.  main x = loop x ; returns 0.
# The hand-written lang/mb_loop.tla.py is a *pure single-frame* loop (no
# CALL_ASSEMBLER, no FRAME_RESET) -- a shape the surface language cannot express:
# `main` is the entry point the runtime invokes directly (it cannot be reached
# through CALL_ASSEMBLER, so a recursive `main` is rejected), and every tail
# self-call lowers to a FRAME_RESET + JUMP in-frame loop.  This definition is
# therefore structurally the same as mb_count (main calls a tail-recursive
# helper) and computes the same value (0); it is kept for parity of the lang set.
let rec loop n =
  if n < 1 then n
  else loop (n - 1)
;;
let rec main x = loop x ;;
