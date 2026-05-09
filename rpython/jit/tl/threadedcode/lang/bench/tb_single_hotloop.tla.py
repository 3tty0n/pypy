# Microbench: minimal backward JUMP + JUMP_IF (tier-2 threaded merge).
# Input: W_IntObject(n), n >= 0. Decrements until n == 0; returns 0.
# Note: LT in this VM is implemented as x <= y; use EQ for a strict zero test.
# Byte layout must match JUMP_IF / JUMP targets (byte PCs).
from rpython.jit.tl.threadedcode import tla

# pc 0-3:  n == 0 ?
# pc 4-5:  JUMP_IF target -> EXIT at 11
# pc 6-10: body then jump to 0
code = [
    tla.DUP,
    tla.CONST_INT, 0,
    tla.EQ,
    tla.JUMP_IF, 11,
    tla.CONST_INT, 1,
    tla.SUB,
    tla.JUMP, 0,
    tla.EXIT,
]
