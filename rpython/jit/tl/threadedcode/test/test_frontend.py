"""Tests for the TLA frontend (frontend.py): compile a Python subset to TLA
bytecode and check it runs correctly under the tier-0 interpreter."""
import py

from rpython.jit.tl.threadedcode.frontend import compile_source, CompileError
from rpython.jit.tl.threadedcode.bytecode import Bytecode
from rpython.jit.tl.threadedcode import tla


def run0(src, x):
    """Compile `src` and run main(x) under the pure (tier-0) interpreter."""
    code = compile_source(src)
    bc = Bytecode(''.join(chr(c) for c in code))
    res = tla.run(bc, tla.W_IntObject(x), tier=0)
    return res.intvalue


FIB = """
def fib(n):
    if n < 2:
        return n
    return fib(n - 1) + fib(n - 2)

def main(x):
    return fib(x)
"""

FACT = """
def fact(n):
    if n < 1:
        return 1
    return n * fact(n - 1)

def main(x):
    return fact(x)
"""

SUM_TAIL = """
def sum_to(n, acc):
    if n < 1:
        return acc
    return sum_to(n - 1, acc + n)

def main(x):
    return sum_to(x, 0)
"""

GCD = """
def gcd(a, b):
    if b == 0:
        return a
    return gcd(b, a % b)

def main(x):
    return gcd(x, 48)
"""

ACK = """
def ack(m, n):
    if m == 0:
        return n + 1
    if n == 0:
        return ack(m - 1, 1)
    return ack(m - 1, ack(m, n - 1))

def main(x):
    return ack(2, x)
"""

TAK = """
def tak(x, y, z):
    if y >= x:
        return z
    return tak(tak(x - 1, y, z), tak(y - 1, z, x), tak(z - 1, x, y))

def main(n):
    return tak(n, 0, 1)
"""


def test_fib():
    assert run0(FIB, 10) == 55
    assert run0(FIB, 1) == 1
    assert run0(FIB, 0) == 0


def test_fact():
    assert run0(FACT, 6) == 720
    assert run0(FACT, 0) == 1


def test_sum_tail():
    assert run0(SUM_TAIL, 100) == 5050
    assert run0(SUM_TAIL, 1) == 1


def test_gcd():
    assert run0(GCD, 36) == 12
    assert run0(GCD, 17) == 1


def test_ack():
    assert run0(ACK, 3) == 9       # ack(2,3) = 9


def test_tak():
    assert run0(TAK, 6) == 0       # tak(6,0,1) == 0 (matches CPython reference)


def test_arithmetic_and_precedence():
    src = """
def main(x):
    return 2 + 3 * 4 - 10 / 2
"""
    assert run0(src, 0) == 2 + 3 * 4 - 10 / 2     # 9


def test_comparisons():
    # exercise every comparison operator through the strict/non-strict bridge
    def cmp_prog(op):
        return ("def main(x):\n"
                "    if x %s 5:\n"
                "        return 1\n"
                "    return 0\n" % op)
    assert run0(cmp_prog("<"), 4) == 1 and run0(cmp_prog("<"), 5) == 0
    assert run0(cmp_prog("<="), 5) == 1 and run0(cmp_prog("<="), 6) == 0
    assert run0(cmp_prog(">"), 6) == 1 and run0(cmp_prog(">"), 5) == 0
    assert run0(cmp_prog(">="), 5) == 1 and run0(cmp_prog(">="), 4) == 0
    assert run0(cmp_prog("=="), 5) == 1 and run0(cmp_prog("=="), 4) == 0
    assert run0(cmp_prog("!="), 4) == 1 and run0(cmp_prog("!="), 5) == 0


def test_errors():
    py.test.raises(CompileError, compile_source, "def f(n):\n return n\n")  # no main
    py.test.raises(CompileError, compile_source,
                   "def main(x):\n return g(x)\n")                          # unknown call
