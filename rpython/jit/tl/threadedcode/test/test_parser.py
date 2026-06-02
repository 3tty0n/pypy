"""Tests for the native TLA frontend (parser.py): parse the OCaml-flavoured
surface language into an AST and compile it to TLA bytecode that runs correctly
under the tier-0 interpreter."""
import py

from rpython.jit.tl.threadedcode.parser import (
    parse, compile_source, CompileError,
    Program, Function, If, LetIn, BinOp, FunApp, Variable, ConstInt)
from rpython.jit.tl.threadedcode.bytecode import Bytecode
from rpython.jit.tl.threadedcode import tla


def run0(src, x):
    """Compile `src` and run main(x) under the pure (tier-0) interpreter."""
    code = compile_source(src)
    bc = Bytecode(''.join(chr(c) for c in code))
    res = tla.run(bc, tla.W_IntObject(x), tier=0)
    return res.intvalue


# --- source programs --------------------------------------------------------

FIB = """
let rec fib n =
  if n < 2 then n
  else fib (n - 1) + fib (n - 2)
;;
let rec main x = fib x ;;
"""

FACT = """
let rec fact n =
  if n < 1 then 1
  else n * fact (n - 1)
;;
let rec main x = fact x ;;
"""

SUM_TAIL = """
let rec sum_to n acc =
  if n < 1 then acc
  else sum_to (n - 1) (acc + n)
;;
let rec main x = sum_to x 0 ;;
"""

GCD = """
let rec gcd a b =
  if b == 0 then a
  else gcd b (a % b)
;;
let rec main x = gcd x 48 ;;
"""

ACK = """
let rec ack m n =
  if m == 0 then n + 1
  else if n == 0 then ack (m - 1) 1
  else ack (m - 1) (ack m (n - 1))
;;
let rec main x = ack 2 x ;;
"""

TAK = """
let rec tak x y z =
  if y >= x then z
  else tak (tak (x - 1) y z) (tak (y - 1) z x) (tak (z - 1) x y)
;;
let rec main n = tak n 0 1 ;;
"""


# --- parsing (AST shape) ----------------------------------------------------

def test_parse_fib():
    prog = parse(FIB)
    assert isinstance(prog, Program)
    assert [f.name for f in prog.functions] == ['fib', 'main']
    fib = prog.functions[0]
    assert fib.params == ['n']
    assert fib.body == If(
        BinOp('<', Variable('n'), ConstInt(2)),
        Variable('n'),
        BinOp('+',
              FunApp('fib', [BinOp('-', Variable('n'), ConstInt(1))]),
              FunApp('fib', [BinOp('-', Variable('n'), ConstInt(2))])))


def test_parse_precedence():
    # `*` binds tighter than `+`
    prog = parse("let rec main x = 2 + 3 * 4 ;;")
    assert prog.functions[0].body == BinOp(
        '+', ConstInt(2), BinOp('*', ConstInt(3), ConstInt(4)))


def test_parse_left_assoc():
    # `-` is left-associative: 10 - 3 - 2 == (10 - 3) - 2
    prog = parse("let rec main x = 10 - 3 - 2 ;;")
    assert prog.functions[0].body == BinOp(
        '-', BinOp('-', ConstInt(10), ConstInt(3)), ConstInt(2))


def test_parse_let_in():
    prog = parse("let rec main x = let y = x + 1 in y * 2 ;;")
    assert prog.functions[0].body == LetIn(
        'y', BinOp('+', Variable('x'), ConstInt(1)),
        BinOp('*', Variable('y'), ConstInt(2)))


# --- compile + run (tier 0) -------------------------------------------------

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
    assert run0(SUM_TAIL, 0) == 0


def test_gcd():
    assert run0(GCD, 36) == 12
    assert run0(GCD, 17) == 1


def test_ack():
    assert run0(ACK, 3) == 9        # ack(2,3) = 9


def test_tak():
    assert run0(TAK, 6) == 0        # tak(6,0,1) == 0


def test_arithmetic_and_precedence():
    assert run0("let rec main x = 2 + 3 * 4 - 10 / 2 ;;", 0) == 2 + 3 * 4 - 10 / 2
    assert run0("let rec main x = 10 - 3 - 2 ;;", 0) == 5
    assert run0("let rec main x = (2 + 3) * 4 ;;", 0) == 20
    assert run0("let rec main x = 17 % 5 ;;", 0) == 2


def test_let_in():
    assert run0("let rec main x = let y = x + 1 in y * y ;;", 4) == 25
    # nested lets, inner shadows the binding while it is in scope
    src = "let rec main x = let a = x + 1 in let b = a + 1 in a + b ;;"
    assert run0(src, 10) == (10 + 1) + (10 + 2)      # 23


def test_comparisons():
    def cmp_prog(op):
        return ("let rec main x = if x %s 5 then 1 else 0 ;;" % op)
    assert run0(cmp_prog("<"), 4) == 1 and run0(cmp_prog("<"), 5) == 0
    assert run0(cmp_prog("<="), 5) == 1 and run0(cmp_prog("<="), 6) == 0
    assert run0(cmp_prog(">"), 6) == 1 and run0(cmp_prog(">"), 5) == 0
    assert run0(cmp_prog(">="), 5) == 1 and run0(cmp_prog(">="), 4) == 0
    assert run0(cmp_prog("=="), 5) == 1 and run0(cmp_prog("=="), 4) == 0
    assert run0(cmp_prog("!="), 4) == 1 and run0(cmp_prog("!="), 5) == 0


def test_comparison_as_value():
    # a comparison used directly as a value (not just in an `if`)
    assert run0("let rec main x = x < 5 ;;", 4) == 1
    assert run0("let rec main x = x < 5 ;;", 5) == 0
    assert run0("let rec main x = x != 5 ;;", 5) == 0
    assert run0("let rec main x = x != 5 ;;", 4) == 1


def test_errors():
    py.test.raises(CompileError, compile_source,
                   "let rec f n = n ;;")                 # no main
    py.test.raises(CompileError, compile_source,
                   "let rec main x = g x ;;")            # unknown call
    py.test.raises(CompileError, compile_source,
                   "let rec main x y = x ;;")            # main arity != 1
