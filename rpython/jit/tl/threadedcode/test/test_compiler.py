from rpython.jit.tl.threadedcode import bytecode as bc
from rpython.jit.tl.threadedcode.tl_rparse import parse_program
from rpython.jit.tl.threadedcode.compiler import compile_program
from rpython.jit.tl.threadedcode.parser import Program, ConstInt


def test_compile_let_add():
    code = compile_program(parse_program('let x = 1 in x + 1'))
    assert bc.CONST_INT in code
    assert bc.DUPN in code
    assert bc.ADD in code
    assert bc.POP1 in code
    assert bc.EXIT in code


def test_compile_const_program():
    code = compile_program(Program([ConstInt(42)]))
    assert code[0] == bc.CONST_INT
    assert code[1] == 42
    assert code[-1] == bc.EXIT


def test_assemble_roundtrip():
    words = compile_program(parse_program('2 + 3'))
    assert words[0] == bc.CONST_INT and words[1] == 2
    assert words[2] == bc.CONST_INT and words[3] == 3
    assert words[4] == bc.ADD
    s = bc.assemble(words)
    assert len(s) == len(words)
