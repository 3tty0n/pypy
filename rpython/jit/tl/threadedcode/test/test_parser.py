from rpython.jit.tl.threadedcode.parser import parse, _parse
from rpython.jit.tl.threadedcode.tl_ast import (
    Program,
    BinOp,
    Variable,
    ConstInt,
    LetIn,
    Function,
    FunApp,
    ArrayMake,
    ArrayLoad,
    ArrayStore,
)

def test_binop():
    assert parse('y + 1') == Program([BinOp("+", Variable("y"), ConstInt(1))])
    assert parse('y - 1') == Program([BinOp("-", Variable("y"), ConstInt(1))])
    assert parse('y < 1') == Program([BinOp("<", Variable("y"), ConstInt(1))])
    assert parse('y == 1') == Program([BinOp("==", Variable("y"), ConstInt(1))])

def test_parentheses():
    assert parse('(x)') == Program([Variable('x')])

def test_let():
    assert parse('let x = 1 in x + 1') == Program(
        [LetIn('x', ConstInt(1),
               BinOp('+', Variable('x'), ConstInt(1)))])

def test_letrec():
    assert parse('let rec f x y = 1;; f 1 2') == Program(
        [Function('f', ['x', 'y'], ConstInt(1)),
         FunApp(Variable('f'), [ConstInt(1), ConstInt(2)])])


def test_array_make():
    p = parse('Array.make 4 0')
    assert p == Program([ArrayMake(ConstInt(4), ConstInt(0))])


def test_array_load_store():
    assert parse('a.(i)') == Program([ArrayLoad(Variable('a'), Variable('i'))])
    assert parse('a.(i) <- x') == Program(
        [ArrayStore(Variable('a'), Variable('i'), Variable('x'))])
