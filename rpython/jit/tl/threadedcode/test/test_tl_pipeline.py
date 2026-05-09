from rpython.jit.tl.threadedcode.tl_pipeline import interpret_int
from rpython.jit.tl.threadedcode.parser import parse
from rpython.jit.tl.threadedcode.tl_rparse import parse_program


def test_pipeline_let_add():
    assert interpret_int('let x = 1 in x + 1') == 2


def test_pipeline_const_expr():
    assert interpret_int('3 + 4') == 7


def test_rparse_matches_ebnf_samples():
    samples = [
        'y + 1',
        'let x = 1 in x + 1',
        'let rec f x y = 1;; f 1 2',
        'Array.make 4 0',
        'a.(i)',
        'a.(i) <- x',
    ]
    for s in samples:
        assert parse(s) == parse_program(s)
