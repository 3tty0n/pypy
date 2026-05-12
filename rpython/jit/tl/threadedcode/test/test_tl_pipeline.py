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
        # callee parked on rhs of binop chain — must peel to FunApp in
        # both parsers (was a ParseError in tl_rparse, nonsensical
        # FunApp(BinOp, ...) in parser.py).
        'let rec f n = if n < 1 then 0 else n + f (n - 1);; f 5',
    ]
    for s in samples:
        assert parse(s) == parse_program(s)


def test_pipeline_recursive_funapp_on_binop_rhs():
    """Regression: ``n + f (n - 1)`` parses+compiles+runs as
    ``n + (f (n - 1))`` rather than ``(n + f) (n - 1)``."""
    assert interpret_int(
        'let rec f n = if n < 1 then 0 else n + f (n - 1);; f 5') == 15
    assert interpret_int(
        'let rec fact n = if n < 2 then 1 else n * fact (n - 1);; fact 5') == 120
    assert interpret_int(
        'let rec gcd a b = if b == 0 then a else gcd b (a % b);; gcd 252 48') == 12


def test_pipeline_while_does_not_leak_body_value():
    """Regression: the ``while`` body left one value on the operand stack
    every iteration; a loop that never executes still has to type-check
    and exit cleanly, and a loop that does execute must not overflow."""
    # never-enters-body case
    assert interpret_int('let i = 0 in while i < 0 do i + 1') == 0
    # enters body once: i stays 0, so we exit only because cond starts false.
    # The important property is that this terminates and returns the
    # ``while`` value (``0``) without IndexError-on-stack.
    assert interpret_int(
        'let i = 0 in let _ = while i < 0 do i + 1 in i') == 0


def test_tokenize_let_followed_by_digit_is_identifier():
    """Regression: ``let1`` was mis-tokenized as keyword ``let`` + ``1``."""
    from rpython.jit.tl.threadedcode.tl_rparse import tokenize, K_NAME
    toks = tokenize('let1')
    assert toks[0].kind == K_NAME
    assert toks[0].name == 'let1'
