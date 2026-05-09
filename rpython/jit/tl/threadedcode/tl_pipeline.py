"""
Source -> bytecode -> TLA interpreter (tier-2 ``Frame._interp``).

The lexer/parser in ``tl_rparse`` and ``compiler.compile_program`` are written
for RPython; ``parser.parse`` (ebnf) is only for CPython tests / tooling.
"""
from rpython.jit.tl.threadedcode.bytecode import assemble, Bytecode
from rpython.jit.tl.threadedcode.compiler import compile_program
from rpython.jit.tl.threadedcode.tl_rparse import parse_program
from rpython.jit.tl.threadedcode.tla import Frame
from rpython.jit.tl.threadedcode.object import (
    W_IntObject,
    W_FloatObject,
    OperationError,
)


def compile_words_from_source(source):
    """Parse with ``tl_rparse`` and return the pre-assemble byte list."""
    prog = parse_program(source)
    return compile_program(prog)


def compile_string_from_source(source):
    """Return packed bytecode string suitable for ``Frame``."""
    return assemble(compile_words_from_source(source))


def interpret_source(source):
    """Run a program and return the ``W_Object`` left by ``EXIT``."""
    bc = Bytecode(compile_string_from_source(source))
    frame = Frame(bc)
    return frame._interp(0)


def interpret_int(source):
    """Like ``interpret_source`` but unwraps a boxed int."""
    w_x = interpret_source(source)
    if isinstance(w_x, W_IntObject):
        return w_x.intvalue
    if isinstance(w_x, W_FloatObject):
        raise OperationError()
    raise OperationError()
