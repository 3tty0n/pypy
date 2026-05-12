"""
RPython-friendly lexer/parser for ``grammar.txt`` (no py.path / ebnfparse).

Builds ``tl_ast`` nodes for ``compiler.compile_program``.
"""
from rpython.rlib.rfloat import string_to_float
from rpython.rlib.rstring import ParseStringError
from rpython.jit.tl.threadedcode.tl_ast import (
    Program,
    ConstInt,
    ConstFloat,
    Variable,
    BinOp,
    LetIn,
    Function,
    FunApp,
    ArrayMake,
    ArrayLoad,
    ArrayStore,
    If,
    While,
)


class ParseError(Exception):
    pass


def _is_ws(c):
    return c == ' ' or c == '\t' or c == '\n'


def _is_alpha(c):
    if c == '_':
        return True
    o = ord(c)
    if 97 <= o <= 122:
        return True
    if 65 <= o <= 90:
        return True
    return False


def _is_digit(c):
    o = ord(c)
    return 48 <= o <= 57


def _starts_with(s, pos, prefix):
    lp = len(prefix)
    if pos + lp > len(s):
        return False
    i = 0
    while i < lp:
        if s[pos + i] != prefix[i]:
            return False
        i += 1
    return True


def _skip_ws(s, pos):
    n = len(s)
    while pos < n and _is_ws(s[pos]):
        pos += 1
    return pos


def _read_decimal(s, pos):
    n = len(s)
    neg = False
    if pos < n and s[pos] == '-':
        neg = True
        pos += 1
    if pos >= n:
        raise ParseError()
    c = s[pos]
    if c == '0':
        pos += 1
        val = 0
    else:
        if not _is_digit(c) or c == '0':
            raise ParseError()
        val = 0
        while pos < n and _is_digit(s[pos]):
            val = val * 10 + (ord(s[pos]) - 48)
            pos += 1
    if neg:
        val = -val
    return val, pos


def _read_float_literal(s, pos):
    n = len(s)
    start = pos
    if pos < n and s[pos] == '-':
        pos += 1
    saw_digit = False
    while pos < n and _is_digit(s[pos]):
        saw_digit = True
        pos += 1
    if pos >= n or s[pos] != '.':
        raise ParseError()
    pos += 1
    while pos < n and _is_digit(s[pos]):
        saw_digit = True
        pos += 1
    if not saw_digit:
        raise ParseError()
    if pos < n and (s[pos] == 'e' or s[pos] == 'E'):
        pos += 1
        if pos < n and (s[pos] == '+' or s[pos] == '-'):
            pos += 1
        if pos >= n or not _is_digit(s[pos]):
            raise ParseError()
        while pos < n and _is_digit(s[pos]):
            pos += 1
    sub = s[start:pos]
    try:
        fv = string_to_float(sub)
    except ParseStringError:
        raise ParseError()
    return fv, pos


def _read_name(s, pos):
    n = len(s)
    if pos >= n or not _is_alpha(s[pos]):
        raise ParseError()
    start = pos
    pos += 1
    while pos < n and (_is_alpha(s[pos]) or _is_digit(s[pos])):
        pos += 1
    return s[start:pos], pos


K_EOF = 0
K_INT = 1
K_FLOAT = 2
K_NAME = 3
K_LPAREN = 4
K_RPAREN = 5
K_PLUS = 6
K_MINUS = 7
K_LT = 8
K_EQEQ = 9
K_DOT = 10
K_EQ = 11
K_SEMI2 = 12
K_LET = 13
K_LETREC = 14
K_IN = 15
K_ARRAY_MAKE = 16
K_ARROW = 17
K_STAR = 18
K_PERCENT = 19
K_GT = 20
K_IF = 21
K_THEN = 22
K_ELSE = 23
K_WHILE = 24
K_DO = 25


class _Tok(object):
    __slots__ = ('kind', 'ival', 'name', 'fval')

    def __init__(self, kind, ival=0, name='', fval=0.0):
        self.kind = kind
        self.ival = ival
        self.name = name
        self.fval = fval


def tokenize(s):
    toks = []
    pos = 0
    n = len(s)
    while pos < n:
        pos = _skip_ws(s, pos)
        if pos >= n:
            break
        c = s[pos]
        if c == '(':
            toks.append(_Tok(K_LPAREN))
            pos += 1
            continue
        if c == '*':
            toks.append(_Tok(K_STAR))
            pos += 1
            continue
        if c == '%':
            toks.append(_Tok(K_PERCENT))
            pos += 1
            continue
        if c == '>':
            toks.append(_Tok(K_GT))
            pos += 1
            continue
        if c == ')':
            toks.append(_Tok(K_RPAREN))
            pos += 1
            continue
        if c == '+':
            toks.append(_Tok(K_PLUS))
            pos += 1
            continue
        if c == '-':
            if pos + 1 < n and _is_digit(s[pos + 1]):
                v, pos = _read_decimal(s, pos)
                toks.append(_Tok(K_INT, v, ''))
            else:
                toks.append(_Tok(K_MINUS))
                pos += 1
            continue
        if c == '<':
            pos += 1
            pos2 = _skip_ws(s, pos)
            if pos2 < n and s[pos2] == '-':
                toks.append(_Tok(K_ARROW))
                pos = pos2 + 1
            else:
                toks.append(_Tok(K_LT))
            continue
        if c == '=':
            if pos + 1 < n and s[pos + 1] == '=':
                toks.append(_Tok(K_EQEQ))
                pos += 2
            else:
                toks.append(_Tok(K_EQ))
                pos += 1
            continue
        if c == '.':
            toks.append(_Tok(K_DOT))
            pos += 1
            continue
        if c == ';' and pos + 1 < n and s[pos + 1] == ';':
            toks.append(_Tok(K_SEMI2))
            pos += 2
            continue
        if _starts_with(s, pos, 'let rec'):
            pos2 = pos + 7
            if pos2 <= n:
                toks.append(_Tok(K_LETREC))
                pos = pos2
                continue
        if _starts_with(s, pos, 'let'):
            pos2 = pos + 3
            if pos2 < n and (_is_alpha(s[pos2]) or _is_digit(s[pos2])):
                pass
            else:
                toks.append(_Tok(K_LET))
                pos = pos2
                continue
        if _starts_with(s, pos, 'if'):
            pos2 = pos + 2
            if pos2 < n and (_is_alpha(s[pos2]) or _is_digit(s[pos2])):
                pass
            else:
                toks.append(_Tok(K_IF))
                pos = pos2
                continue
        if _starts_with(s, pos, 'then'):
            pos2 = pos + 4
            if pos2 < n and (_is_alpha(s[pos2]) or _is_digit(s[pos2])):
                pass
            else:
                toks.append(_Tok(K_THEN))
                pos = pos2
                continue
        if _starts_with(s, pos, 'else'):
            pos2 = pos + 4
            if pos2 < n and (_is_alpha(s[pos2]) or _is_digit(s[pos2])):
                pass
            else:
                toks.append(_Tok(K_ELSE))
                pos = pos2
                continue
        if _starts_with(s, pos, 'while'):
            pos2 = pos + 5
            if pos2 < n and (_is_alpha(s[pos2]) or _is_digit(s[pos2])):
                pass
            else:
                toks.append(_Tok(K_WHILE))
                pos = pos2
                continue
        if _starts_with(s, pos, 'do'):
            pos2 = pos + 2
            if pos2 < n and (_is_alpha(s[pos2]) or _is_digit(s[pos2])):
                pass
            else:
                toks.append(_Tok(K_DO))
                pos = pos2
                continue
        if _starts_with(s, pos, 'Array.make'):
            pos2 = pos + 10
            if pos2 < n:
                nc = s[pos2]
                if _is_alpha(nc) or _is_digit(nc):
                    pass
                else:
                    toks.append(_Tok(K_ARRAY_MAKE))
                    pos = pos2
                    continue
            else:
                toks.append(_Tok(K_ARRAY_MAKE))
                pos = pos2
                continue
        if _starts_with(s, pos, 'in'):
            pos2 = pos + 2
            if pos2 < n:
                nc = s[pos2]
                if _is_alpha(nc) or _is_digit(nc):
                    pass
                else:
                    toks.append(_Tok(K_IN))
                    pos = pos2
                    continue
            else:
                toks.append(_Tok(K_IN))
                pos = pos2
                continue
        if _is_digit(c):
            p2 = pos
            while p2 < n and _is_digit(s[p2]):
                p2 += 1
            if p2 < n and s[p2] == '.':
                fv, pos = _read_float_literal(s, pos)
                toks.append(_Tok(K_FLOAT, 0, '', fv))
            else:
                v, pos = _read_decimal(s, pos)
                toks.append(_Tok(K_INT, v, ''))
            continue
        if _is_alpha(c):
            name, pos = _read_name(s, pos)
            toks.append(_Tok(K_NAME, 0, name))
            continue
        raise ParseError()
    toks.append(_Tok(K_EOF))
    return toks


def _peel_funapp_rhs(node, args):
    """If ``node`` is a right-leaning BinOp chain whose rightmost leaf is a
    Variable, return a copy with that leaf replaced by ``FunApp(var, args)``.
    Returns ``None`` otherwise (caller should signal a parse error).
    """
    if isinstance(node, Variable):
        return FunApp(node, args)
    if isinstance(node, BinOp):
        new_right = _peel_funapp_rhs(node.right, args)
        if new_right is None:
            return None
        return BinOp(node.op, node.left, new_right)
    return None


class RParser(object):
    def __init__(self, toks):
        self.toks = toks
        self.i = 0
        self.n = len(toks)

    def peek(self):
        return self.toks[self.i]

    def expect(self, k):
        t = self.peek()
        if t.kind != k:
            raise ParseError()
        self.i += 1
        return t

    def parse_program(self):
        exprs = []
        while self.peek().kind != K_EOF:
            exprs.append(self.parse_expr())
        return Program(exprs)

    def _can_start_simple_expr(self):
        k = self.peek().kind
        if k == K_LPAREN or k == K_INT or k == K_FLOAT or k == K_NAME:
            return True
        return False

    def parse_expr(self):
        t = self.peek()
        if t.kind == K_IF:
            self.expect(K_IF)
            cnd = self.parse_expr()
            self.expect(K_THEN)
            thn = self.parse_expr()
            self.expect(K_ELSE)
            els = self.parse_expr()
            return If(cnd, thn, els)
        if t.kind == K_WHILE:
            self.expect(K_WHILE)
            cnd = self.parse_simple_expr()
            self.expect(K_DO)
            body = self.parse_expr()
            return While(cnd, body)
        if t.kind == K_LET:
            self.expect(K_LET)
            nm = self.expect(K_NAME).name
            self.expect(K_EQ)
            rhs = self.parse_expr()
            self.expect(K_IN)
            body = self.parse_expr()
            return LetIn(nm, rhs, body)
        if t.kind == K_LETREC:
            self.expect(K_LETREC)
            fn = self.expect(K_NAME).name
            args = []
            while self.peek().kind != K_EQ:
                args.append(self.expect(K_NAME).name)
            self.expect(K_EQ)
            body = self.parse_expr()
            self.expect(K_SEMI2)
            return Function(fn, args, body)
        if t.kind == K_ARRAY_MAKE:
            self.expect(K_ARRAY_MAKE)
            sz = self.parse_simple_expr()
            ini = self.parse_simple_expr()
            return ArrayMake(sz, ini)
        se = self.parse_simple_expr()
        args = []
        while self._can_start_simple_expr():
            args.append(self.parse_simple_expr())
        if len(args) == 0:
            return se
        if isinstance(se, Variable):
            return FunApp(se, args)
        # Right-recursive simple_expr may have parked the callee as the
        # rightmost leaf of a BinOp chain (e.g. ``n + f (n - 1)`` parses with
        # ``se = BinOp(+, n, f)`` and ``args = [n-1]``). Promote that leaf to
        # a FunApp in place.
        peeled = _peel_funapp_rhs(se, args)
        if peeled is None:
            raise ParseError()
        return peeled

    def parse_simple_expr(self):
        if self.peek().kind == K_LPAREN:
            self.expect(K_LPAREN)
            e = self.parse_simple_expr()
            self.expect(K_RPAREN)
            return e
        left = self.parse_atom_expr()
        t = self.peek()
        if t.kind == K_PLUS:
            self.expect(K_PLUS)
            right = self.parse_simple_expr()
            return BinOp('+', left, right)
        if t.kind == K_MINUS:
            self.expect(K_MINUS)
            right = self.parse_simple_expr()
            return BinOp('-', left, right)
        if t.kind == K_LT:
            self.expect(K_LT)
            right = self.parse_simple_expr()
            return BinOp('<', left, right)
        if t.kind == K_EQEQ:
            self.expect(K_EQEQ)
            right = self.parse_simple_expr()
            return BinOp('==', left, right)
        if t.kind == K_STAR:
            self.expect(K_STAR)
            right = self.parse_simple_expr()
            return BinOp('*', left, right)
        if t.kind == K_PERCENT:
            self.expect(K_PERCENT)
            right = self.parse_simple_expr()
            return BinOp('%', left, right)
        if t.kind == K_GT:
            self.expect(K_GT)
            right = self.parse_simple_expr()
            return BinOp('>', left, right)
        return left

    def parse_atom_expr(self):
        left = self.parse_atom()
        if self.peek().kind == K_DOT:
            self.expect(K_DOT)
            self.expect(K_LPAREN)
            idx = self.parse_simple_expr()
            self.expect(K_RPAREN)
            if self.peek().kind == K_ARROW:
                self.expect(K_ARROW)
                val = self.parse_simple_expr()
                return ArrayStore(left, idx, val)
            return ArrayLoad(left, idx)
        return left

    def parse_atom(self):
        t = self.peek()
        if t.kind == K_INT:
            self.expect(K_INT)
            return ConstInt(t.ival)
        if t.kind == K_FLOAT:
            self.expect(K_FLOAT)
            return ConstFloat(t.fval)
        if t.kind == K_NAME:
            self.expect(K_NAME)
            return Variable(t.name)
        raise ParseError()


def parse_program(source):
    toks = tokenize(source)
    p = RParser(toks)
    return p.parse_program()
