"""A native frontend (parser + compiler) for the TLA threaded-code language.

This is the OCaml-flavoured surface language described by grammar.txt: a program
is a list of `let rec` integer functions, with `if/then/else`, `let .. in` local
bindings, the usual arithmetic/comparison operators, and recursion.  ``parse``
builds an AST; ``compile_source`` lowers that AST to a TLA bytecode int list --
exactly the ``code = [...]`` found in lang/*.tla.py and consumed by tla.run().

    let rec fib n =
      if n < 2 then n
      else fib (n - 1) + fib (n - 2)
    ;;
    let rec main x = fib x ;;

``main`` must take exactly one parameter (the integer the runtime pushes onto the
initial frame); it is laid out at pc 0 and ends with EXIT, every other function
ends with RET.  A self tail-call in tail position (e.g. ``let rec loop n acc =
... loop (n - 1) (acc + n)``) compiles to the in-frame FRAME_RESET + JUMP loop --
the JIT-friendly shape used by the hand-written lang/mb_*.tla.py and sum-tail --
rather than a fresh recursive frame.

This module is a *build-time* tool (like bytecode.assemble and frontend.py): it
runs under plain CPython/PyPy2, is never translated, and may use the rlib parsing
library, dicts and exceptions freely.

Notes on the JIT-facing bytecode
--------------------------------
The comparison ops are non-strict: LT dispatches W_Object.le (a <= b) and GT
dispatches ge (a >= b).  The compiler bridges the strict/inequality comparisons
``< > !=`` by emitting the complementary op; in an ``if`` condition it just swaps
the then/else branches (no extra ops), and where a comparison is used as a plain
value it negates the result with ``== 0``.  Either way the source semantics are
exact.
"""

import os

import py
from rpython.rlib.parsing.ebnfparse import parse_ebnf, make_parse_function

from rpython.jit.tl.threadedcode import bytecode as bc


currentdir = os.path.dirname(os.path.abspath(__file__))
grammar = py.path.local(currentdir).join('grammar.txt').read("rt")
_regexs, _rules, _ToAST = parse_ebnf(grammar)
_parse = make_parse_function(_regexs, _rules, eof=True)


# ---------------------------------------------------------------------------
# AST
# ---------------------------------------------------------------------------

class Node(object):
    """Abstract AST node with structural equality (for the parser tests)."""

    def __eq__(self, other):
        return (self.__class__ == other.__class__ and
                self.__dict__ == other.__dict__)

    def __ne__(self, other):
        return not self == other


class Program(Node):
    def __init__(self, functions):
        self.functions = functions

    def __repr__(self):
        return "Program(%r)" % (self.functions,)


class Function(Node):
    def __init__(self, name, params, body):
        self.name = name
        self.params = params        # list of str
        self.body = body            # expression node

    def __repr__(self):
        return "Function(%r, %r, %r)" % (self.name, self.params, self.body)


class If(Node):
    def __init__(self, cond, then, orelse):
        self.cond = cond
        self.then = then
        self.orelse = orelse

    def __repr__(self):
        return "If(%r, %r, %r)" % (self.cond, self.then, self.orelse)


class LetIn(Node):
    def __init__(self, name, value, body):
        self.name = name
        self.value = value
        self.body = body

    def __repr__(self):
        return "LetIn(%r, %r, %r)" % (self.name, self.value, self.body)


class BinOp(Node):
    def __init__(self, op, left, right):
        self.op = op                # source operator string
        self.left = left
        self.right = right

    def __repr__(self):
        return "BinOp(%r, %r, %r)" % (self.op, self.left, self.right)


class FunApp(Node):
    def __init__(self, funcname, args):
        self.funcname = funcname    # str
        self.args = args            # list of expression nodes

    def __repr__(self):
        return "FunApp(%r, %r)" % (self.funcname, self.args)


class Variable(Node):
    def __init__(self, name):
        self.name = name

    def __repr__(self):
        return "Variable(%r)" % (self.name,)


class ConstInt(Node):
    def __init__(self, intval):
        self.intval = intval

    def __repr__(self):
        return "ConstInt(%r)" % (self.intval,)


# ---------------------------------------------------------------------------
# Parse tree -> AST.  Operates on the ToAST-cleaned tree (keywords/punctuation
# dropped, definition list flattened, `if` rendered as 3 expr children).  The
# precedence-chain rules keep every level, so each visitor recurses through the
# single-child case.
# ---------------------------------------------------------------------------

class _Transformer(object):
    def visit_main(self, node):
        return Program([self.visit_definition(c) for c in node.children])

    def visit_definition(self, node):
        # children: VARIABLE(name), formal_args, expr(body)
        name = node.children[0].additional_info
        params = self._formal_args(node.children[1])
        body = self.visit_expr(node.children[2])
        return Function(name, params, body)

    def _formal_args(self, node):
        names = []
        while True:
            names.append(node.children[0].additional_info)
            if len(node.children) == 1:
                break
            node = node.children[1]
        return names

    def visit_expr(self, node):
        children = node.children
        if len(children) == 1:                      # cmp_expr
            return self.visit_cmp(children[0])
        # 3 children: `let .. in` (VARIABLE first) or `if` (expr first)
        if children[0].symbol == 'VARIABLE':
            return LetIn(children[0].additional_info,
                         self.visit_expr(children[1]),
                         self.visit_expr(children[2]))
        return If(self.visit_expr(children[0]),
                  self.visit_expr(children[1]),
                  self.visit_expr(children[2]))

    def visit_cmp(self, node):
        children = node.children
        if len(children) == 1:
            return self.visit_add(children[0])
        return BinOp(children[1].additional_info,
                     self.visit_add(children[0]),
                     self.visit_add(children[2]))

    def visit_add(self, node):
        return self._left_assoc(node, self.visit_mul)

    def visit_mul(self, node):
        return self._left_assoc(node, self.visit_app)

    def _left_assoc(self, node, visit_operand):
        # the chain rule is right-recursive (a OP (b OP (c ...))); collect the
        # operands/operators and fold *left* so `a - b - c` == `(a - b) - c`.
        operands = []
        ops = []
        while True:
            operands.append(visit_operand(node.children[0]))
            if len(node.children) == 3:
                ops.append(node.children[1].additional_info)
                node = node.children[2]
            else:
                break
        res = operands[0]
        for i, op in enumerate(ops):
            res = BinOp(op, res, operands[i + 1])
        return res

    def visit_app(self, node):
        # app_expr is right-nested juxtaposition: atom (atom (atom ...)).  A
        # single atom is just a value; two or more is `f arg1 arg2 ...`.
        atoms = []
        while True:
            atoms.append(self.visit_atom(node.children[0]))
            if len(node.children) == 2:
                node = node.children[1]
            else:
                break
        if len(atoms) == 1:
            return atoms[0]
        head = atoms[0]
        if not isinstance(head, Variable):
            raise CompileError("only direct calls `f a b ...` are supported")
        return FunApp(head.name, atoms[1:])

    def visit_atom(self, node):
        child = node.children[0]
        if child.symbol == 'DECIMAL':
            return ConstInt(int(child.additional_info))
        if child.symbol == 'VARIABLE':
            return Variable(child.additional_info)
        if child.symbol == 'expr':                  # parenthesised
            return self.visit_expr(child)
        raise CompileError("unexpected atom: %s" % child.symbol)


_transformer = _Transformer()


def parse(source):
    """Parse TLA source text into a Program AST."""
    tree = _parse(source).visit(_ToAST())
    if isinstance(tree, list):
        tree = tree[0]
    return _transformer.visit_main(tree)


# ---------------------------------------------------------------------------
# AST -> bytecode
# ---------------------------------------------------------------------------

class CompileError(Exception):
    pass


# Bytecode LT == `<=`, GT == `>=`, EQ == `==`.  For an `if` condition we map
# each source comparison to (bytecode_op, swap_branches); swap=True means the
# bytecode test is the negation of the source test, so the then/else branches
# are exchanged instead of emitting an explicit NOT.
_CMP_SWAP = {
    '<=': (bc.LT, False),
    '>=': (bc.GT, False),
    '==': (bc.EQ, False),
    '<':  (bc.GT, True),    # a <  b == not (a >= b)
    '>':  (bc.LT, True),    # a >  b == not (a <= b)
    '!=': (bc.EQ, True),    # a != b == not (a == b)
}

_ARITH = {
    '+': bc.ADD,
    '-': bc.SUB,
    '*': bc.MUL,
    '/': bc.DIV,
    '%': bc.MOD,
}

_COMPARISONS = ('<', '>', '<=', '>=', '==', '!=')


class _Label(object):
    __slots__ = ('name',)

    def __init__(self, name=''):
        self.name = name


class _Ref(object):
    """A 1-byte placeholder for a _Label's resolved byte offset."""
    __slots__ = ('label',)

    def __init__(self, label):
        self.label = label


class _Func(object):
    def __init__(self, node):
        self.name = node.name
        self.params = node.params
        self.body = node.body
        self.label = _Label(node.name)
        self.body_label = _Label(node.name + ':body')


class Compiler(object):
    def __init__(self):
        self.items = []            # int | _Label | _Ref
        self.funcs = {}            # name -> _Func
        self.cur = None            # current _Func
        self.depth = 0             # operand-stack depth from frame base
        self.slot_of = {}          # name -> absolute frame slot

    # -- low-level emit -------------------------------------------------------
    def _op(self, opcode):
        self.items.append(opcode)

    def _byte(self, b):
        if not (0 <= b <= 255):
            raise CompileError("byte argument out of range: %d" % b)
        self.items.append(b)

    def _ref(self, label):
        self.items.append(_Ref(label))

    def _place(self, label):
        self.items.append(label)

    def _push_const(self, value):
        if 0 <= value <= 255:
            self._op(bc.CONST_INT); self._byte(value)
        elif value > 255:
            self._op(bc.CONST_N)
            self._byte((value >> 24) & 0xff); self._byte((value >> 16) & 0xff)
            self._byte((value >> 8) & 0xff); self._byte(value & 0xff)
        elif -255 <= value < 0:
            self._op(bc.CONST_NEG_INT); self._byte(-value)
        else:
            v = -value
            self._op(bc.CONST_NEG_N)
            self._byte((v >> 24) & 0xff); self._byte((v >> 16) & 0xff)
            self._byte((v >> 8) & 0xff); self._byte(v & 0xff)
        self.depth += 1

    def _read_var(self, name):
        if name not in self.slot_of:
            raise CompileError("unknown name %r in %s" % (name, self.cur.name))
        off = self.depth - self.slot_of[name] - 1
        self._op(bc.DUPN); self._byte(off)
        self.depth += 1

    # -- expressions (value position: leave exactly one value on top) ---------
    def compile_value(self, node):
        if isinstance(node, ConstInt):
            self._push_const(node.intval)
        elif isinstance(node, Variable):
            self._read_var(node.name)
        elif isinstance(node, BinOp):
            self._binop_value(node)
        elif isinstance(node, FunApp):
            self._call(node)
        elif isinstance(node, If):
            self._if_value(node)
        elif isinstance(node, LetIn):
            self._let_value(node)
        else:
            raise CompileError("unsupported expression: %r" % (node,))

    def _binop_value(self, node):
        if node.op in _ARITH:
            self.compile_value(node.left)
            self.compile_value(node.right)
            self._op(_ARITH[node.op]); self.depth -= 1
            return
        if node.op in _COMPARISONS:
            bcop, swap = _CMP_SWAP[node.op]
            self.compile_value(node.left)
            self.compile_value(node.right)
            self._op(bcop); self.depth -= 1
            if swap:                       # negate: (cmp) == 0
                self._push_const(0)
                self._op(bc.EQ); self.depth -= 1
            return
        raise CompileError("unsupported operator %r" % (node.op,))

    def _call(self, node):
        callee = self.funcs.get(node.funcname)
        if callee is None:
            raise CompileError("call to unknown function %r" % node.funcname)
        argnum = len(node.args)
        if argnum != len(callee.params):
            raise CompileError("%s expects %d args, got %d"
                               % (node.funcname, len(callee.params), argnum))
        # push dummy (callee slot0), then the args; CALL_ASSEMBLER drops the
        # args and pushes the result; POP1 discards the dummy, keeping the
        # result on top.
        self._push_const(0)
        for a in node.args:
            self.compile_value(a)
        self._op(bc.CALL_ASSEMBLER); self._ref(callee.label); self._byte(argnum)
        self.depth -= argnum               # args dropped
        self.depth += 1                    # result pushed
        self._op(bc.POP1); self.depth -= 1

    def _emit_cond(self, cond):
        """Emit a condition leaving a bool on top; return the swap flag."""
        if isinstance(cond, BinOp) and cond.op in _COMPARISONS:
            bcop, swap = _CMP_SWAP[cond.op]
            self.compile_value(cond.left)
            self.compile_value(cond.right)
            self._op(bcop); self.depth -= 1
            return swap
        # general truthiness: nonzero is true
        self.compile_value(cond)
        return False

    def _if_value(self, node):
        depth0 = self.depth
        swap = self._emit_cond(node.cond)
        # JUMP_IF jumps to the true-label when the bool is true; the false path
        # falls through.  swap=True means the bytecode test is negated, so the
        # branches are exchanged.
        false_branch = node.then if swap else node.orelse
        true_branch = node.orelse if swap else node.then
        lthen = _Label('then')
        self._op(bc.JUMP_IF); self._ref(lthen)
        self.depth = depth0
        self.compile_value(false_branch)
        lend = _Label('endif')
        self._op(bc.JUMP); self._ref(lend)
        self._place(lthen)
        self.depth = depth0
        self.compile_value(true_branch)
        self._place(lend)

    def _let_value(self, node):
        depth0 = self.depth
        self.compile_value(node.value)         # let-local at slot depth0
        saved = self.slot_of.get(node.name, None)
        self.slot_of[node.name] = depth0
        self.compile_value(node.body)          # result on top, local below
        self._op(bc.POP1); self.depth -= 1     # drop the local, keep result
        self._restore(node.name, saved)

    def _restore(self, name, saved):
        if saved is None:
            del self.slot_of[name]
        else:
            self.slot_of[name] = saved

    # -- expressions (tail position: produce value and leave the function) ----
    def compile_tail(self, node, allow_loop=True):
        if isinstance(node, If):
            depth0 = self.depth
            swap = self._emit_cond(node.cond)
            false_branch = node.then if swap else node.orelse
            true_branch = node.orelse if swap else node.then
            lthen = _Label('then')
            self._op(bc.JUMP_IF); self._ref(lthen)
            self.depth = depth0
            self.compile_tail(false_branch, allow_loop)
            self._place(lthen)
            self.depth = depth0
            self.compile_tail(true_branch, allow_loop)
            return
        if isinstance(node, LetIn):
            depth0 = self.depth
            self.compile_value(node.value)
            saved = self.slot_of.get(node.name, None)
            self.slot_of[node.name] = depth0
            # The frame now carries an extra local, which the FRAME_RESET tail
            # loop has no room for, so disallow looping inside the let body.
            self.compile_tail(node.body, allow_loop=False)
            self._restore(node.name, saved)
            return
        if (allow_loop and isinstance(node, FunApp)
                and node.funcname == self.cur.name
                and self.cur.name != 'main'):
            self._tail_self_call(node)
            return
        # ordinary value in tail position: compute it and return.
        self.compile_value(node)
        if self.cur.name == 'main':
            self._op(bc.EXIT)
        else:
            self._op(bc.RET); self._byte(len(self.cur.params))

    def _tail_self_call(self, node):
        # Reuse the current frame instead of recursing: evaluate the k new args
        # on top, then FRAME_RESET (o=k, l=0, n=k) shifts them down over the
        # current param slots, preserving the return slot, and JUMP back to the
        # function entry -- the JIT-friendly tail-loop shape of the hand-written
        # lang/sum-tail.tla.py / mb_*.tla.py.  Identical to frontend.py's scheme.
        k = len(self.cur.params)
        if len(node.args) != k:
            raise CompileError("tail call arity mismatch in %s" % self.cur.name)
        for a in node.args:
            self.compile_value(a)
        self._op(bc.FRAME_RESET); self._byte(k); self._byte(0); self._byte(k)
        self.depth -= k
        self._op(bc.JUMP); self._ref(self.cur.body_label)

    # -- function / program ---------------------------------------------------
    def function(self, func):
        self.cur = func
        self._place(func.label)
        self._place(func.body_label)
        if func.name == 'main':
            if len(func.params) != 1:
                raise CompileError("main must take exactly one parameter")
            self.slot_of = {func.params[0]: 0}
            self.depth = 1
        else:
            # callee frame: [dummy(slot0), p1(slot1)..pk(slotk), ret(slot k+1)]
            self.slot_of = {}
            for i, p in enumerate(func.params):
                self.slot_of[p] = i + 1
            self.depth = len(func.params) + 2
        self.compile_tail(func.body)

    def compile(self, program):
        if not program.functions:
            raise CompileError("no function definitions found")
        for fn in program.functions:
            if fn.name in self.funcs:
                raise CompileError("duplicate function %r" % fn.name)
            self.funcs[fn.name] = _Func(fn)
        if 'main' not in self.funcs:
            raise CompileError("program must define a 'main' function")
        # main first (pc 0), then the rest in source order.
        order = [self.funcs['main']]
        for fn in program.functions:
            if fn.name != 'main':
                order.append(self.funcs[fn.name])
        for func in order:
            self.function(func)
        return self._resolve()

    def _resolve(self):
        # first pass: byte offset of every label
        pos = {}
        p = 0
        for it in self.items:
            if isinstance(it, _Label):
                pos[id(it)] = p
            else:
                p += 1
        # second pass: flatten to ints, substituting label offsets
        out = []
        for it in self.items:
            if isinstance(it, _Label):
                continue
            if isinstance(it, _Ref):
                target = pos[id(it.label)]
                if not (0 <= target <= 255):
                    raise CompileError(
                        "jump/call target %d exceeds one byte; program too "
                        "large for the 1-byte address encoding" % target)
                out.append(target)
            else:
                out.append(it)
        return out


def compile_source(source):
    """Compile TLA source text to a bytecode int list (like lang/*.tla.py)."""
    return Compiler().compile(parse(source))


def compile_to_string(source):
    """Compile to the packed byte string accepted by Bytecode(...)."""
    return bc.assemble(compile_source(source))


if __name__ == '__main__':
    import sys
    src = sys.stdin.read() if len(sys.argv) < 2 else open(sys.argv[1]).read()
    code = compile_source(src)
    # Emit a lang/*.tla.py-style module for inspection.
    print "from rpython.jit.tl.threadedcode import tla"
    print "code = ["
    for x in code:
        print "    %d," % x
    print "]"
