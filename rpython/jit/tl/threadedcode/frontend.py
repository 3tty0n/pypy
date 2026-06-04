"""A small frontend (compiler) for the TLA threaded-code language.

It compiles a restricted, side-effect-free subset of *Python* down to TLA
bytecode, so shootout-style benchmarks can be written in readable source code
instead of hand-assembled opcode lists (compare lang/fib.tla.py).

Supported subset
----------------
    def name(p1, p2, ...):       # integer functions, recursion allowed
        if <cmp>:                # if / elif / else; every path must `return`
            ...
        return <expr>            # the only statement that produces a value

    expr      : + - * / %, unary -, integer literals, parameters, calls, ()
    cmp       : <  >  <=  >=  ==  !=   (between two exprs)

The entry function must be called ``main`` and take exactly one parameter --
the integer CLI argument that targettla/run() pushes onto the initial frame.
``main`` is laid out at pc 0 and ends with EXIT; every other function ends with
RET.

Usage
-----
    from rpython.jit.tl.threadedcode.frontend import compile_source
    code = compile_source('''
        def fib(n):
            if n < 2:
                return n
            return fib(n - 1) + fib(n - 2)

        def main(x):
            return fib(x)
    ''')
    # `code` is a list of ints, exactly like the `code = [...]` in lang/*.tla.py

This module is a *build-time* tool (like bytecode.assemble); it runs under
plain CPython/PyPy and is never translated, so it may use `ast`, exceptions,
etc. freely.

Notes on the JIT
----------------
The bytecode comparison ops are non-strict (LT == `<=`, GT == `>=`); the
compiler bridges strict `<`/`>`/`!=` by emitting the complementary op and
swapping the branches, so source semantics are exact.

Self tail-recursion of the form ``return f(<args>)`` where f is the current
function is compiled to the in-frame FRAME_RESET + JUMP loop (the JIT-friendly
shape used by lang/mb_*.tla.py) instead of a fresh CALL_ASSEMBLER frame.  Other
calls use CALL_ASSEMBLER.
"""

import ast

from rpython.jit.tl.threadedcode import bytecode as bc


class CompileError(Exception):
    pass


# --- comparison mapping ------------------------------------------------------
# Bytecode LT dispatches W_Object.le  (a <= b); GT dispatches ge (a >= b).
# Map each source comparison to (bytecode_op, swap_branches).  swap=True means
# the bytecode test is the *negation* of the source test, so the then/else
# branches must be exchanged.
_CMP = {
    ast.LtE: (bc.LT, False),
    ast.GtE: (bc.GT, False),
    ast.Eq:  (bc.EQ, False),
    ast.Lt:  (bc.GT, True),    # a < b   == not (a >= b)
    ast.Gt:  (bc.LT, True),    # a > b   == not (a <= b)
    ast.NotEq: (bc.EQ, True),  # a != b  == not (a == b)
}

_BINOP = {
    ast.Add: bc.ADD,
    ast.Sub: bc.SUB,
    ast.Mult: bc.MUL,
    ast.Div: bc.DIV,
    ast.FloorDiv: bc.DIV,
    ast.Mod: bc.MOD,
}


class _Label(object):
    """A jump/call target whose byte offset is resolved after layout."""
    __slots__ = ('name',)

    def __init__(self, name=''):
        self.name = name


class _Ref(object):
    """A 1-byte placeholder in the instruction stream for a _Label's offset."""
    __slots__ = ('label',)

    def __init__(self, label):
        self.label = label


class _Func(object):
    def __init__(self, node):
        self.node = node
        self.name = node.name
        self.params = [a.id for a in _func_args(node)]
        self.label = _Label(node.name)


def _func_args(node):
    # py2 ast: node.args.args is a list of ast.Name; py3: ast.arg
    out = []
    for a in node.args.args:
        if isinstance(a, ast.Name):
            out.append(a)
        else:                      # py3 ast.arg -> shim with .id
            shim = ast.Name()
            shim.id = a.arg
            out.append(shim)
    return out


class Compiler(object):
    def __init__(self):
        self.items = []            # ints | _Label | _Ref
        self.funcs = {}            # name -> _Func
        self.cur = None            # current _Func being compiled
        self.depth = 0             # operand-stack depth from the frame base
        self.slot_of = {}          # param name -> frame slot index

    # -- low level emit -------------------------------------------------------
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

    # -- stack-depth helpers --------------------------------------------------
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
        slot = self.slot_of[name]
        off = self.depth - slot - 1
        self._op(bc.DUPN); self._byte(off)
        self.depth += 1

    # -- expressions ----------------------------------------------------------
    def expr(self, node):
        if isinstance(node, ast.Num):
            self._push_const(int(node.n))
        elif isinstance(node, ast.Name):
            self._read_var(node.id)
        elif isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
            if isinstance(node.operand, ast.Num):
                self._push_const(-int(node.operand.n))
            else:
                self._push_const(0)
                self.expr(node.operand)
                self._op(bc.SUB); self.depth -= 1
        elif isinstance(node, ast.BinOp):
            if type(node.op) not in _BINOP:
                raise CompileError("unsupported operator %s" % type(node.op).__name__)
            self.expr(node.left)
            self.expr(node.right)
            self._op(_BINOP[type(node.op)]); self.depth -= 1
        elif isinstance(node, ast.Call):
            self._call(node)
        else:
            raise CompileError("unsupported expression: %s" % ast.dump(node))

    def _call(self, node):
        if not isinstance(node.func, ast.Name):
            raise CompileError("only direct calls f(...) are supported")
        name = node.func.id
        if name not in self.funcs:
            raise CompileError("call to unknown function %r" % name)
        callee = self.funcs[name]
        argnum = len(node.args)
        if argnum != len(callee.params):
            raise CompileError("%s expects %d args, got %d"
                               % (name, len(callee.params), argnum))
        # convention: push a dummy (callee slot0), then the args, CALL_ASSEMBLER
        # drops the args and pushes the result, POP1 discards the dummy.
        self._push_const(0)                 # dummy slot0
        for a in node.args:
            self.expr(a)
        self._op(bc.CALL_ASSEMBLER); self._ref(callee.label); self._byte(argnum)
        # CALL_ASSEMBLER drops the argnum args and pushes the call result, so
        # the stack goes [.., dummy, args] -> [.., dummy, result].
        self.depth -= argnum                # args dropped
        self.depth += 1                     # result pushed
        self._op(bc.POP1)                   # discard dummy, keep result on top
        self.depth -= 1

    # -- comparison (for `if`) ------------------------------------------------
    def _emit_compare(self, node):
        """Emit a comparison, returning swap flag (True => negated test)."""
        if not (isinstance(node, ast.Compare) and len(node.ops) == 1):
            raise CompileError("if-condition must be a single comparison")
        op = type(node.ops[0])
        if op not in _CMP:
            raise CompileError("unsupported comparison %s" % op.__name__)
        bcop, swap = _CMP[op]
        self.expr(node.left)
        self.expr(node.comparators[0])
        self._op(bcop); self.depth -= 1     # two operands -> one bool
        return swap

    # -- statements -----------------------------------------------------------
    def block(self, stmts):
        if not stmts:
            raise CompileError("%s: a code path does not return" % self.cur.name)
        head = stmts[0]
        if isinstance(head, ast.Return):
            self._return(head.value)
            # statements after an unconditional return are dead; ignore.
            return
        if isinstance(head, ast.If):
            else_stmts = head.orelse if head.orelse else stmts[1:]
            self._if(head.test, head.body, else_stmts)
            return
        raise CompileError("unsupported statement: %s" % ast.dump(head))

    def _if(self, test, then_stmts, else_stmts):
        depth0 = self.depth
        swap = self._emit_compare(test)        # leaves a bool, consumed by JUMP_IF
        ltrue = _Label('then')
        self._op(bc.JUMP_IF); self._ref(ltrue)
        # The comparison is net stack-neutral: it pushes two operands, the
        # compare op pops both and pushes one bool, and JUMP_IF's is_true pops
        # that bool -- so both branches start at the pre-`if` depth (depth0).
        false_stmts = else_stmts if not swap else then_stmts
        true_stmts = then_stmts if not swap else else_stmts
        self.depth = depth0
        self.block(false_stmts)
        self._place(ltrue)
        self.depth = depth0
        self.block(true_stmts)

    def _return(self, value):
        # self tail-call: return f(same-arity args) where f is the current func
        if (self.cur.name != 'main'
                and isinstance(value, ast.Call)
                and isinstance(value.func, ast.Name)
                and value.func.id == self.cur.name):
            self._tail_self_call(value)
            return
        self.expr(value)
        if self.cur.name == 'main':
            self._op(bc.EXIT)
        else:
            self._op(bc.RET); self._byte(len(self.cur.params))

    def _tail_self_call(self, node):
        # Reuse the current frame instead of recursing: evaluate the new args on
        # top of the stack, then FRAME_RESET (o=l=argnum, n=argnum) shifts them
        # down over the current locals and jumps back to the function entry --
        # the JIT-friendly tail-loop shape of lang/mb_*.tla.py.
        k = len(self.cur.params)
        if len(node.args) != k:
            raise CompileError("tail call arity mismatch in %s" % self.cur.name)
        for a in node.args:
            self.expr(a)
        # stack now: [dummy, p1..pk, ret, newarg1..newargk] (sp = 2k+2).
        # FRAME_RESET o=k, l=0, n=k copies the k new args down onto the k param
        # slots (new_base = sp-o-n-l-1 = 1), preserves the return slot, and
        # sets sp back to k+2 -- the same frame shape we started the call with.
        self._op(bc.FRAME_RESET); self._byte(k); self._byte(0); self._byte(k)
        self.depth -= k
        self._op(bc.JUMP); self._ref(self.cur.body_label)

    # -- function / program ---------------------------------------------------
    def function(self, func):
        self.cur = func
        self._place(func.label)
        body_label = _Label(func.name + ':body')
        func.body_label = body_label
        self._place(body_label)
        if func.name == 'main':
            # entry frame is just [x]; the one param sits at slot 0.
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
        self.block(func.node.body)

    def compile(self, tree):
        funcdefs = [n for n in tree.body if isinstance(n, ast.FunctionDef)]
        if not funcdefs:
            raise CompileError("no function definitions found")
        for n in funcdefs:
            self.funcs[n.name] = _Func(n)
        if 'main' not in self.funcs:
            raise CompileError("program must define a 'main' function")
        # main first (pc 0), then the rest in source order.
        order = [self.funcs['main']]
        for n in funcdefs:
            if n.name != 'main':
                order.append(self.funcs[n.name])
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


def compile_source(src):
    """Compile a source string to a TLA bytecode list (list of ints)."""
    tree = ast.parse(src)
    return Compiler().compile(tree)


def compile_to_string(src):
    """Compile to the packed byte string accepted by Bytecode(...)."""
    return bc.assemble(compile_source(src))


if __name__ == '__main__':
    import sys
    src = sys.stdin.read() if len(sys.argv) < 2 else open(sys.argv[1]).read()
    code = compile_source(src)
    # Emit a lang/*.tla.py-style module for inspection.
    names = {v: k for k, v in enumerate(bc.bytecodes)}
    print "from rpython.jit.tl.threadedcode import tla"
    print "code = ["
    for x in code:
        print "    %d," % x
    print "]"
