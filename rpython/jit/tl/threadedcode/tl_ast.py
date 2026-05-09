"""AST for the threaded TL mini-language (RPython-friendly; no py/ebnf)."""


class Node(object):
    def __eq__(self, other):
        if self.__class__ is not other.__class__:
            return False
        return self.__dict__ == other.__dict__

    def __ne__(self, other):
        return not self == other


class Program(Node):
    def __init__(self, exprs):
        self.exprs = exprs


class ConstInt(Node):
    def __init__(self, intval):
        self.intval = intval


class ConstFloat(Node):
    def __init__(self, floatval):
        self.floatval = floatval


class Variable(Node):
    def __init__(self, val):
        self.val = val


class BinOp(Node):
    def __init__(self, op, left, right):
        self.op = op
        self.left = left
        self.right = right


class LetIn(Node):
    def __init__(self, name, rhs, body):
        self.name = name
        self.rhs = rhs
        self.body = body


class ArrayMake(Node):
    def __init__(self, size_expr, init_expr):
        self.size_expr = size_expr
        self.init_expr = init_expr


class ArrayLoad(Node):
    def __init__(self, array_expr, index_expr):
        self.array_expr = array_expr
        self.index_expr = index_expr


class ArrayStore(Node):
    def __init__(self, array_expr, index_expr, value_expr):
        self.array_expr = array_expr
        self.index_expr = index_expr
        self.value_expr = value_expr


class Function(Node):
    def __init__(self, funcname, args, body):
        self.funcname = funcname
        self.args = args
        self.body = body


class FunApp(Node):
    def __init__(self, callee, args):
        self.callee = callee
        self.args = args
