import py
import os
from rpython.rlib.parsing.ebnfparse import parse_ebnf, make_parse_function
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

currentdir = os.path.dirname(os.path.abspath(__file__))
grammar = py.path.local(currentdir).join('grammar.txt').read("rt")
regexs, rules, ToAST = parse_ebnf(grammar)
_parse = make_parse_function(regexs, rules, eof=True)


class Transformer(object):
    def _grab_exprs(self, star):
        exprs = []
        while len(star.children) == 2:
            exprs.append(self.visit_expr(star.children[0]))
            star = star.children[1]
        exprs.append(self.visit_expr(star.children[0]))
        return exprs

    def visit_main(self, node):
        exprs = self._grab_exprs(node.children[0])
        return Program(exprs)

    def visit_expr(self, node):
        ch0 = node.children[0]
        ai = getattr(ch0, 'additional_info', None)
        if ai == 'if':
            return If(
                self.visit_expr(node.children[1]),
                self.visit_expr(node.children[3]),
                self.visit_expr(node.children[5]))
        if ai == 'while':
            return While(
                self.visit_simple_expr(node.children[1]),
                self.visit_expr(node.children[3]))
        if ai == 'let':
            varname = node.children[1].additional_info
            rhs = self.visit_expr(node.children[3])
            body = self.visit_expr(node.children[5])
            return LetIn(varname, rhs, body)
        if ai == 'let rec':
            funcname = node.children[1].additional_info
            formal_args = self.visit_formal_args(node.children[2])
            body = self.visit_expr(node.children[4])
            return Function(funcname, formal_args, body)
        if ai == 'Array.make':
            return ArrayMake(
                self.visit_simple_expr(node.children[1]),
                self.visit_simple_expr(node.children[2]))
        if len(node.children) == 1:
            return self.visit_simple_expr(ch0)
        if len(node.children) == 2:
            callee = self.visit_simple_expr(ch0)
            args = self.visit_actual_args(node.children[1])
            return FunApp(callee, args)
        raise NotImplementedError(str(node))

    def visit_simple_expr(self, node):
        children = node.children
        chnode = children[0]
        if getattr(chnode, 'additional_info', None) == '(':
            return self.visit_simple_expr(children[1])
        if (len(children) == 8 and
                getattr(children[1], 'additional_info', None) == '.' and
                getattr(children[5], 'additional_info', None) == '<'):
            return ArrayStore(
                self.visit_atom(children[0]),
                self.visit_simple_expr(children[3]),
                self.visit_simple_expr(children[7]))
        if (len(children) == 5 and
                getattr(children[1], 'symbol', None) == 'DOT'):
            return ArrayLoad(
                self.visit_atom(children[0]),
                self.visit_simple_expr(children[3]))
        if len(children) == 1:
            ch = children[0]
            sym = getattr(ch, 'symbol', None)
            if sym == 'array_load':
                c = ch.children
                return ArrayLoad(
                    self.visit_atom(c[0]),
                    self.visit_simple_expr(c[3]))
            if sym == 'array_store':
                c = ch.children
                return ArrayStore(
                    self.visit_atom(c[0]),
                    self.visit_simple_expr(c[3]),
                    self.visit_simple_expr(c[7]))
            return self.visit_atom(children[0])
        if len(children) == 3:
            op = getattr(children[1], 'additional_info', None)
            if op in ('+', '-', '<', '==', '*', '%', '>'):
                return BinOp(
                    op,
                    self.visit_atom(children[0]),
                    self.visit_simple_expr(children[2]))
        raise NotImplementedError(str(node))

    def visit_formal_args(self, node):
        args = []
        while True:
            args.append(node.children[0].additional_info)
            if len(node.children) == 1:
                break
            node = node.children[1]
        return args

    def visit_actual_args(self, node):
        args = []
        while True:
            args.append(self.visit_simple_expr(node.children[0]))
            if len(node.children) == 1:
                break
            node = node.children[1]
        return args

    def visit_atom(self, node):
        chnode = node.children[0]
        if chnode.symbol == 'DECIMAL':
            return ConstInt(int(chnode.additional_info))
        if chnode.symbol == 'VARIABLE':
            return Variable(chnode.additional_info)
        if chnode.symbol == 'FLOAT':
            return ConstFloat(float(chnode.additional_info))
        raise NotImplementedError


transformer = Transformer()


def parse(source):
    return transformer.visit_main(_parse(source))
