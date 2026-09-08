"""@pe.residualize: inlining a marked helper into the PE entry graph."""

from rpython.translator.translator import TranslationContext, graphof
from rpython.translator.backendopt.partialeval import PartialEvaluator
from rpython.rlib import pe

from pypy.interpreter.pe_cogen import inline_residualized

OP_HELPER = 0
OP_PLAIN = 1


def get_graph(fn, signature):
    t = TranslationContext()
    t.buildannotator().build_types(fn, signature)
    t.buildrtyper().specialize()
    return graphof(t, fn), t


def _make_marked():
    @pe.residualize
    def _helper(x):
        return x + 3

    def interp_step(opcode, oparg, pc, value):
        if opcode == OP_HELPER:
            return pc, _helper(value)
        else:
            return pc, value - oparg

    interp_step._pe_static_args_ = ("opcode",)
    interp_step._pe_split_args_ = ("pc",)
    return interp_step


def _make_unmarked():
    def _helper(x):
        return x + 3

    def interp_step(opcode, oparg, pc, value):
        if opcode == OP_HELPER:
            return pc, _helper(value)
        else:
            return pc, value - oparg

    interp_step._pe_static_args_ = ("opcode",)
    interp_step._pe_split_args_ = ("pc",)
    return interp_step


def test_residualize_inlines_marked_helper():
    interp_step = _make_marked()
    graph, t = get_graph(interp_step, [int, int, int, int])

    inlined = inline_residualized(t, graph)
    assert inlined is not graph

    pe_eval = PartialEvaluator(t)
    helper_template = pe_eval.make_symbolic_template(
        OP_HELPER, inlined, {"opcode": OP_HELPER})

    ops = [op.opname for op in helper_template.operations]
    assert "int_add" in ops
    assert "direct_call" not in ops


def test_residualize_leaves_other_opcode_template_unchanged():
    interp_step = _make_marked()
    graph, t = get_graph(interp_step, [int, int, int, int])

    pe_eval = PartialEvaluator(t)
    original_plain = pe_eval.make_symbolic_template(
        OP_PLAIN, graph, {"opcode": OP_PLAIN})

    inlined = inline_residualized(t, graph)
    pe_eval2 = PartialEvaluator(t)
    inlined_plain = pe_eval2.make_symbolic_template(
        OP_PLAIN, inlined, {"opcode": OP_PLAIN})

    def ops(template):
        return [op.opname for op in template.operations]

    assert ops(original_plain) == ops(inlined_plain)


def test_without_decorator_call_survives():
    interp_step = _make_unmarked()
    graph, t = get_graph(interp_step, [int, int, int, int])

    inlined = inline_residualized(t, graph)
    # No callee is marked: the graph is returned untouched, not copied.
    assert inlined is graph

    pe_eval = PartialEvaluator(t)
    helper_template = pe_eval.make_symbolic_template(
        OP_HELPER, inlined, {"opcode": OP_HELPER})
    ops = [op.opname for op in helper_template.operations]
    assert "direct_call" in ops
