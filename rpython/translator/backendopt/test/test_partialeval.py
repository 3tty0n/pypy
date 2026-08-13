from rpython.flowspace.model import checkgraph, summary
from rpython.translator.translator import TranslationContext, graphof
from rpython.rtyper.llinterp import LLInterpreter
from rpython.rlib.jit import pe_entry_point
from rpython.translator.backendopt.partialeval import (PartialEvaluator,
    specialize_graph, specialize_entry_point, specialize_variant,
    specialize_split_graph, make_rtyped_constant)

def get_graph(fn, signature):
    t = TranslationContext()
    t.buildannotator().build_types(fn, signature)
    t.buildrtyper().specialize()
    return graphof(t, fn), t

def run_graph(graph, t, args):
    checkgraph(graph)
    interp = LLInterpreter(t.rtyper)
    return interp.eval_graph(graph, args)

def to_llvalue(t, var, value):
    s_value = t.annotator.binding(var)
    r_value = t.rtyper.getrepr(s_value)
    return r_value.convert_const(value)


def test_pe_entry_point_split_metadata():
    @pe_entry_point(static=("code",), split=("pc",))
    def dispatch(code, pc, x):
        return x

    assert dispatch._pe_entry_point_
    assert dispatch._pe_static_args_ == ("code",)
    assert dispatch._pe_split_args_ == ("pc",)

def test_specialize_static_integer():
    def f(x, y):
        return x * 2 + y

    f._pe_entry_point_ = True
    f._pe_static_args_ = ("x",)

    graph, t = get_graph(f, [int, int])

    residual = specialize_entry_point(t, graph, {"x": 10})

    assert summary(residual) == {"int_add": 1}


def test_specialize_static_branch():
    def f(x, y):
        if x == 0:
            return y + 1
        else:
            return y * 2

    f._pe_entry_point_ = True
    f._pe_static_args_ = ("x",)

    graph, t = get_graph(f, [int, int])
    residual = specialize_entry_point(t, graph, {"x": 0})
    assert summary(residual) == {"int_add": 1}

    interp = LLInterpreter(t.rtyper)
    assert interp.eval_graph(residual, [999, 10]) == 11


LOAD = 0
ADD = 1
HALT = 2

def test_specialize_static_string_getitem():
    def lookup(code, pc):
        return ord(code[pc])

    lookup._pe_entry_point_ = True
    lookup._pe_static_args_ = ("code", "pc",)

    code = chr(LOAD) + chr(ADD)
    graph, t = get_graph(lookup, [str, int])

    residual = specialize_entry_point(
        t, graph, {"code": code, "pc": 0})

    ops = summary(residual)
    print(ops)

    assert "direct_call" not in ops
    assert "cast_char_to_int" not in ops

def test_specicalize_dispatch_simple_1():
    opcode = chr(LOAD) + chr(ADD)

    def dispatch(opcode, pc, x):
        opcode = ord(opcode[pc])
        if opcode == LOAD:
            return x + 10
        elif opcode == ADD:
            return x + 20
        return -1

    dispatch._pe_entry_point_ = True
    dispatch._pe_static_args_ = ("opcode", "pc",)

    graph, t = get_graph(dispatch, [str, int, int])

    residual = specialize_entry_point(
        t, graph, {"opcode": opcode, "pc": 1}) # Specialize to ADD

    ops = summary(residual)
    assert "int_eq" not in ops
    assert ops == {"int_add": 1}

    interp = LLInterpreter(t.rtyper)
    ll_code = to_llvalue(t, graph.startblock.inputargs[0], opcode)
    assert interp.eval_graph(residual, [ll_code, 999, 1]) == 20 + 1


def test_specialize_dispatch_simple_2():
    code = chr(LOAD) + chr(ADD)

    def dispatch(code, pc, x):
        opcode = ord(code[pc])

        if opcode == LOAD:
            return 1, x + 10
        elif opcode == ADD:
            return 2, x + 20

        return -1, x

    dispatch._pe_entry_point_ = True
    dispatch._pe_static_args_ = ("code",)

    graph, t = get_graph(dispatch, [str, int, int])

    residual = specialize_entry_point(
        t,
        graph,
        {"code": code},
    )

    ops = summary(residual)
    # pc is still dynamic, so dispatch remains for now.
    assert "int_eq" in ops

    ll_code = to_llvalue(t, graph.startblock.inputargs[0], code)
    interp = LLInterpreter(t.rtyper)
    res = interp.eval_graph(residual, [ll_code, 0, 1])
    assert res.item0 == 1 and res.item1 == 11
    res = interp.eval_graph(residual, [ll_code, 1, 1])
    assert res.item0 == 2 and res.item1 == 21


def test_specialize_dispatch_simple_3():
    code = chr(LOAD) + chr(ADD) + chr(HALT)

    def dispatch(code, pc, x):
        opcode = ord(code[pc])

        if opcode == LOAD:
            return 1, x + 10
        elif opcode == ADD:
            return 2, x + 20
        elif opcode == HALT:
            return -1, x
        return -1, x

    dispatch._pe_entry_point_ = True
    dispatch._pe_static_args_ = ("code",)
    dispatch._pe_split_args_ = ("pc",)

    graph, t = get_graph(dispatch, [str, int, int])

    pe = PartialEvaluator(t)
    residual0 = specialize_variant(
        pe, graph, {"code": code}, {"pc": 0})
    residual1 = specialize_variant(
        pe, graph, {"code": code}, {"pc": 1})

    assert residual0 is not residual1
    assert summary(residual0) == {"int_add": 1, "malloc": 1,
                                  "setfield": 2}
    assert summary(residual1) == {"int_add": 1, "malloc": 1,
                                  "setfield": 2}
    assert specialize_variant(
        pe, graph, {"code": code}, {"pc": 0}) is residual0

    ll_code = to_llvalue(t, graph.startblock.inputargs[0], code)
    interp = LLInterpreter(t.rtyper)
    res = interp.eval_graph(residual0, [ll_code, 999, 1])
    assert res.item0 == 1 and res.item1 == 11
    res = interp.eval_graph(residual1, [ll_code, 999, 1])
    assert res.item0 == 2 and res.item1 == 21

    connected = specialize_split_graph(
        PartialEvaluator(t), graph, {"code": code}, {"pc": 0})
    res = interp.eval_graph(connected, [ll_code, 999, 1])
    assert res.item0 == -1 and res.item1 == 31


def test_connect_split_backedge_reuses_variant():
    code = chr(LOAD) + chr(ADD)

    def dispatch(code, pc, x):
        opcode = ord(code[pc])
        if opcode == LOAD:
            return 1, x + 10
        return 0, x + 20

    dispatch._pe_entry_point_ = True
    dispatch._pe_static_args_ = ("code",)
    dispatch._pe_split_args_ = ("pc",)

    graph, t = get_graph(dispatch, [str, int, int])
    pe = PartialEvaluator(t)
    connected = specialize_split_graph(
        pe, graph, {"code": code}, {"pc": 0})

    checkgraph(connected)
    assert len(pe.cache) == 2
    assert summary(connected) == {"int_add": 2}
