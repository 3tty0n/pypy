from rpython.flowspace.model import checkgraph, summary
from rpython.translator.translator import TranslationContext, graphof
from rpython.rtyper.llinterp import LLInterpreter
from rpython.translator.backendopt.partialeval import specialize_graph, specialize_entry_point

def get_graph(fn, signature):
    t = TranslationContext()
    t.buildannotator().build_types(fn, signature)
    t.buildrtyper().specialize()
    return graphof(t, fn), t

def run_graph(graph, t, args):
    checkgraph(graph)
    interp = LLInterpreter(t.rtyper)
    return interp.eval_graph(graph, args)


def test_specialize_static_integer():
    def f(x, y):
        return x * 2 + y

    f._pe_entry_point_ = True
    f._pe_static_args_ = ("x",)

    graph, t = get_graph(f, [int, int])

    residual = specialize_entry_point(t, graph, {"x": 10})

    assert summary(residual) == {"int_add": 1}
