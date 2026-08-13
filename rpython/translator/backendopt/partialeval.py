from rpython.flowspace.model import checkgraph, copygraph, Constant, Variable
from rpython.translator import simplify
from rpython.translator.backendopt.constfold import constant_fold_graph

def find_pe_entrypoints(translator):
    result = []

    for graph in translator.graphs:
        func = getattr(graph, "func", None)
        if func is None:
            continue

        if getattr(graph.func, "_pe_entry_point_", False):
            result.append(graph)

    return result

def replace_uses(graph, replacements):
    for block in graph.iterblocks():
        for op in block.operations:
            op.args = [replacements.get(arg, arg) for arg in op.args]

            if block.exitswitch in replacements:
                block.exitswitch = replacements[block.exitswitch]

            for link in block.exits:
                link.args = [replacements.get(arg, arg) for arg in link.args]


def make_rtyped_constant(translator, var, value):
    rtyper = translator.rtyper

    s_value = translator.annotator.binding(var)
    r_value = rtyper.getrepr(s_value)

    llvalue = r_value.convert_const(value)

    return Constant(llvalue, var.concretetype)

def specialize_entry_point(translator, graph, static_values):
    func = graph.func
    argnames, vararg, kwarg = graph.signature

    declared_static = func._pe_static_args_

    static_args = {}

    for name in declared_static:
        if name not in static_values:
            raise Exception("missing static value for PE argument %r" % (name,))
        index = argnames.index(name)
        var = graph.startblock.inputargs[index]
        const = make_rtyped_constant(translator, var, static_values[name])
        static_args[index] = const

    return specialize_graph(translator, graph, static_args)

def specialize_graph(translator, graph, static_args):
    residual = copygraph(graph)
    inputargs = residual.startblock.inputargs
    replacements = {}

    for index, const in static_args.items():
        replacements[inputargs[index]] = const

    replace_uses(residual, replacements)

    constant_fold_graph(residual)
    simplify.cleanup_graph(residual)
    checkgraph(residual)

    return residual


def partial_evaluate(translator, static_program):
    graphs = find_pe_entrypoints(translator)

    for graph in graphs:
        specialize_entry_point(translator, graph)
        simplify.cleanup_graph(graph)
        checkgraph(graph)
