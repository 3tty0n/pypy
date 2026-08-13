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


def _is_safe_pe_callable(fn):
    return (
        fn.__name__.startswith("ll_stritem")
    )

def try_fold_static_calls(translator, op):
    if op.opname != "direct_call":
        return None

    if not all(isinstance(arg, Constant) for arg in op.args):
        return None

    fnptr = op.args[0].value
    funcobj = fnptr._obj
    callable = funcobj._callable

    if not _is_safe_pe_callable(callable):
        return None

    args = [arg.value for arg in op.args[1:]]
    result = callable(*args)
    return Constant(result, op.result.concretetype)


def fold_static_calls(translator, graph):
    replacements = {}
    for block in graph.iterblocks():
        newops = []
        for op in block.operations:
            op.args = [replacements.get(arg, arg) for arg in op.args]

            result = try_fold_static_calls(translator, op)

            if result is not None:
                replacements[op.result] = result
            else:
                newops.append(op)

            block.operations = newops

            block.exitswitch = replacements.get(
                block.exitswitch, block.exitswitch)

            for link in block.exits:
                link.args = [replacements.get(arg, arg) for arg in link.args]

def specialize_entry_point(translator, graph, static_values):
    func = graph.func
    argnames, vararg, kwarg = graph.signature
    assert vararg is None
    assert kwarg is None

    indexed_values = {}

    declared = func._pe_static_args_

    for name in declared:
        if name not in static_values:
            raise ValueError("missing static value for PE argument %r" % (name,))

        try:
            index = argnames.index(name)
        except ValueError:
            raise ValueError("unknown PE argument %r: arguments are %r" % (name, argnames))

        indexed_values[index] = static_values[name]

    return specialize_graph(translator, graph, indexed_values)

def specialize_graph(translator, graph, indexed_values):
    static_constants = {}

    for index, value in indexed_values.items():
        original_var = graph.startblock.inputargs[index]

        static_constants[index] = make_rtyped_constant(
            translator,
            original_var,
            value,
        )

    residual = copygraph(graph)
    replacements = {}

    for index, const in static_constants.items():
        residual_var = residual.startblock.inputargs[index]
        replacements[residual_var] = const

    replace_uses(residual, replacements)

    fold_static_calls(translator, residual)

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
