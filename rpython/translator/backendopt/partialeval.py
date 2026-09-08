"""Offline PE: specialize and connect residual graphs for PE entry points."""
from rpython.flowspace.model import (c_last_exception, checkgraph, copygraph,
                                     Constant, Link, Variable)
from rpython.rlib.objectmodel import compute_hash
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


def _pe_argument_names(graph):
    static_names = graph.func._pe_static_args_
    split_names = getattr(graph.func, "_pe_split_args_", ())
    return static_names, split_names


def partial_evaluate(translator, static_program):
    """Specialize and install all declared PE entry points."""
    graphs = find_pe_entrypoints(translator)
    pe = PartialEvaluator(translator)
    installed = []

    for graph in graphs:
        if graph in static_program:
            entry = static_program[graph]
        elif graph.func in static_program:
            entry = static_program[graph.func]
        else:
            raise ValueError("missing PE values for graph %r" % (graph,))

        if isinstance(entry, tuple):
            static_env, split_env = entry
        else:
            static_env, split_env = entry, {}

        _, split_names = _pe_argument_names(graph)
        if split_names:
            installed_graph = pe.install_split_graph(
                graph, static_env, split_env)
        else:
            installed_graph = pe.install_graph(graph, static_env)
        simplify.cleanup_graph(installed_graph)
        checkgraph(installed_graph)
        installed.append(installed_graph)
    return installed


def replace_uses(graph, replacements):
    for block in graph.iterblocks():
        for op in block.operations:
            op.args = [replacements.get(arg, arg) for arg in op.args]

        if block.exitswitch in replacements:
            block.exitswitch = replacements[block.exitswitch]
            _resolve_constant_switch(block)

        for link in block.exits:
            link.args = [replacements.get(arg, arg) for arg in link.args]


def _resolve_constant_switch(block):
    """Collapse a switch whose selector just became a constant."""
    switch = block.exitswitch
    if not isinstance(switch, Constant) or switch is c_last_exception:
        return
    from rpython.translator.backendopt.jitcode_emitter import HoleConstant
    # A HoleConstant here would fold on the sentinel instead of a real value.
    assert not isinstance(switch, HoleConstant)
    if block.exits[-1].exitcase == "default":
        default, candidates = block.exits[-1], block.exits[:-1]
    else:
        default, candidates = None, block.exits
    for taken in candidates:
        if taken.llexitcase == switch.value:
            break
    else:
        if default is None:
            return
        taken = default
    block.exitswitch = None
    taken.exitcase = None
    taken.llexitcase = None
    block.recloseblock(taken)


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
        if block.operations:
            newops = []
            for op in block.operations:
                op.args = [replacements.get(arg, arg) for arg in op.args]

                result = try_fold_static_calls(translator, op)

                if result is not None:
                    replacements[op.result] = result
                else:
                    newops.append(op)

            block.operations = newops

        if block.exitswitch in replacements:
            block.exitswitch = replacements[block.exitswitch]
            _resolve_constant_switch(block)

        for link in block.exits:
            link.args = [replacements.get(arg, arg) for arg in link.args]

def specialize_entry_point(translator, graph, static_values):
    return PartialEvaluator(translator).specialize(graph, static_values, {})


def _argument_values(graph, declared, values, kind):
    argnames, vararg, kwarg = graph.signature
    assert vararg is None
    assert kwarg is None

    indexed_values = {}

    for name in declared:
        if name not in values:
            raise ValueError("missing %s value for PE argument %r" %
                             (kind, name))

        try:
            index = argnames.index(name)
        except ValueError:
            raise ValueError(
                "unknown PE argument %r: arguments are %r" %
                (name, argnames))

        indexed_values[index] = values[name]

    return indexed_values


class CacheValue(object):
    """Hash a specialization value with the RPython hashing primitive."""
    def __init__(self, value):
        self.value = value

    def __hash__(self):
        return compute_hash(self.value)

    def __eq__(self, other):
        if type(self.value) is not type(other.value):
            return False
        if isinstance(self.value,
                      (str, unicode, int, long, float, tuple)):
            return self.value == other.value
        if self.value is None:
            return True
        return self.value is other.value


class PartialEvaluator(object):
    """Create and cache one residual graph per PE specialization state."""

    def __init__(self, translator):
        self.translator = translator
        self.cache = {}

    def make_key(self, graph, static_env, split_env):
        static_names, split_names = _pe_argument_names(graph)
        return (
            graph,
            tuple((name, CacheValue(static_env[name]))
                  for name in static_names),
            tuple((name, CacheValue(split_env[name]))
                  for name in split_names),
        )

    def specialize(self, graph, static_env, split_env):
        indexed_values = self._specialization_values(
            graph, static_env, split_env)

        key = self.make_key(graph, static_env, split_env)
        if key in self.cache:
            return self.cache[key]

        residual = copygraph(graph)
        # Register before recursing, so backedges reuse this variant.
        self.cache[key] = residual
        try:
            _specialize_copied_graph(
                self.translator, graph, residual, indexed_values)
        except Exception:
            del self.cache[key]
            raise
        return residual

    def _specialization_values(self, graph, static_env, split_env):
        static_names, split_names = _pe_argument_names(graph)
        overlap = set(static_names).intersection(split_names)
        if overlap:
            raise ValueError(
                "PE arguments cannot be both static and split: %r" %
                (sorted(overlap),))

        indexed_values = _argument_values(
            graph, static_names, static_env, "static")
        indexed_values.update(_argument_values(
            graph, split_names, split_env, "split"))
        return indexed_values

    def specialize_split_graph(self, graph, static_env, split_env,
                               terminal_values=(-1,)):
        _, split_names = _pe_argument_names(graph)
        if len(split_names) != 1:
            raise ValueError(
                "split graph connection requires one split argument")

        residual = self.specialize(graph, static_env, split_env)
        connector = _SplitGraphConnector(
            self, graph, static_env, split_names[0], terminal_values,
            residual.returnblock, residual.exceptblock)
        connector.connect(residual)
        checkgraph(residual)
        return residual

    def install_split_graph(self, graph, static_env, split_env,
                            terminal_values=(-1,)):
        """Install a connected residual CFG in the original graph object."""
        residual = self.specialize_split_graph(
            graph, static_env, split_env, terminal_values)
        return self._install_residual(graph, residual)

    def install_graph(self, graph, static_env):
        residual = self.specialize(graph, static_env, {})
        return self._install_residual(graph, residual)

    def make_template(self, key, graph, static_env, split_env,
                      terminal_values=(-1,)):
        """Build the first-stage template IR for one residual variant."""
        from rpython.translator.backendopt.partialeval_template import (
            ResidualTemplateGenerator)
        residual = self.specialize(graph, static_env, split_env)
        transitions = _find_split_transitions(residual)
        generator = ResidualTemplateGenerator(terminal_values)
        return generator.from_residual_graph(key, residual, transitions)

    def make_symbolic_template(self, key, graph, static_env,
                               terminal_values=(-1,), pc_name=None,
                               oparg_name="oparg"):
        """Specialize offline-static inputs and lift late-static pc values."""
        from rpython.translator.backendopt.partialeval_template import (
            ResidualTemplateGenerator)
        _, split_names = _pe_argument_names(graph)
        if pc_name is None:
            if not split_names:
                raise ValueError(
                    "symbolic templates require at least one split argument")
            pc_name = split_names[0]
        # Further split args are late-static state, resolved per block like pc.
        state_names = tuple(name for name in split_names if name != pc_name)
        static_names, _ = _pe_argument_names(graph)
        indexed_values = _argument_values(
            graph, static_names, static_env, "static")
        residual = copygraph(graph)
        _specialize_copied_graph(
            self.translator, graph, residual, indexed_values)
        transitions = _find_split_transitions(residual)
        generator = ResidualTemplateGenerator(terminal_values)
        hole_names = getattr(graph.func, "_pe_hole_args_", ("oparg2",))
        return generator.from_symbolic_residual_graph(
            key, residual, transitions, pc_name, oparg_name,
            extra_oparg_names=hole_names, state_names=state_names)

    def _install_residual(self, graph, residual):
        graph.startblock = residual.startblock
        graph.returnblock = residual.returnblock
        graph.exceptblock = residual.exceptblock
        if graph not in self.translator.graphs:
            self.translator.graphs.append(graph)
        checkgraph(graph)
        return graph


# Thin forwarders onto PartialEvaluator, kept for existing test call sites.
def specialize_variant(pe, graph, static_values, split_values):
    return pe.specialize(graph, static_values, split_values)


def specialize_split_graph(pe, graph, static_values, split_values,
                           terminal_values=(-1,)):
    return pe.specialize_split_graph(
        graph, static_values, split_values, terminal_values)


def install_split_graph(pe, graph, static_values, split_values,
                        terminal_values=(-1,)):
    return pe.install_split_graph(
        graph, static_values, split_values, terminal_values)


class _SplitTransition(object):
    def __init__(self, block, result_var, fields):
        self.block = block
        self.result_var = result_var
        self.fields = fields

    def constant_next_value(self):
        value = self.fields["item0"]
        if isinstance(value, Constant):
            return value.value
        return None

    def dynamic_values(self, skip=0):
        """Residual values passed to the successor, minus late-static ones."""
        result = []
        index = 1
        while "item%d" % index in self.fields:
            if not (2 <= index < 2 + skip):
                result.append(self.fields["item%d" % index])
            index += 1
        return result

    def state_values(self, count):
        return [self.fields["item%d" % (2 + offset)]
                for offset in range(count)]


def _find_split_transitions(graph):
    transitions = []
    for block in graph.iterblocks():
        if (len(block.exits) != 1 or
                block.exits[0].target is not graph.returnblock):
            continue
        linkargs = block.exits[0].args
        if len(linkargs) != 1 or not isinstance(linkargs[0], Variable):
            continue
        tuple_var = linkargs[0]
        fields = {}
        found_malloc = False
        for op in block.operations:
            if op.result is tuple_var and op.opname == "malloc":
                found_malloc = True
            elif (op.opname == "setfield" and
                  op.args[0] is tuple_var and
                  isinstance(op.args[1], Constant)):
                fields[op.args[1].value] = op.args[2]
        if found_malloc and "item0" in fields:
            transitions.append(_SplitTransition(block, tuple_var, fields))
    return transitions


class _SplitGraphConnector(object):
    """Turn static ``next_split`` results into residual CFG edges."""

    def __init__(self, pe, graph, static_env, split_name, terminal_values,
                 final_returnblock, final_exceptblock):
        self.pe = pe
        self.graph = graph
        self.static_env = static_env
        self.split_name = split_name
        self.terminal_values = terminal_values
        self.final_returnblock = final_returnblock
        self.final_exceptblock = final_exceptblock
        self.connected_edges = set()

    def connect(self, residual):
        self._connect_to_final_exception(residual)
        for transition in _find_split_transitions(residual):
            self._connect_transition(residual, transition)

    def _connect_transition(self, residual, transition):
        next_value = transition.constant_next_value()
        if next_value is None:
            return
        if next_value in self.terminal_values:
            self._connect_to_final_return(residual, transition)
            return

        next_split_env = {self.split_name: next_value}
        successor = self.pe.specialize(
            self.graph, self.static_env, next_split_env)

        edge = (residual, next_value)
        if edge not in self.connected_edges:
            # Mark before recursing: a successor may lead back here.
            self.connected_edges.add(edge)
            self.connect(successor)

        linkargs = self._successor_arguments(
            transition, next_split_env)
        self._remove_result_tuple(transition)
        transition.block.recloseblock(
            Link(linkargs, successor.startblock))

    def _connect_to_final_return(self, residual, transition):
        if residual.returnblock is self.final_returnblock:
            return
        return_value = list(transition.block.exits[0].args)
        transition.block.recloseblock(
            Link(return_value, self.final_returnblock))

    def _connect_to_final_exception(self, residual):
        if residual.exceptblock is self.final_exceptblock:
            return
        for block in list(residual.iterblocks()):
            for link in block.exits:
                if link.target is residual.exceptblock:
                    link.target = self.final_exceptblock

    def _successor_arguments(self, transition, next_split_env):
        static_names, split_names = _pe_argument_names(self.graph)
        argnames = self.graph.signature[0]
        dynamic_indexes = [
            index for index, name in enumerate(argnames)
            if name not in static_names and name not in split_names
        ]
        dynamic_values = self._dynamic_result_values(
            transition, dynamic_indexes)

        linkargs = []
        for index, name in enumerate(argnames):
            if name in static_names:
                value = self._constant_argument(index, self.static_env[name])
            elif name in split_names:
                value = self._constant_argument(index, next_split_env[name])
            else:
                value = dynamic_values[index]
            linkargs.append(value)
        return linkargs

    def _dynamic_result_values(self, transition, dynamic_indexes):
        if len(transition.fields) - 1 != len(dynamic_indexes):
            raise ValueError(
                "split result must be (next_split, dynamic_args...)")

        result = {}
        for position, index in enumerate(dynamic_indexes):
            field_name = "item%d" % (position + 1)
            result[index] = transition.fields[field_name]
        return result

    def _constant_argument(self, index, value):
        original_var = self.graph.startblock.inputargs[index]
        return make_rtyped_constant(self.pe.translator, original_var, value)

    def _remove_result_tuple(self, transition):
        tuple_var = transition.result_var
        transition.block.operations = [
            op for op in transition.block.operations
            if not (op.result is tuple_var or
                    (op.opname == "setfield" and op.args[0] is tuple_var))
        ]


def specialize_graph(translator, graph, indexed_values):
    residual = copygraph(graph)
    _specialize_copied_graph(translator, graph, residual, indexed_values)
    return residual


def _specialize_copied_graph(translator, graph, residual, indexed_values):
    static_constants = {}

    for index, value in indexed_values.items():
        original_var = graph.startblock.inputargs[index]

        static_constants[index] = make_rtyped_constant(
            translator,
            original_var,
            value,
        )

    replacements = {}

    for index, const in static_constants.items():
        residual_var = residual.startblock.inputargs[index]
        replacements[residual_var] = const

    replace_uses(residual, replacements)

    fold_static_calls(translator, residual)

    constant_fold_graph(residual)
    simplify.cleanup_graph(residual)
    checkgraph(residual)
