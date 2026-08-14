"""Intermediate representation for offline-generated residual templates.

The objects in this module describe code-independent work completed by the
offline partial evaluator.  They deliberately contain no runtime linker or
JitCode details.
"""

from rpython.flowspace.model import Constant


class TemplateHole(object):
    """A typed value supplied while a concrete code object is linked."""

    def __init__(self, kind, name=None):
        self.kind = kind
        self.name = name or kind

    def __repr__(self):
        return "%s(%r)" % (self.__class__.__name__, self.name)

    def resolve(self, bindings):
        return bindings[self.name]


class PcHole(TemplateHole):
    def __init__(self):
        TemplateHole.__init__(self, "pc")


class OpargHole(TemplateHole):
    def __init__(self):
        TemplateHole.__init__(self, "oparg")


class NextPcHole(TemplateHole):
    def __init__(self, pc, instruction_size):
        TemplateHole.__init__(self, "next-pc")
        self.pc = pc
        self.instruction_size = instruction_size

    def resolve(self, bindings):
        return bindings[self.pc.name] + self.instruction_size


class CodeConstHole(TemplateHole):
    def __init__(self, index):
        TemplateHole.__init__(self, "code-constant", "const[%s]" % index)
        self.index = index

    def resolve(self, bindings):
        return bindings["code"].co_consts[bindings[self.index.name]]


class AbsoluteTarget(object):
    """A bytecode target stored directly in ``oparg``."""

    def __init__(self, oparg):
        self.oparg = oparg

    def resolve(self, bindings):
        return bindings[self.oparg.name]


class RelativeTarget(object):
    """A bytecode target relative to the following instruction."""

    def __init__(self, next_pc, oparg):
        self.next_pc = next_pc
        self.oparg = oparg

    def resolve(self, bindings):
        return (self.next_pc.resolve(bindings) +
                bindings[self.oparg.name])


class Continue(object):
    """Continue execution at a late-static or concrete split value."""

    def __init__(self, target, dynamic_values):
        self.target = target
        self.dynamic_values = tuple(dynamic_values)


class Finish(object):
    """Return from the residual program."""

    def __init__(self, values):
        self.values = tuple(values)


class Branch(object):
    """Keep a dynamic condition with two late-static bytecode targets."""

    def __init__(self, condition, true_target, false_target, dynamic_values):
        self.condition = condition
        self.true_target = true_target
        self.false_target = false_target
        self.dynamic_values = tuple(dynamic_values)


class ResidualTemplate(object):
    def __init__(self, key, operations, holes, terminators,
                 residual_graph=None):
        self.key = key
        self.operations = tuple(operations)
        self.holes = tuple(holes)
        self.terminators = tuple(terminators)
        # Kept as an offline-only lowering source.  Runtime linking only uses
        # the immutable template data above; translation-time lowering can use
        # the graph to preserve arbitrary residual control flow and exception
        # edges without reconstructing it from a flat operation list.
        self.residual_graph = residual_graph

    def resolve_targets(self, bindings):
        """Resolve only late-static control targets for a code instance."""
        targets = []
        for terminator in self.terminators:
            if isinstance(terminator, Branch):
                targets.append((
                    _resolve_target(terminator.true_target, bindings),
                    _resolve_target(terminator.false_target, bindings),
                ))
            else:
                target = getattr(terminator, "target", None)
                targets.append(_resolve_target(target, bindings))
        return targets


def _resolve_target(target, bindings):
    if hasattr(target, "resolve"):
        return target.resolve(bindings)
    return target


class ResidualTemplateCatalog(object):
    def __init__(self):
        self._templates = {}

    def add(self, template):
        if template.key in self._templates:
            raise ValueError("duplicate residual template %r" %
                             (template.key,))
        self._templates[template.key] = template

    def lookup(self, key):
        return self._templates[key]

    def keys(self):
        return self._templates.keys()


class LinkedTemplateBlock(object):
    def __init__(self, pc, opcode, oparg, template, bindings=None):
        self.pc = pc
        self.opcode = opcode
        self.oparg = oparg
        self.template = template
        self.bindings = bindings or {"pc": pc, "oparg": oparg}
        self.successors = []
        self.has_finish = False
        self.is_loop_header = False


class LinkedResidualProgram(object):
    def __init__(self, entry_pc, blocks):
        self.entry_pc = entry_pc
        self.blocks = blocks
        self.backedges = ()
        self.loop_headers = ()

    def analyze_loops(self):
        dominators = _compute_dominators(self.entry_pc, self.blocks)
        backedges = []
        for source, block in self.blocks.items():
            for target in block.successors:
                if target in dominators[source]:
                    backedges.append((source, target))

        loop_headers = sorted(set(target for source, target in backedges))
        for pc in loop_headers:
            self.blocks[pc].is_loop_header = True
        self.backedges = tuple(sorted(backedges))
        self.loop_headers = tuple(loop_headers)
        return self

    def metadata(self, entry_positions=None):
        metadata = {
            "entry_pc": self.entry_pc,
            "block_pcs": tuple(sorted(self.blocks)),
            "loop_headers": self.loop_headers,
            "backedges": self.backedges,
        }
        if entry_positions is not None:
            metadata["entry_positions"] = entry_positions
        return metadata

    def attach_to_jitcode(self, jitcode, entry_positions=None):
        """Preserve offline CFG facts for the meta-interpreter."""
        from rpython.jit.codewriter.jitcode import PEJitCodeMetadata
        if entry_positions is None:
            entry_positions = {}
        entry_pcs = tuple(sorted(entry_positions))
        positions = tuple(entry_positions[pc] for pc in entry_pcs)
        sources = tuple(source for source, target in self.backedges)
        targets = tuple(target for source, target in self.backedges)
        jitcode.pe_metadata = PEJitCodeMetadata(
            self.entry_pc, tuple(sorted(self.blocks)), self.loop_headers,
            sources, targets, entry_pcs, positions)
        return jitcode

    def lower(self, codewriter, name="offline-residual", portal_jd=None):
        """Lower all linked blocks as one graph and one relocatable JitCode.

        This is deliberately a translation-time operation.  The resulting
        JitCode, including its concrete entry offsets, can be embedded in the
        translated runtime; no FunctionGraph copying or register allocation is
        needed when the code is executed.
        """
        return LinkedResidualLowerer(codewriter).lower(
            self, name, portal_jd=portal_jd)


class LoweredResidualProgram(object):
    def __init__(self, graph, jitcode, entry_positions):
        self.graph = graph
        self.jitcode = jitcode
        self.entry_positions = entry_positions


class LinkedResidualLowerer(object):
    """Turn a linked template CFG into a single codewriter JitCode."""

    def __init__(self, codewriter):
        self.codewriter = codewriter

    def lower(self, program, name="offline-residual", portal_jd=None):
        self.portal_jd = portal_jd
        graph, entry_blocks = self._make_graph(program, name)
        jitcode, entry_positions = self._assemble(
            graph, entry_blocks, name, portal_jd)
        program.attach_to_jitcode(jitcode, entry_positions)
        return LoweredResidualProgram(graph, jitcode, entry_positions)

    def _make_graph(self, program, name):
        from rpython.flowspace.model import (checkgraph, Constant,
                                             FunctionGraph, Link, copygraph)
        from rpython.translator.backendopt.partialeval import (
            _find_split_transitions, replace_uses)

        instances = {}
        for pc, linked_block in program.blocks.items():
            source = linked_block.template.residual_graph
            if source is None:
                raise ValueError(
                    "template %r has no residual graph for lowering" %
                    (linked_block.template.key,))
            graph = copygraph(source)
            replacements = self._late_static_replacements(
                graph, linked_block)
            replace_uses(graph, replacements)
            self._remove_runtime_loop_markers(graph)
            transitions = _find_split_transitions(graph)
            terminators = linked_block.template.terminators
            if len(terminators) == 1 and len(transitions) > 1:
                # RTyping may retain equivalent syntactic fallthrough exits
                # after the symbolic template has canonicalized them.
                terminators = terminators * len(transitions)
            if len(transitions) != len(terminators):
                raise ValueError(
                    "template %r changed transition shape during lowering "
                    "(%d graph exits, %d terminators)" %
                    (linked_block.template.key, len(transitions),
                     len(terminators)))
            instances[pc] = (graph, transitions, terminators)

        entry_graph = instances[program.entry_pc][0]
        final_return = entry_graph.returnblock
        if self.portal_jd is not None:
            from rpython.flowspace.model import Block, Variable
            portal_result = self.portal_jd.portal_graph.returnblock.inputargs[0]
            result = Variable("result")
            result.concretetype = portal_result.concretetype
            final_return = Block([result])
            final_return.operations = ()
            final_return.exits = ()
        final_except = entry_graph.exceptblock

        for pc, linked_block in program.blocks.items():
            graph, transitions, terminators = instances[pc]
            self._merge_exception_block(graph, final_except)
            bindings = linked_block.bindings
            targets = linked_block.template.resolve_targets(bindings)
            if len(targets) == 1 and len(terminators) > 1:
                targets = targets * len(terminators)
            for transition, terminator, target in zip(
                    transitions, terminators, targets):
                if isinstance(terminator, Finish):
                    if self.portal_jd is None:
                        transition.block.exits[0].target = final_return
                    else:
                        result = transition.fields["item1"]
                        self._remove_result_tuple(transition)
                        transition.block.recloseblock(
                            Link([result], final_return))
                    continue
                if isinstance(terminator, Branch):
                    self._connect_branch(
                        graph, transition, terminator, target, instances,
                        program, linked_block)
                    continue
                successor_graph = instances[target][0]
                args = self._successor_arguments(
                    successor_graph, instances[target][0].signature[0],
                    program.blocks[target], self._runtime_values(transition),
                    self._available_carried_values(
                        graph, transition.block))
                self._remove_result_tuple(transition)
                transition.block.recloseblock(
                    Link(args, successor_graph.startblock))

        entry_blocks = dict(
            (pc, instances[pc][0].startblock) for pc in instances)
        if self.portal_jd is None:
            startblock = entry_graph.startblock
        else:
            startblock = self._make_entry_wrapper(
                entry_graph, program.blocks[program.entry_pc])
        graph = FunctionGraph(name, startblock)
        graph.returnblock = final_return
        graph.exceptblock = final_except
        checkgraph(graph)
        return graph, entry_blocks

    def _make_entry_wrapper(self, entry_graph, linked_block):
        from rpython.flowspace.model import Block, Constant, Link, Variable
        runtime_names = ("self", "bytecode")
        original = self._named_start_arguments(entry_graph)
        inputs = []
        runtime = {}
        for name in runtime_names:
            source = original[name]
            value = Variable(name)
            value.concretetype = source.concretetype
            inputs.append(value)
            runtime[name] = value
        wrapper = Block(inputs)
        static = dict(linked_block.bindings)
        static["opcode"] = linked_block.opcode
        args = []
        for name, target in zip(entry_graph.signature[0],
                                entry_graph.startblock.inputargs):
            if name in runtime:
                args.append(runtime[name])
            else:
                args.append(Constant(static[name], target.concretetype))
        wrapper.closeblock(Link(args, entry_graph.startblock))
        return wrapper

    def _connect_branch(self, graph, transition, terminator, targets,
                        instances, program, linked_block):
        """Bypass a return-tuple phi for a late-static dynamic branch.

        Flow graphs commonly represent ``condition ? jump_pc : next_pc`` by
        branching first and joining both values in the block which allocates
        the result tuple.  At template-link time both pc alternatives are
        known.  Redirect each incoming link to the corresponding specialized
        block while retaining the residual condition which selects the link.
        """
        from rpython.flowspace.model import Constant

        split_value = transition.fields["item0"]
        definitions = dict((op.result, op) for op in
                           transition.block.operations)
        stored = {}
        for op in transition.block.operations:
            if (op.opname == "setfield" and
                    isinstance(op.args[1], Constant)):
                stored[op.args[0], op.args[1].value] = op.args[2]
        while split_value in definitions:
            definition = definitions[split_value]
            if definition.opname == "same_as":
                split_value = definition.args[0]
            elif (definition.opname == "getfield" and
                  isinstance(definition.args[1], Constant) and
                  (definition.args[0], definition.args[1].value) in stored):
                split_value = stored[
                    definition.args[0], definition.args[1].value]
            else:
                break
        try:
            split_index = transition.block.inputargs.index(split_value)
        except ValueError:
            raise ValueError(
                "Branch pc %r is not a phi input %r; ops=%r" %
                (split_value, transition.block.inputargs,
                 transition.block.operations))

        phi_block = transition.block
        incoming = self._incoming_links(graph, phi_block)
        while len(incoming) == 1:
            predecessor, link = incoming[0]
            split_value = link.args[split_index]
            predecessor_defs = dict((op.result, op)
                                    for op in predecessor.operations)
            while (split_value in predecessor_defs and
                   predecessor_defs[split_value].opname == "same_as"):
                split_value = predecessor_defs[split_value].args[0]
            if split_value not in predecessor.inputargs:
                break
            phi_block = predecessor
            split_index = predecessor.inputargs.index(split_value)
            incoming = self._incoming_links(graph, phi_block)
        if len(incoming) != 2:
            raise ValueError("Branch requires exactly two residual exits")

        expected = set(targets)
        seen = set()
        dynamic = self._runtime_values(transition)
        all_definitions = {}
        all_stored = {}
        for block in graph.iterblocks():
            for op in block.operations:
                all_definitions[op.result] = op
                if (op.opname == "setfield" and
                        isinstance(op.args[1], Constant)):
                    all_stored[op.args[0], op.args[1].value] = op.args[2]
        origins = _variable_origins(graph)
        named = self._named_start_arguments(graph)
        input_indexes = dict((var, index) for index, var in
                             enumerate(transition.block.inputargs))
        for predecessor, link in incoming:
            target_value = link.args[split_index]
            while (target_value in all_definitions and
                   all_definitions[target_value].opname == "same_as"):
                target_value = all_definitions[target_value].args[0]
            if not isinstance(target_value, Constant):
                origin = origins.get(target_value, target_value)
                variable_name = getattr(origin, "_name", "")
                if variable_name.startswith("pc"):
                    resolved = linked_block.bindings["pc"]
                elif variable_name.startswith("oparg"):
                    resolved = linked_block.bindings["oparg"]
                else:
                    lifted = _lift_target(
                        target_value, all_definitions, all_stored, origins,
                        named.get("pc"), named.get("oparg"), PcHole(),
                        OpargHole())
                    resolved = _resolve_target(
                        lifted, linked_block.bindings)
                if isinstance(resolved, int):
                    target_value = Constant(resolved,
                                            target_value.concretetype)
            if not isinstance(target_value, Constant):
                if link.exitcase is True:
                    target = targets[0]
                elif link.exitcase is False:
                    target = targets[1]
                else:
                    raise ValueError(
                        "Branch target did not become late-static")
            else:
                target = target_value.value
            if target not in expected:
                raise ValueError("unexpected Branch target %r" % (target,))
            seen.add(target)
            successor_graph = instances[target][0]
            branch_dynamic = []
            for value in dynamic:
                index = input_indexes.get(value, -1)
                if index >= 0:
                    value = link.args[index]
                branch_dynamic.append(value)
            args = self._successor_arguments(
                successor_graph, successor_graph.signature[0],
                program.blocks[target], branch_dynamic,
                self._available_carried_values(graph, predecessor))
            link.args = args
            link.target = successor_graph.startblock
        if seen != expected:
            raise ValueError("Branch residual exits do not cover both targets")

    def _incoming_links(self, graph, target):
        incoming = []
        for block in graph.iterblocks():
            for link in block.exits:
                if link.target is target:
                    incoming.append((block, link))
        return incoming

    def _late_static_replacements(self, graph, linked_block):
        from rpython.flowspace.model import Constant
        values = dict(linked_block.bindings)
        values["opcode"] = linked_block.opcode
        replacements = {}
        for name, var in zip(graph.signature[0], graph.startblock.inputargs):
            if name in values:
                replacements[var] = Constant(values[name], var.concretetype)
        return replacements

    def _named_start_arguments(self, graph):
        return dict(zip(graph.signature[0], graph.startblock.inputargs))

    def _remove_runtime_loop_markers(self, graph):
        """The linked CFG already carries its loop structure offline."""
        from rpython.flowspace.model import Constant
        for block in graph.iterblocks():
            operations = [
                op for op in block.operations
                if not ((op.opname == "jit_force_virtualizable" and
                         self.portal_jd is None) or
                        (op.opname == "jit_marker" and op.args and
                         isinstance(op.args[0], Constant) and
                         op.args[0].value in
                         ("can_enter_jit", "loop_header")))]
            if len(operations) != len(block.operations):
                block.operations = operations

    def _available_carried_values(self, graph, block):
        origins = _variable_origins(graph)
        available = list(block.inputargs)
        available += [op.result for op in block.operations]
        result = {}
        for name, start_value in self._named_start_arguments(graph).items():
            start_origin = origins.get(start_value, start_value)
            for value in available:
                if origins.get(value, value) is start_origin:
                    result[name] = value
                    break
        return result

    def _runtime_values(self, transition):
        from rpython.flowspace.model import Constant
        from rpython.rtyper.lltypesystem import lltype
        def is_null(value):
            return (isinstance(value, Constant) and
                    (value.value is None or
                     (isinstance(value.concretetype, lltype.Ptr) and
                      not value.value)))
        return [value for value in transition.dynamic_values()
                if not is_null(value)]

    def _successor_arguments(self, graph, argnames, linked_block,
                             dynamic_values, carried_values=None):
        from rpython.flowspace.model import Constant
        static_values = dict(linked_block.bindings)
        static_values["opcode"] = linked_block.opcode
        dynamic_values = iter(dynamic_values)
        result = []
        for name, var in zip(argnames, graph.startblock.inputargs):
            if name in static_values:
                result.append(Constant(static_values[name], var.concretetype))
            else:
                try:
                    result.append(next(dynamic_values))
                except StopIteration:
                    if carried_values is not None and name in carried_values:
                        result.append(carried_values[name])
                    elif name == "bytecode":
                        from rpython.rtyper.lltypesystem import lltype
                        result.append(Constant(
                            lltype.nullptr(var.concretetype.TO),
                            var.concretetype))
                    else:
                        raise ValueError(
                            "not enough dynamic transition values for %r" %
                            (argnames,))
        try:
            next(dynamic_values)
        except StopIteration:
            return result
        raise ValueError("too many dynamic transition values")

    def _merge_exception_block(self, graph, final_except):
        if graph.exceptblock is final_except:
            return
        for block in list(graph.iterblocks()):
            for link in block.exits:
                if link.target is graph.exceptblock:
                    link.target = final_except

    def _remove_result_tuple(self, transition):
        tuple_var = transition.result_var
        transition.block.operations = [
            op for op in transition.block.operations
            if not (op.result is tuple_var or
                    (op.opname == "setfield" and op.args[0] is tuple_var))]

    def _assemble(self, graph, entry_blocks, name, portal_jd=None):
        from rpython.flowspace.model import copygraph
        from rpython.jit.codewriter.assembler import JitCode
        from rpython.jit.codewriter.flatten import flatten_graph, KINDS
        from rpython.jit.codewriter.jtransform import transform_graph
        from rpython.jit.codewriter.liveness import compute_liveness
        from rpython.jit.codewriter.regalloc import perform_register_allocation
        from rpython.rtyper.lltypesystem import llmemory

        # Codewriter transformation mutates exitswitches and operations.  Keep
        # the linked FunctionGraph valid for inspection and LLInterpreter
        # differential tests, and lower a structural copy instead.
        original_blocks = list(graph.iterblocks())
        lowered_graph = copygraph(graph, shallowvars=True)
        lowered_blocks = list(lowered_graph.iterblocks())
        blockmap = dict(zip(original_blocks, lowered_blocks))
        lowered_entries = dict(
            (pc, blockmap[block]) for pc, block in entry_blocks.items())

        callcontrol = self.codewriter.callcontrol
        if portal_jd is None:
            portal_jd = callcontrol.jitdriver_sd_from_portal_graph(
                lowered_graph)
        transform_graph(
            lowered_graph, self.codewriter.cpu, callcontrol, portal_jd)
        regallocs = dict((kind, perform_register_allocation(lowered_graph, kind))
                         for kind in KINDS)
        ssarepr = flatten_graph(
            lowered_graph, regallocs, cpu=callcontrol.cpu)
        if portal_jd is not None:
            with_liveness = []
            for insn in ssarepr.insns:
                if insn[0] == "goto":
                    with_liveness.append(("-live-",))
                with_liveness.append(insn)
            ssarepr.insns = with_liveness
        compute_liveness(ssarepr)
        num_regs = dict(
            (kind, max(regallocs[kind]._coloring.values()) + 1
             if regallocs[kind]._coloring else 0)
            for kind in KINDS)
        # This JitCode is entered by the meta-interpreter, not by an inline
        # residual call.  Give it a typed null address nevertheless: leaving
        # fnaddr as Python None would widen JitCode.fnaddr to Address|None
        # during annotator helper completion and make native translation fail.
        jitcode = JitCode(name, fnaddr=llmemory.NULL)
        self.codewriter.assembler.assemble(ssarepr, jitcode, num_regs)
        entry_positions = dict(
            (pc, self.codewriter.assembler.label_positions[block])
            for pc, block in lowered_entries.items())
        return jitcode, entry_positions


class ResidualTemplateLinker(object):
    """Link code-independent templates into a code-specific residual CFG."""

    def __init__(self, catalog, instruction_size=2, decoder=None):
        self.catalog = catalog
        self.instruction_size = instruction_size
        self.decoder = decoder

    def link(self, code, entry_pc=0):
        blocks = {}
        pending = [entry_pc]
        while pending:
            pc = pending.pop()
            if pc in blocks:
                continue

            if self.decoder is None:
                opcode = ord(code[pc])
                oparg = ord(code[pc + 1])
                bindings = {"pc": pc, "oparg": oparg, "code": code}
            else:
                opcode, oparg, bindings = self.decoder(code, pc)
            template = self.catalog.lookup(opcode)
            block = LinkedTemplateBlock(pc, opcode, oparg, template, bindings)
            # Publish before following successors so backedges hit the cache.
            blocks[pc] = block

            for terminator, targets in zip(
                    template.terminators,
                    template.resolve_targets(bindings)):
                if isinstance(terminator, Finish):
                    block.has_finish = True
                    continue
                if isinstance(targets, tuple):
                    resolved = targets
                else:
                    resolved = (targets,)
                for target in resolved:
                    if target not in block.successors:
                        block.successors.append(target)
                    if target not in blocks:
                        pending.append(target)
        return LinkedResidualProgram(entry_pc, blocks).analyze_loops()


def _compute_dominators(entry_pc, blocks):
    all_pcs = set(blocks)
    predecessors = dict((pc, set()) for pc in blocks)
    for source, block in blocks.items():
        for target in block.successors:
            predecessors[target].add(source)

    dominators = dict((pc, set(all_pcs)) for pc in blocks)
    dominators[entry_pc] = set([entry_pc])
    changed = True
    while changed:
        changed = False
        for pc in blocks:
            if pc == entry_pc:
                continue
            preds = predecessors[pc]
            if preds:
                common = set(all_pcs)
                for pred in preds:
                    common.intersection_update(dominators[pred])
            else:
                common = set()
            new_dominators = common.union(set([pc]))
            if new_dominators != dominators[pc]:
                dominators[pc] = new_dominators
                changed = True
    return dominators


class ResidualTemplateGenerator(object):
    """Normalize concrete residual variants into the template IR.

    This is the first bridge from the existing PE.  A later generator will
    accept symbolic split values and emit typed holes directly.
    """

    def __init__(self, terminal_values=(-1,)):
        self.terminal_values = terminal_values

    def from_residual_graph(self, key, graph, transitions):
        operations = []
        for block in graph.iterblocks():
            operations.extend(block.operations)

        terminators = []
        for transition in transitions:
            next_value = transition.constant_next_value()
            dynamic_values = transition.dynamic_values()
            if next_value in self.terminal_values:
                terminators.append(Finish(dynamic_values))
            else:
                terminators.append(Continue(next_value, dynamic_values))

        return ResidualTemplate(key, operations, (), terminators)

    def from_symbolic_residual_graph(self, key, graph, transitions,
                                     pc_name="pc", oparg_name="oparg"):
        """Lift simple residual pc expressions to late-static targets."""
        argnames = graph.signature[0]
        inputs = dict(zip(argnames, graph.startblock.inputargs))
        pc_var = inputs.get(pc_name)
        oparg_var = inputs.get(oparg_name)
        definitions = _operation_definitions(graph)
        stored_fields = _stored_fields(graph)
        origins = _variable_origins(graph)

        pc = PcHole()
        oparg = OpargHole()
        holes = [pc, oparg]
        terminators = []
        lifted_results = set()
        for transition in transitions:
            split_value = transition.fields["item0"]
            target = _lift_target(
                split_value, definitions, stored_fields, origins,
                pc_var, oparg_var, pc, oparg)
            if split_value in definitions:
                lifted_results.add(split_value)
            dynamic_values = transition.dynamic_values()
            if isinstance(target, Constant) and target.value in self.terminal_values:
                terminators.append(Finish(dynamic_values))
            else:
                terminators.append(Continue(target, dynamic_values))

        operations = []
        for block in graph.iterblocks():
            for op in block.operations:
                # Tuple construction and lifted pc arithmetic belong to the
                # template/linker boundary, not to residual execution.
                if (op.opname in ("malloc", "setfield") or
                        op.result in lifted_results):
                    continue
                operations.append(op)
        return ResidualTemplate(
            key, operations, holes, terminators, residual_graph=graph)

    def symbolic_fallthrough(self, key, operations, dynamic_values,
                             instruction_size):
        pc = PcHole()
        next_pc = NextPcHole(pc, instruction_size)
        return ResidualTemplate(
            key, operations, (pc, next_pc),
            (Continue(next_pc, dynamic_values),))

    def symbolic_absolute_jump(self, key, operations, dynamic_values):
        oparg = OpargHole()
        target = AbsoluteTarget(oparg)
        return ResidualTemplate(
            key, operations, (oparg,),
            (Continue(target, dynamic_values),))

    def symbolic_relative_jump(self, key, operations, dynamic_values,
                               instruction_size):
        pc = PcHole()
        oparg = OpargHole()
        next_pc = NextPcHole(pc, instruction_size)
        target = RelativeTarget(next_pc, oparg)
        return ResidualTemplate(
            key, operations, (pc, oparg, next_pc),
            (Continue(target, dynamic_values),))

    def symbolic_absolute_branch(self, key, operations, condition,
                                 dynamic_values, instruction_size):
        pc = PcHole()
        oparg = OpargHole()
        next_pc = NextPcHole(pc, instruction_size)
        jump_target = AbsoluteTarget(oparg)
        terminator = Branch(
            condition, next_pc, jump_target, dynamic_values)
        return ResidualTemplate(
            key, operations, (pc, oparg, next_pc), (terminator,))

    def symbolic_finish(self, key, operations, dynamic_values):
        return ResidualTemplate(
            key, operations, (), (Finish(dynamic_values),))


def _operation_definitions(graph):
    result = {}
    for block in graph.iterblocks():
        for op in block.operations:
            result[op.result] = op
    return result


def _stored_fields(graph):
    result = {}
    for block in graph.iterblocks():
        for op in block.operations:
            if (op.opname == "setfield" and
                    isinstance(op.args[1], Constant)):
                key = (op.args[0], op.args[1].value)
                result[key] = op.args[2]
    return result


def _variable_origins(graph):
    origins = dict((var, var) for var in graph.startblock.inputargs)
    changed = True
    while changed:
        changed = False
        for block in graph.iterblocks():
            for link in block.exits:
                for value, target_var in zip(link.args,
                                             link.target.inputargs):
                    origin = origins.get(value)
                    if origin is not None and target_var not in origins:
                        origins[target_var] = origin
                        changed = True
    return origins


def _lift_target(value, definitions, stored_fields, origins,
                 pc_var, oparg_var, pc, oparg):
    origin = origins.get(value, value)
    if origin is pc_var:
        return pc
    if origin is oparg_var:
        return AbsoluteTarget(oparg)
    if isinstance(value, Constant):
        return value

    op = definitions.get(value)
    if op is not None and op.opname == "same_as":
        return _lift_target(
            op.args[0], definitions, stored_fields, origins,
            pc_var, oparg_var, pc, oparg)
    if (op is not None and op.opname == "getfield" and
            isinstance(op.args[1], Constant)):
        stored = stored_fields.get((op.args[0], op.args[1].value))
        if stored is not None:
            return _lift_target(
                stored, definitions, stored_fields, origins,
                pc_var, oparg_var, pc, oparg)
    if op is not None and op.opname == "int_add":
        left, right = op.args
        left_origin = origins.get(left, left)
        right_origin = origins.get(right, right)
        if left_origin is pc_var and isinstance(right, Constant):
            return NextPcHole(pc, right.value)
        if right_origin is pc_var and isinstance(left, Constant):
            return NextPcHole(pc, left.value)
    raise ValueError("unsupported symbolic split expression %r" % (value,))
