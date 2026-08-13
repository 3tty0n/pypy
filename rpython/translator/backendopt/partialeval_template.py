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
    def __init__(self, key, operations, holes, terminators):
        self.key = key
        self.operations = tuple(operations)
        self.holes = tuple(holes)
        self.terminators = tuple(terminators)

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
    def __init__(self, pc, opcode, oparg, template):
        self.pc = pc
        self.opcode = opcode
        self.oparg = oparg
        self.template = template
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


class ResidualTemplateLinker(object):
    """Link code-independent templates into a code-specific residual CFG."""

    def __init__(self, catalog, instruction_size=2):
        self.catalog = catalog
        self.instruction_size = instruction_size

    def link(self, code, entry_pc=0):
        blocks = {}
        pending = [entry_pc]
        while pending:
            pc = pending.pop()
            if pc in blocks:
                continue

            opcode = ord(code[pc])
            oparg = ord(code[pc + 1])
            template = self.catalog.lookup(opcode)
            block = LinkedTemplateBlock(pc, opcode, oparg, template)
            # Publish before following successors so backedges hit the cache.
            blocks[pc] = block

            bindings = {"pc": pc, "oparg": oparg, "code": code}
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
        origins = _variable_origins(graph)

        pc = PcHole()
        oparg = OpargHole()
        holes = [pc, oparg]
        terminators = []
        lifted_results = set()
        for transition in transitions:
            split_value = transition.fields["item0"]
            target = _lift_target(
                split_value, definitions, origins,
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
        return ResidualTemplate(key, operations, holes, terminators)

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


def _lift_target(value, definitions, origins,
                 pc_var, oparg_var, pc, oparg):
    origin = origins.get(value, value)
    if origin is pc_var:
        return pc
    if origin is oparg_var:
        return AbsoluteTarget(oparg)
    if isinstance(value, Constant):
        return value

    op = definitions.get(value)
    if op is not None and op.opname == "int_add":
        left, right = op.args
        left_origin = origins.get(left, left)
        right_origin = origins.get(right, right)
        if left_origin is pc_var and isinstance(right, Constant):
            return NextPcHole(pc, right.value)
        if right_origin is pc_var and isinstance(left, Constant):
            return NextPcHole(pc, left.value)
    raise ValueError("unsupported symbolic split expression %r" % (value,))
