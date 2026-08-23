"""Intermediate representation for offline-generated residual templates.

The objects in this module describe code-independent work completed by the
offline partial evaluator.  They deliberately contain no runtime linker or
JitCode details.
"""

from rpython.flowspace.model import Constant
from rpython.rtyper.lltypesystem.lloperation import llop


class Resolvable(object):
    """Common base for anything with resolve(bindings) -> value: lets
    _resolve_operand/_resolve_target use isinstance instead of hasattr
    (RPython's hasattr only works on compile-time constants), and gives
    the annotator one type to unify these otherwise-unrelated classes
    under.
    """


class TemplateHole(Resolvable):
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


class TargetExpr(Resolvable):
    """Common base of AbsoluteTarget/RelativeTarget/ExprTarget -- see
    Resolvable's docstring for why."""


class AbsoluteTarget(TargetExpr):
    """A bytecode target stored directly in ``oparg``."""

    def __init__(self, oparg):
        self.oparg = oparg

    def resolve(self, bindings):
        return bindings[self.oparg.name]


class RelativeTarget(TargetExpr):
    """A bytecode target relative to the following instruction."""

    def __init__(self, next_pc, oparg):
        self.next_pc = next_pc
        self.oparg = oparg

    def resolve(self, bindings):
        return (self.next_pc.resolve(bindings) +
                bindings[self.oparg.name])


class ConstExpr(Resolvable):
    """Wraps an already-concrete value so ExprTarget.args has one uniform
    Resolvable item type; RPython cannot unify a bare int with an
    instance in the same list.
    """

    def __init__(self, value):
        self.value = value

    def resolve(self, bindings):
        return self.value


class ExprTarget(TargetExpr):
    """A target computed from holes by a pure operation.

    SOM-style bytecodes reach their target as ``pc + oparg`` (relative),
    ``pc - oparg`` (backward), or ``pc + (oparg + (oparg2 << 8))`` (two-byte
    offset).  Those expressions are late-static: every leaf is either a hole or
    a compile-time constant, so the whole tree collapses to an integer once a
    concrete code object is linked.

    The operation is evaluated with RPython's own ``llop`` implementation
    rather than one written out here, so a resolved target has exactly the
    width and wrap-around behaviour the interpreter itself would produce.
    """

    def __init__(self, opname, restype, args):
        self.opname = opname
        self.restype = restype
        # A list, not a tuple: RPython's rtyper only supports iterating a
        # tuple whose length is statically 1; most late-static ops have 2.
        self.args = list(args)

    def __repr__(self):
        return "%s(%s)" % (self.opname,
                           ", ".join([repr(arg) for arg in self.args]))

    def resolve(self, bindings):
        values = [_resolve_operand(arg, bindings) for arg in self.args]
        return _apply_late_static_op(self.opname, self.restype, values)


def _is_late_static_operation(opname):
    """Can this operation be evaluated at link time from holes alone?

    ``canfold`` is RPython's own purity flag: a foldable operation is a pure
    function of its arguments and cannot raise, which is precisely the
    condition for lifting it out of the residual code into a target
    expression.  Anything that reads memory or calls out stays dynamic.
    """
    try:
        return getattr(llop, opname).canfold
    except AttributeError:
        return False


# The unsigned variants are here because an interpreter may carry its pc
# unsigned, which makes every operation on a late-static target unsigned too.
_BINARY_LATE_STATIC_OPS = (
    "int_add", "int_sub", "int_mul", "int_and", "int_or", "int_xor",
    "int_lshift", "int_rshift", "uint_rshift", "int_floordiv", "int_mod",
    "int_eq", "int_ne", "int_lt", "int_le", "int_gt", "int_ge",
    "uint_add", "uint_sub", "uint_mul", "uint_and", "uint_or", "uint_xor",
    "uint_lshift", "uint_floordiv", "uint_mod",
    "uint_eq", "uint_ne", "uint_lt", "uint_le", "uint_gt", "uint_ge",
)
# The casts are here because a target mixing a signed oparg with an unsigned
# pc converts one to the other before combining them.
_UNARY_LATE_STATIC_OPS = ("int_neg", "int_invert", "int_is_true",
                          "uint_invert", "uint_is_true",
                          "cast_int_to_uint", "cast_uint_to_int")


def _apply_late_static_op(opname, restype, values):
    """Evaluate one late-static ExprTarget operation.

    Not getattr(llop, opname)(restype, *values): RPython's annotator
    needs a literal attribute name, and splat args aren't RPython-legal.
    Add new opnames to the tables above and a branch below.
    """
    if opname in _BINARY_LATE_STATIC_OPS:
        left = values[0]
        right = values[1]
        if opname == "int_add":
            return llop.int_add(restype, left, right)
        if opname == "int_sub":
            return llop.int_sub(restype, left, right)
        if opname == "int_mul":
            return llop.int_mul(restype, left, right)
        if opname == "int_and":
            return llop.int_and(restype, left, right)
        if opname == "int_or":
            return llop.int_or(restype, left, right)
        if opname == "int_xor":
            return llop.int_xor(restype, left, right)
        if opname == "int_lshift":
            return llop.int_lshift(restype, left, right)
        if opname == "int_rshift":
            return llop.int_rshift(restype, left, right)
        if opname == "uint_rshift":
            return llop.uint_rshift(restype, left, right)
        if opname == "int_floordiv":
            return llop.int_floordiv(restype, left, right)
        if opname == "int_mod":
            return llop.int_mod(restype, left, right)
        if opname == "int_eq":
            return llop.int_eq(restype, left, right)
        if opname == "int_ne":
            return llop.int_ne(restype, left, right)
        if opname == "int_lt":
            return llop.int_lt(restype, left, right)
        if opname == "int_le":
            return llop.int_le(restype, left, right)
        if opname == "int_gt":
            return llop.int_gt(restype, left, right)
        if opname == "int_ge":
            return llop.int_ge(restype, left, right)
        if opname == "uint_add":
            return llop.uint_add(restype, left, right)
        if opname == "uint_sub":
            return llop.uint_sub(restype, left, right)
        if opname == "uint_mul":
            return llop.uint_mul(restype, left, right)
        if opname == "uint_and":
            return llop.uint_and(restype, left, right)
        if opname == "uint_or":
            return llop.uint_or(restype, left, right)
        if opname == "uint_xor":
            return llop.uint_xor(restype, left, right)
        if opname == "uint_lshift":
            return llop.uint_lshift(restype, left, right)
        if opname == "uint_floordiv":
            return llop.uint_floordiv(restype, left, right)
        if opname == "uint_mod":
            return llop.uint_mod(restype, left, right)
        if opname == "uint_eq":
            return llop.uint_eq(restype, left, right)
        if opname == "uint_ne":
            return llop.uint_ne(restype, left, right)
        if opname == "uint_lt":
            return llop.uint_lt(restype, left, right)
        if opname == "uint_le":
            return llop.uint_le(restype, left, right)
        if opname == "uint_gt":
            return llop.uint_gt(restype, left, right)
        if opname == "uint_ge":
            return llop.uint_ge(restype, left, right)
    if opname in _UNARY_LATE_STATIC_OPS:
        only = values[0]
        if opname == "int_neg":
            return llop.int_neg(restype, only)
        if opname == "int_invert":
            return llop.int_invert(restype, only)
        if opname == "int_is_true":
            return llop.int_is_true(restype, only)
        if opname == "uint_invert":
            return llop.uint_invert(restype, only)
        if opname == "uint_is_true":
            return llop.uint_is_true(restype, only)
        if opname == "cast_int_to_uint":
            return llop.cast_int_to_uint(restype, only)
        if opname == "cast_uint_to_int":
            return llop.cast_uint_to_int(restype, only)
    # Not %r: RPython's rtyper only implements %s/%d/... formatting.
    raise ValueError(
        "no runtime-cogen dispatch for late-static op %s -- add it to "
        "_apply_late_static_op (partialeval_template.py)" % (opname,))


def _resolve_operand(value, bindings):
    # Not ``hasattr(value, "resolve")``: see Resolvable's own docstring.
    if isinstance(value, Resolvable):
        return value.resolve(bindings)
    return value


class Terminator(object):
    """Common base of Continue/Finish/Branch: RPython unifies a list's
    item type across classes only when they share a base.
    """


class Continue(Terminator):
    """Continue execution at a late-static or concrete split value."""

    # Class-level default: object-typed attrs need one so the annotator
    # learns the type from the class itself, not from instance-discovery
    # order. Only object-typed attrs get this -- tuple attrs (dynamic_
    # values/state) can't: RPython tuples of different lengths never
    # unify, and this project's interpreters use both 0- and 1-element
    # state tuples.
    target = None

    def __init__(self, target, dynamic_values, state=()):
        self.target = target
        self.dynamic_values = tuple(dynamic_values)
        # Further late-static interpreter state for the successor, as
        # (name, expression) pairs -- an operand-stack depth, for instance.
        # A list, not a tuple: an interpreter with no late-static state
        # gives an empty one, and RPython cannot iterate a 0-tuple.
        self.state = list(state)


class Finish(Terminator):
    """Return from the residual program."""

    def __init__(self, values, state=()):
        self.values = tuple(values)
        # A list, not a tuple: an interpreter with no late-static state
        # gives an empty one, and RPython cannot iterate a 0-tuple.
        self.state = list(state)


class Branch(Terminator):
    """Keep a dynamic condition with two late-static bytecode targets."""

    # Object-typed class defaults -- see Continue's own note above.
    condition = None
    true_target = None
    false_target = None

    def __init__(self, condition, true_target, false_target, dynamic_values,
                 state=()):
        self.condition = condition
        self.true_target = true_target
        self.false_target = false_target
        self.dynamic_values = tuple(dynamic_values)
        # A list, not a tuple: an interpreter with no late-static state
        # gives an empty one, and RPython cannot iterate a 0-tuple.
        self.state = list(state)


class ResidualTemplate(object):
    def __init__(self, key, operations, holes, terminators,
                 residual_graph=None):
        self.key = key
        self.operations = tuple(operations)
        self.holes = tuple(holes)
        # A list, not a tuple: different opcodes have different terminator
        # counts, and an RPython tuple attribute has one fixed shape.
        self.terminators = list(terminators)
        # Kept as an offline-only lowering source.  Runtime linking only uses
        # the immutable template data above; translation-time lowering can use
        # the graph to preserve arbitrary residual control flow and exception
        # edges without reconstructing it from a flat operation list.
        self.residual_graph = residual_graph

    def resolve_state(self, bindings):
        """Resolve each terminator's late-static successor state.

        Plain loops, not a list-comp wrapping dict(genexpr): that would
        close over bindings, which RPython does not allow.
        """
        result = []
        for terminator in self.terminators:
            state = {}
            for name, expr in terminator.state:
                state[name] = _resolve_operand(expr, bindings)
            result.append(state)
        return result

    def resolve_targets(self, bindings):
        """Resolve only late-static control targets for a code instance.

        One list per terminator (0/1/2 targets): RPython needs a uniform
        item type, so no bare value for some and a tuple for others.
        """
        targets = []
        for terminator in self.terminators:
            if isinstance(terminator, Branch):
                targets.append([
                    _resolve_target(terminator.true_target, bindings),
                    _resolve_target(terminator.false_target, bindings),
                ])
            elif isinstance(terminator, Continue):
                targets.append([_resolve_target(terminator.target, bindings)])
            else:
                targets.append([])
        return targets


def flatten_resolved_targets(targets_by_terminator, num_exits):
    """One target int per fragment exit slot: a single-terminator
    template's lone target is repeated to fill every exit slot.
    """
    flat = []
    for target_list in targets_by_terminator:
        for target in target_list:
            flat.append(target)
    if len(flat) == 1 and num_exits > 1:
        only = flat[0]
        flat = []
        for _index in range(num_exits):
            flat.append(only)
    return flat


def _resolve_target(target, bindings):
    # Not hasattr(target, "resolve"): see Resolvable's own docstring.
    # Resolvable, not TargetExpr: target can also be a bare TemplateHole.
    if isinstance(target, Resolvable):
        return target.resolve(bindings)
    # Assert pins the type: without it, the annotator unions this
    # branch's SomeInstance return with the 'if' branch's SomeInteger
    # and raises UnionError.
    assert isinstance(target, int)
    return target


class LinkedTemplateBlock(object):
    """One instruction of a program, and the template generated for it.

    ``bindings`` is the complete specialization environment for this block:
    every late-static name the template left as a hole, *including* the static
    key under the name the step function declared for it.  Keeping it complete
    is what lets the lowerer substitute constants without knowing which name
    means what.
    """

    def __init__(self, pc, key, template, bindings):
        self.pc = pc
        self.key = key
        self.template = template
        self.bindings = bindings
        # Extra late-static values this block was reached with.
        self.state = {}
        self.successors = []
        self.has_finish = False
        self.is_loop_header = False


class LinkedResidualProgram(object):
    def __init__(self, entry_pc, blocks, state_names=()):
        self.entry_pc = entry_pc
        self.blocks = blocks
        # Lists, not tuples: filled in later with a dynamic number of
        # entries; an RPython tuple's length must be fixed at annotation.
        self.backedges = []
        self.loop_headers = []
        # How many blocks are the synthetic "leave" fallback rather than a
        # real instruction's template; set by GeneratingExtension.generate.
        self.leave_blocks = 0
        # Passed in, not reassigned after construction: RPython would
        # need to unify a 0-length and a 1-length tuple on this attribute
        # across the whole build, which it cannot do.
        # Not tuple(state_names): the tuple() builtin has no rtyper
        # support; every real caller already passes a tuple.
        self.state_names = state_names

    def analyze_loops(self):
        dominators = _compute_dominators(self.entry_pc, self.blocks)
        backedges = []
        for source, block in self.blocks.items():
            for target in block.successors:
                if target in dominators[source]:
                    backedges.append((source, target))
        # Not .sort(): RPython lists of tuples have no comparison to
        # sort by.
        _sort_pairs(backedges)

        # Not sorted(set(genexpr)): none of sorted()/genexpr/set() are
        # RPython-legal. Dict-as-set stands in for set().
        seen = {}
        loop_headers = []
        for _source, target in backedges:
            if target not in seen:
                seen[target] = True
                loop_headers.append(target)
        sort_ints(loop_headers)

        for pc in loop_headers:
            self.blocks[pc].is_loop_header = True
        self.backedges = backedges
        self.loop_headers = loop_headers
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
        # Plain loops, not tuple(sorted(...))/tuple(genexpr): neither is
        # RPython-legal here, and there is nothing to gain from a tuple.
        entry_pcs = []
        for pc in entry_positions:
            entry_pcs.append(pc)
        sort_ints(entry_pcs)
        positions = []
        for pc in entry_pcs:
            positions.append(entry_positions[pc])
        sources = []
        targets = []
        for source, target in self.backedges:
            sources.append(source)
            targets.append(target)
        block_pcs = []
        for pc in self.blocks:
            block_pcs.append(pc)
        sort_ints(block_pcs)
        jitcode.pe_metadata = PEJitCodeMetadata(
            self.entry_pc, block_pcs, self.loop_headers,
            sources, targets, entry_pcs, positions)
        return jitcode

    def lower(self, codewriter, name="offline-residual", portal_jd=None,
              runtime_names=("self", "bytecode"), null_names=("bytecode",),
              jit_merge_point_args=()):
        """Build the residual graph and assemble it into one JitCode.

        Two stages with quite different dependencies, kept separate on the
        lowerer below: building the graph is pure flow-graph work, while
        assembling needs the codewriter, and through it the whole-program view
        that turns a call into either a JitCode or a residual call.  Only the
        second stage is inherently translation-time, so it is the one a runtime
        linker would have to replace -- by assembling each *template* once,
        offline, and patching the result.
        """
        lowerer = LinkedResidualLowerer(runtime_names, null_names,
                                        self.state_names, jit_merge_point_args)
        return lowerer.lower(codewriter, self, name, portal_jd=portal_jd)


class LoweredResidualProgram(object):
    def __init__(self, graph, jitcode, entry_positions):
        self.graph = graph
        self.jitcode = jitcode
        self.entry_positions = entry_positions


class LinkedResidualLowerer(object):
    """Turn a linked template CFG into a graph, and that graph into a JitCode.

    The codewriter is deliberately not held here: it is an argument of
    ``assemble`` alone, so that ``build_graph`` cannot come to depend on it by
    accident.  That boundary is the interesting one -- everything on the graph
    side is ordinary flow-graph manipulation that a runtime linker could do,
    and everything past it needs a compiler that only exists while translating.
    """

    def __init__(self, runtime_names=("self", "bytecode"),
                 null_names=("bytecode",), state_names=(),
                 jit_merge_point_args=()):
        # Result-tuple items after the return value that are late-static state
        # rather than values carried to the successor at runtime.
        self.state_names = tuple(state_names)
        self.state_count = len(self.state_names)
        # Step-function arguments holding the jitdriver's greens then reds, in
        # that order.  When given, each loop header of the linked program gets
        # a real jit_merge_point, which is what lets the meta-interpreter close
        # a loop *inside* the program rather than only at its entry.
        self.jit_merge_point_args = tuple(jit_merge_point_args)
        # Residual parameters the portal supplies at trace start, in the order
        # the entry wrapper expects them.
        self.runtime_names = tuple(runtime_names)
        # Residual pointer parameters that are dead in the linked CFG and may
        # be passed as null on a successor edge.
        self.null_names = tuple(null_names)

    def lower(self, codewriter, program, name="offline-residual",
              portal_jd=None):
        graph, entry_blocks = self.build_graph(program, name, portal_jd)
        jitcode, entry_positions = self.assemble(
            codewriter, graph, entry_blocks, name, portal_jd)
        program.attach_to_jitcode(jitcode, entry_positions)
        return LoweredResidualProgram(graph, jitcode, entry_positions)

    def build_graph(self, program, name, portal_jd=None):
        """The residual FunctionGraph for a generated program.

        No codewriter, no JitCode: instantiating templates, substituting the
        late-static bindings and wiring the edges is flow-graph work only.
        """
        self.portal_jd = portal_jd
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
            if self.jit_merge_point_args and (
                    pc in program.loop_headers or pc == program.entry_pc):
                self._insert_merge_point(graph)
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
            # resolve_targets returns one list per terminator (see its
            # own docstring); unwrap it here rather than change the
            # contract other callers depend on.
            targets_by_terminator = linked_block.template.resolve_targets(
                bindings)
            if len(targets_by_terminator) == 1 and len(terminators) > 1:
                targets_by_terminator = targets_by_terminator * len(terminators)
            for transition, terminator, target_list in zip(
                    transitions, terminators, targets_by_terminator):
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
                        graph, transition, terminator, tuple(target_list),
                        instances, program, linked_block)
                    continue
                target = target_list[0]
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
        runtime_names = self.runtime_names
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
        static = linked_block.bindings
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
                    ctx = _LiftContext(graph, named.get("pc"),
                                       named.get("oparg"), PcHole(),
                                       OpargHole(), self._state_holes(named))
                    lifted = _lift_target(target_value, ctx)
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
        values = linked_block.bindings
        replacements = {}
        for name, var in zip(graph.signature[0], graph.startblock.inputargs):
            if name in values:
                replacements[var] = Constant(values[name], var.concretetype)
        return replacements

    def _named_start_arguments(self, graph):
        return dict(zip(graph.signature[0], graph.startblock.inputargs))

    def _state_holes(self, named):
        """Holes for the late-static state names this program carries."""
        return [(named[name], TemplateHole("state", name))
                for name in self.state_names if name in named]

    def _remove_runtime_loop_markers(self, graph):
        """Drop the interpreter's own loop markers from a linked block.

        With merge points inserted, ``can_enter_jit`` is kept: it becomes the
        ``loop_header`` that arms the meta-interpreter, and the merge point at
        the target block is what actually closes the loop.  Without them the
        linked CFG carries its loop structure offline instead.
        """
        from rpython.flowspace.model import Constant
        dropped = ("loop_header",) if self.jit_merge_point_args else (
            "can_enter_jit", "loop_header")
        for block in graph.iterblocks():
            operations = [
                op for op in block.operations
                if not ((op.opname == "jit_force_virtualizable" and
                         self.portal_jd is None) or
                        (op.opname == "jit_marker" and op.args and
                         isinstance(op.args[0], Constant) and
                         op.args[0].value in dropped))]
            if len(operations) != len(block.operations):
                block.operations = operations

    def _insert_merge_point(self, graph):
        """Put a jit_merge_point at the top of a linked loop header.

        The greens and reds are taken from the block's own input arguments, so
        the meta-interpreter sees the values as they are *here*, not as they
        were when the trace started.  That is what lets it close a loop nested
        inside the linked program; with only the entry marked, such a loop has
        nothing to close it and tracing unrolls it until memory runs out.
        """
        self._insert_marker(graph, "jit_merge_point")

    def _insert_bailout_point(self, graph):
        """Put a pe_bailout_point at the top of a linked (non-header) block.

        Same greens and reds, read off the block's own input arguments, as
        ``_insert_merge_point`` would use -- jtransform just turns the
        "pe_bailout_point" marker into a no-op while tracing instead of a
        real merge point.  What it buys is a cheap place for the *blackhole*
        interpreter to bail out of a residual jitcode at every block
        boundary, rather than running all the way to the next real
        jit_merge_point.
        """
        self._insert_marker(graph, "pe_bailout_point")

    def _insert_marker(self, graph, key):
        """Shared operand construction for _insert_merge_point/_insert_bailout_point."""
        from rpython.flowspace.model import Constant, SpaceOperation, Variable
        from rpython.rtyper.lltypesystem import lltype

        block = graph.startblock
        named = dict(zip(graph.signature[0], block.inputargs))
        missing = [n for n in self.jit_merge_point_args if n not in named]
        if missing:
            raise ValueError("%s args %r are not arguments of %r"
                             % (key, missing, graph.name))
        result = Variable()
        result.concretetype = lltype.Void
        args = [Constant(key, lltype.Void),
                Constant(self.portal_jd.jitdriver, lltype.Void)]
        args += [named[name] for name in self.jit_merge_point_args]
        block.operations = [SpaceOperation("jit_marker", args, result)] + list(
            block.operations)

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
        return [value for value in transition.dynamic_values(self.state_count)
                if not is_null(value)]

    def _successor_arguments(self, graph, argnames, linked_block,
                             dynamic_values, carried_values=None):
        from rpython.flowspace.model import Constant
        static_values = linked_block.bindings
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
                    elif name in self.null_names:
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

    def assemble(self, codewriter, graph, entry_blocks, name, portal_jd=None):
        """Compile the residual graph into a JitCode.

        This is the translation-time half.  ``transform_graph`` resolves each
        call into either a JitCode to enter or a residual call to record, which
        needs the whole-program view a translated binary no longer has.
        """
        self.codewriter = codewriter
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


def sort_ints(items):
    """In-place insertion sort of a list of ints.

    Not items.sort(): RPython lists have no .sort() method at all.
    ponytail: O(n^2), fine for the tens of items this ever sees.
    """
    index = 1
    while index < len(items):
        key = items[index]
        gap = index - 1
        while gap >= 0 and items[gap] > key:
            items[gap + 1] = items[gap]
            gap -= 1
        items[gap + 1] = key
        index += 1


def sort_strings(items):
    """In-place insertion sort of a list of strings.

    Not items.sort(): RPython lists have no .sort() method at all.
    ponytail: O(n^2), fine for the tens of items this ever sees.
    """
    index = 1
    while index < len(items):
        key = items[index]
        gap = index - 1
        while gap >= 0 and items[gap] > key:
            items[gap + 1] = items[gap]
            gap -= 1
        items[gap + 1] = key
        index += 1


def _pair_less(a, b):
    if a[0] != b[0]:
        return a[0] < b[0]
    return a[1] < b[1]


def _sort_pairs(pairs):
    """In-place lexicographic sort of a list of (int, int) tuples.

    Not pairs.sort(): RPython lists of tuples have no < to sort by.
    ponytail: O(n^2), fine for a program's small backedge list.
    """
    index = 1
    while index < len(pairs):
        key = pairs[index]
        gap = index - 1
        while gap >= 0 and _pair_less(key, pairs[gap]):
            pairs[gap + 1] = pairs[gap]
            gap -= 1
        pairs[gap + 1] = key
        index += 1


def _dictset_equal(a, b):
    """Are these two dict-as-set objects the same set of keys?"""
    if len(a) != len(b):
        return False
    for key in a:
        if key not in b:
            return False
    return True


def _compute_dominators(entry_pc, blocks):
    """Textbook iterative dominator computation.

    Dict-as-set throughout: RPython has no native set type.
    """
    predecessors = {}
    for pc in blocks:
        predecessors[pc] = {}
    for source, block in blocks.items():
        for target in block.successors:
            predecessors[target][source] = True

    dominators = {}
    for pc in blocks:
        full = {}
        for other in blocks:
            full[other] = True
        dominators[pc] = full
    dominators[entry_pc] = {entry_pc: True}

    changed = True
    while changed:
        changed = False
        for pc in blocks:
            if pc == entry_pc:
                continue
            preds = predecessors[pc]
            common = {}
            seen_first = False
            for pred in preds:
                if not seen_first:
                    for key in dominators[pred]:
                        common[key] = True
                    seen_first = True
                else:
                    pred_doms = dominators[pred]
                    stale = []
                    for key in common:
                        if key not in pred_doms:
                            stale.append(key)
                    for key in stale:
                        del common[key]
            common[pc] = True
            if not _dictset_equal(common, dominators[pc]):
                dominators[pc] = common
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
                                     pc_name="pc", oparg_name="oparg",
                                     extra_oparg_names=("oparg2",),
                                     state_names=()):
        """Lift expressions rooted at the declared split pc to targets."""
        argnames = graph.signature[0]
        inputs = dict(zip(argnames, graph.startblock.inputargs))
        pc_var = inputs.get(pc_name)
        if pc_var is None:
            raise ValueError("split argument %r is not in the graph" %
                             (pc_name,))
        oparg_var = inputs.get(oparg_name)

        pc = PcHole()
        oparg = OpargHole()
        holes = [pc, oparg]
        # Interpreters with multi-byte operands (SOM's two-byte jumps) decode
        # more than one operand; each extra one is its own late-static hole.
        extra_holes = []
        for name in extra_oparg_names:
            var = inputs.get(name)
            if var is None:
                continue
            hole = TemplateHole("oparg", name)
            extra_holes.append((var, hole))
            holes.append(hole)
        # Further late-static state (an operand-stack depth, say) behaves just
        # like the pc: its own hole, and its successor value lifted from the
        # step function's result rather than carried at runtime.
        state_holes = []
        for name in state_names:
            var = inputs.get(name)
            if var is None:
                raise ValueError("split argument %r is not in the graph" %
                                 (name,))
            hole = TemplateHole("state", name)
            state_holes.append((var, hole))
            extra_holes.append((var, hole))
            holes.append(hole)

        ctx = _LiftContext(graph, pc_var, oparg_var, pc, oparg, extra_holes)
        terminators = []
        lifted_results = set()
        for transition in transitions:
            split_value = transition.fields["item0"]
            target = _lift_target(split_value, ctx)
            if split_value in ctx.definitions:
                lifted_results.add(split_value)

            state = []
            for offset, value in enumerate(
                    transition.state_values(len(state_names))):
                expr = _lift_target(value, ctx)
                if isinstance(expr, Constant):
                    expr = expr.value
                if value in ctx.definitions:
                    lifted_results.add(value)
                state.append((state_names[offset], expr))

            dynamic_values = transition.dynamic_values(skip=len(state_names))
            if isinstance(target, Constant) and target.value in self.terminal_values:
                terminators.append(Finish(dynamic_values, state))
            else:
                terminators.append(Continue(target, dynamic_values, state))

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


def _phi_sources(graph):
    """For each block input, the values its predecessors pass in.

    Indexing a list with a possibly-negative index makes RPython emit a branch
    (``if i < 0: i += len(l)``), so even a value the interpreter never merges
    reaches its use as a two-predecessor phi.  Recording every source lets such
    a degenerate merge be seen through.
    """
    sources = {}
    for block in graph.iterblocks():
        for link in block.exits:
            for value, target_var in zip(link.args, link.target.inputargs):
                sources.setdefault(target_var, []).append(value)
    return sources


def _same_value(one, other):
    if one is other:
        return True
    return (isinstance(one, Constant) and isinstance(other, Constant) and
            one.concretetype == other.concretetype and one.value == other.value)


class _LiftContext(object):
    """Everything needed to decide whether a value is late-static."""

    def __init__(self, graph, pc_var, oparg_var, pc, oparg, extra_holes):
        self.definitions = _operation_definitions(graph)
        self.stored_fields = _stored_fields(graph)
        self.origins = _variable_origins(graph)
        self.phi_sources = _phi_sources(graph)
        self.pc_var = pc_var
        self.oparg_var = oparg_var
        self.pc = pc
        self.oparg = oparg
        self.extra_holes = tuple(extra_holes)

    def follow(self, value, seen=None):
        """See through merges whose incoming values are all the same.

        Stops at a genuine merge -- one whose predecessors really do supply
        different values -- and returns it unchanged, so the caller treats it
        as dynamic.
        """
        if seen is None:
            seen = ()
        if any(value is entry for entry in seen):
            return value
        sources = self.phi_sources.get(value)
        if not sources:
            return value
        resolved = [self.follow(source, seen + (value,)) for source in sources]
        first = resolved[0]
        for other in resolved[1:]:
            if not _same_value(first, other):
                return value
        return first

    def defining_op(self, value):
        return self.definitions.get(value)


def _lift_target(value, ctx):
    value = ctx.follow(value)
    origin = ctx.origins.get(value, value)
    if origin is ctx.pc_var:
        return ctx.pc
    if origin is ctx.oparg_var:
        return AbsoluteTarget(ctx.oparg)
    if isinstance(value, Constant):
        return value

    op = ctx.defining_op(value)
    if op is not None and op.opname == "same_as":
        return _lift_target(op.args[0], ctx)
    if (op is not None and op.opname == "getfield" and
            isinstance(op.args[1], Constant)):
        stored = ctx.stored_fields.get((op.args[0], op.args[1].value))
        if stored is not None:
            return _lift_target(stored, ctx)
    if op is not None and op.opname == "int_add":
        left, right = op.args
        left_origin = ctx.origins.get(left, left)
        right_origin = ctx.origins.get(right, right)
        if left_origin is ctx.pc_var and isinstance(right, Constant):
            return NextPcHole(ctx.pc, right.value)
        if right_origin is ctx.pc_var and isinstance(left, Constant):
            return NextPcHole(ctx.pc, left.value)
    # General case: any integer expression over holes and constants, which is
    # how relative, backward and two-byte-offset jumps compute their target,
    # and how a bytecode's stack effect updates the operand-stack depth.
    lifted = _lift_operand(value, ctx)
    if lifted is not None:
        return lifted
    raise ValueError("unsupported symbolic split expression %r" % (value,))


def _lift_operand(value, ctx):
    """Lift one operand of a target expression, or None if it stays dynamic."""
    value = ctx.follow(value)
    origin = ctx.origins.get(value, value)
    if origin is ctx.pc_var:
        return ctx.pc
    if origin is ctx.oparg_var:
        return ctx.oparg
    for var, hole in ctx.extra_holes:
        if origin is var:
            return hole
    if isinstance(value, Constant):
        # ConstExpr, not the bare value: see its own docstring.
        return ConstExpr(value.value)

    op = ctx.defining_op(value)
    if op is None:
        return None
    if op.opname == "same_as":
        return _lift_operand(op.args[0], ctx)
    if op.opname == "getfield" and isinstance(op.args[1], Constant):
        stored = ctx.stored_fields.get((op.args[0], op.args[1].value))
        if stored is None:
            return None
        return _lift_operand(stored, ctx)
    if _is_late_static_operation(op.opname):
        args = []
        for arg in op.args:
            lifted = _lift_operand(arg, ctx)
            if lifted is None:
                return None
            args.append(lifted)
        return ExprTarget(op.opname, op.result.concretetype, args)
    return None
