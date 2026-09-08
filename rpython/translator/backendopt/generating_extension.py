"""The generating extension an interpreter's step function specializes into."""

from rpython.rlib.debug import debug_print
from rpython.rlib.rarithmetic import intmask
from rpython.translator.backendopt.partialeval_template import (
    Finish, LinkedResidualProgram, LinkedTemplateBlock)


def _states_equal(a, b):
    # RPython has no dict comparison (binaryop.py's ne raises).
    if len(a) != len(b):
        return False
    for key, value in a.items():
        if key not in b or b[key] != value:
            return False
    return True


class GeneratingExtension(object):
    """Residual templates for one interpreter, plus the decoder for its code."""

    def __init__(self, templates, decoder, static_name,
                 unsupported=None, policy=None, state_names=(),
                 leave_key=-1, ref_hole_names=()):
        self.templates = dict(templates)
        self.decoder = decoder
        self.ref_hole_names = tuple(ref_hole_names)
        self.handler_edges = {}
        self.leave_key = leave_key
        self.state_names = tuple(state_names)
        self.policy = policy
        self.static_name = static_name
        self.unsupported = dict(unsupported or {})
        # (pc, key) that made the last generate() decline; (-1, -1) is none.
        self.last_blocked = (-1, -1)
        self.decline_reason = None

    @classmethod
    def from_step_function(cls, translator, step_function, keys, decoder,
                           static_name=None, terminal_values=(-1,),
                           policy=None, graph=None, ref_hole_names=()):
        """Specialize ``step_function`` once per key."""
        from rpython.translator.backendopt.partialeval import PartialEvaluator
        from rpython.translator.translator import graphof

        if graph is None:
            graph = graphof(translator, step_function)
        if static_name is None:
            static_name = graph.func._pe_static_args_[0]
        split_names = getattr(graph.func, "_pe_split_args_", ())
        state_names = tuple(name for name in split_names[1:])
        evaluator = PartialEvaluator(translator)

        skipped = getattr(graph.func, "_pe_skip_keys_", ())
        if policy is None:
            policy = getattr(graph.func, "_pe_link_policy_", None)
        templates = {}
        unsupported = {}
        for key in skipped:
            unsupported[key] = ValueError("declared not specializable")
        for key in keys:
            if key in unsupported:
                continue
            try:
                templates[key] = evaluator.make_symbolic_template(
                    key, graph, {static_name: key},
                    terminal_values=terminal_values)
            except Exception as error:  # noqa: BLE001 - a diagnostic catalogue
                import traceback
                error.pe_traceback = traceback.format_exc()
                unsupported[key] = error
        return cls(templates, decoder, static_name, unsupported,
                   policy=policy, state_names=state_names,
                   ref_hole_names=ref_hole_names)

    def handles(self, key):
        return key in self.templates

    def generate(self, code, entry_pc=0, entry_state=None):
        """Residual code for ``code``, entered at ``entry_pc``."""
        from rpython.rtyper.annlowlevel import cast_instance_to_gcref

        blocks = {}
        ref_bindings = {}
        for name in self.ref_hole_names:
            ref_bindings[name] = cast_instance_to_gcref(code)
        if entry_state is None:
            entry_state = {}
        state_names = self.state_names
        # Not dict(entry_state): RPython's dict has no mapping-arg ctor.
        pending = [(entry_pc, entry_state.copy())]
        self.last_blocked = (-1, -1)
        self.decline_reason = None
        leave_pcs = {}
        handler_pcs = {}

        while pending:
            pc, state = pending.pop()
            if pc in blocks:
                if not _states_equal(blocks[pc].state, state):
                    raise ValueError(
                        "pc %d is reachable with conflicting late-static "
                        "state" % (pc,))
                continue

            block = self._decode_block(code, pc, state, ref_bindings, leave_pcs)
            if block is None:
                return None
            blocks[pc] = block

            self._enqueue_successors(
                pc, block.key, block.template, block.bindings, block,
                pending, blocks)
            self._enqueue_handler_edge(
                pc, block.key, block.bindings, state, block, pending, blocks,
                handler_pcs)

        program = LinkedResidualProgram(entry_pc, blocks, state_names)
        program.leave_pcs = leave_pcs
        program.handler_pcs = handler_pcs
        program.leave_blocks = len(leave_pcs)
        program = program.analyze_loops()
        # After loop analysis so a policy can ask about loop headers.
        if self.policy is not None and not self.policy(program, code):
            self.decline_reason = (
                "declined by policy (%d blocks)" % len(program.blocks))
            return None
        return program

    def _decode_block(self, code, pc, state, ref_bindings, leave_pcs):
        """Decode and bind one pc into a block; None signals a decline."""
        key, bindings = self.decoder(code, pc)
        if key not in self.templates:
            # Fall back to the leave template rather than declining.
            if self.leave_key < 0 or self.leave_key not in self.templates:
                self.last_blocked = (intmask(pc), key)
                return None
            key = self.leave_key
            leave_pcs[intmask(pc)] = True
        template = self.templates[key]

        bindings = bindings.copy()
        bindings[self.static_name] = key
        bindings.update(state)
        block = LinkedTemplateBlock(pc, key, template, bindings, ref_bindings)
        block.state = state
        return block

    def _enqueue_successors(self, pc, key, template, bindings, block,
                            pending, blocks):
        terminators = template.terminators
        targets_by_index = template.resolve_targets(bindings)
        states_by_index = template.resolve_state(bindings)
        for index in range(len(terminators)):
            terminator = terminators[index]
            targets = targets_by_index[index]
            next_state = states_by_index[index]
            if isinstance(terminator, Finish):
                block.has_finish = True
                continue
            for target in targets:
                debug_print("pe-cogen-scan edge", intmask(pc), "key",
                            key, "->", intmask(target))
                if target not in block.successors:
                    block.successors.append(target)
                if target not in blocks:
                    pending.append((target, next_state))

    def _enqueue_handler_edge(self, pc, key, bindings, state, block,
                              pending, blocks, handler_pcs):
        if key not in self.handler_edges:
            return
        pc_name, offset_name = self.handler_edges[key]
        target = bindings[pc_name] + bindings[offset_name]
        debug_print("pe-cogen-scan handler edge", intmask(pc),
                    "->", intmask(target))
        handler_pcs[intmask(target)] = True
        if target not in block.successors:
            block.successors.append(target)
        if target not in blocks:
            pending.append((target, state))

    def report(self, name_of=None):
        """Which instructions specialized, and why the rest did not."""
        total = len(self.templates) + len(self.unsupported)
        lines = ["generating extension: %d/%d instructions specialized"
                 % (len(self.templates), total)]
        for key in sorted(self.unsupported):
            error = self.unsupported[key]
            label = name_of(key) if name_of is not None else str(key)
            lines.append("  unsupported %-28s %s: %s"
                         % (label, type(error).__name__, error))
        return "\n".join(lines)
