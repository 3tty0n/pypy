"""The generating extension an interpreter's step function specializes into.

Construction (specializing the step function per opcode) happens once per
interpreter; ``generate`` then turns a concrete code object into residual
code cheaply, resolving the late-static holes the templates left behind.
"""

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
    """Residual templates for one interpreter, plus the decoder for its code.

    ``templates`` maps a static key (opcode) to its residual template;
    ``unsupported`` records keys that failed to specialize.  ``decoder``
    is supplied by the interpreter: ``decoder(code, pc) -> (key, bindings)``,
    where ``bindings`` gives every late-static name its value at this pc.
    """

    def __init__(self, templates, decoder, static_name,
                 unsupported=None, policy=None, state_names=(),
                 leave_key=-1, ref_hole_names=()):
        self.templates = dict(templates)
        self.decoder = decoder
        # Hole names bound to the code object itself (cast to a GCREF), not
        # to anything the decoder works out -- "pycode", say.  Empty by
        # default: a toy interpreter whose "code" isn't a real RPython
        # instance (a raw byte string, in several tests) must not have this
        # attempted on it.
        self.ref_hole_names = tuple(ref_hole_names)
        # {key: (hole naming pc after the instruction, hole naming offset)};
        # a handler is reachable at their sum without the template branching
        # there itself.  Counts for reachability and loop analysis only.
        self.handler_edges = {}
        # Static key of the "leave" template used for an instruction with no
        # template of its own; -1 means none.
        self.leave_key = leave_key
        self.state_names = tuple(state_names)
        # (program, code) -> bool; declines a generated program. None accepts
        # everything.
        self.policy = policy
        self.static_name = static_name
        self.unsupported = dict(unsupported or {})
        # (pc, key) that made the last generate() decline; (-1, -1) is none.
        self.last_blocked = (-1, -1)
        # Set instead of last_blocked when the policy declined the program.
        self.decline_reason = None

    @classmethod
    def from_step_function(cls, translator, step_function, keys, decoder,
                           static_name=None, terminal_values=(-1,),
                           policy=None, graph=None, ref_hole_names=()):
        """Specialize ``step_function`` once per key; a key that fails to
        specialize is recorded, not raised on."""
        from rpython.translator.backendopt.partialeval import PartialEvaluator
        from rpython.translator.translator import graphof

        if graph is None:
            graph = graphof(translator, step_function)
        if static_name is None:
            static_name = graph.func._pe_static_args_[0]
        split_names = getattr(graph.func, "_pe_split_args_", ())
        state_names = tuple(name for name in split_names[1:])
        evaluator = PartialEvaluator(translator)

        # Keys the interpreter declared off limits, recorded like a failure.
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
        """Residual code for ``code``, entered at ``entry_pc``.  Returns None
        when a reachable instruction has no template.  ``entry_state`` seeds
        late-static values other than the pc (an operand-stack depth, say)."""
        from rpython.rtyper.annlowlevel import cast_instance_to_gcref

        blocks = {}
        # Ref holes bound to the code object itself ("pycode" for pypy), so
        # trace-time ConstPtr folding sees it as a constant, not a green.
        ref_bindings = {}
        for name in self.ref_hole_names:
            ref_bindings[name] = cast_instance_to_gcref(code)
        if entry_state is None:
            entry_state = {}
        # Not `x or {}`: mixing dict/tuple types at a merge point fails
        # RPython annotation, even on just the falsy-empty-dict branch.
        state_names = self.state_names
        # Not `dict(entry_state)`: RPython's dict has no mapping-arg
        # constructor, only `.copy()`.
        pending = [(entry_pc, entry_state.copy())]
        # Reset both: a call that succeeds, or fails for the other reason,
        # must not leave a stale explanation from some earlier call lying
        # around.
        self.last_blocked = (-1, -1)
        self.decline_reason = None
        leave_blocks = 0
        leave_pcs = {}
        handler_pcs = {}

        while pending:
            pc, state = pending.pop()
            if pc in blocks:
                # RPython has no dict comparison (binaryop.py's ne raises).
                if not _states_equal(blocks[pc].state, state):
                    # RPython's rtyper can't %r-format a dict either.
                    # Message names the pc only.
                    raise ValueError(
                        "pc %d is reachable with conflicting late-static "
                        "state" % (pc,))
                continue

            key, bindings = self.decoder(code, pc)
            if key not in self.templates:
                # Fall back to the leave template instead of declining the
                # whole program, if one was configured.
                if self.leave_key < 0 or self.leave_key not in self.templates:
                    # intmask: the sentinel below is signed, and a guest
                    # interpreter may carry its pc unsigned.
                    self.last_blocked = (intmask(pc), key)
                    return None
                key = self.leave_key
                leave_blocks += 1
                leave_pcs[intmask(pc)] = True
            template = self.templates[key]

            # Not `dict(bindings)`: RPython dict has no mapping-arg ctor.
            bindings = bindings.copy()
            bindings[self.static_name] = key
            bindings.update(state)
            block = LinkedTemplateBlock(pc, key, template, bindings,
                                        ref_bindings)
            block.state = state
            # Publish before following successors so backedges hit the cache.
            blocks[pc] = block

            # RPython's zip() only takes two iterables; index instead.
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
                # targets is already a plain list (0-2 entries) here.
                for target in targets:
                    debug_print("pe-cogen-scan edge", intmask(pc), "key",
                                key, "->", intmask(target))
                    if target not in block.successors:
                        block.successors.append(target)
                    if target not in blocks:
                        pending.append((target, next_state))
            if key in self.handler_edges:
                pc_name, offset_name = self.handler_edges[key]
                target = bindings[pc_name] + bindings[offset_name]
                debug_print("pe-cogen-scan handler edge", intmask(pc),
                            "->", intmask(target))
                handler_pcs[intmask(target)] = True
                if target not in block.successors:
                    block.successors.append(target)
                if target not in blocks:
                    pending.append((target, state))

        program = LinkedResidualProgram(entry_pc, blocks, state_names)
        program.leave_pcs = leave_pcs
        program.handler_pcs = handler_pcs
        program.leave_blocks = leave_blocks
        program = program.analyze_loops()
        # After the loop analysis, so a policy can ask about loop headers --
        # the usual reason to decline is that there is nothing here the
        # meta-tracer will re-enter often enough to pay for the install.
        if self.policy is not None and not self.policy(program, code):
            self.decline_reason = (
                "declined by policy (%d blocks)" % len(program.blocks))
            return None
        return program

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
