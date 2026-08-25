"""The generating extension an interpreter's step function specializes into.

In Futamura's terms, applying a partial evaluator to an interpreter yields a
compiler: not a compiled program, but a program that compiles.  That is what
this module builds.

Construction depends only on the interpreter, so it happens once, during
translation: the offline partial evaluator is applied to the step function for
every value of its static argument, producing one residual template per
instruction.  ``generate`` then supplies a concrete code object and returns
residual code for it, resolving the late-static values -- program counters,
operand bytes, an operand-stack depth -- that the templates left as holes.

The split matters because it is what the two halves cost.  Specializing the
interpreter is expensive and happens once per interpreter; generating code for
a program is cheap and happens once per program.  Nothing here needs a
codewriter or a JitCode: turning the generated program into executable form is
a separate back end, which is what still ties this to translation time.
"""

from rpython.rlib.debug import debug_print
from rpython.rlib.rarithmetic import intmask
from rpython.translator.backendopt.partialeval_template import (
    Finish, LinkedResidualProgram, LinkedTemplateBlock)


def _states_equal(a, b):
    """RPython has no dict comparison (binaryop.py's ne raises); by hand."""
    if len(a) != len(b):
        return False
    for key, value in a.items():
        if key not in b or b[key] != value:
            return False
    return True


class GeneratingExtension(object):
    """Residual templates for one interpreter, plus the decoder for its code.

    ``templates`` maps a static key -- an opcode -- to the residual template
    the partial evaluator produced for it.  ``unsupported`` records the keys it
    could not specialize, so a caller can report *why* a program was left to
    the generic interpreter instead of only that it was.

    ``decoder`` is supplied by the interpreter, never assumed here: only the
    interpreter knows how its own instructions are laid out.  It is called once
    per reachable instruction,

        decoder(code, pc) -> (key, bindings)

    where ``key`` selects the template and ``bindings`` gives every late-static
    name its value at this pc: the pc itself, each operand the step function
    declares as a hole, and anything else the decoder can work out from the
    code object but the instruction bytes alone do not determine -- a send's
    arity read from a literal table, for instance.  The static key and the
    propagated state are added here, so what reaches the back end is the whole
    specialization environment.
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
        # Edges the templates cannot express: an instruction that installs
        # a handler makes the handler reachable, though it never branches
        # there itself.  {key: (hole naming the pc after the instruction,
        # hole naming the offset)}: the handler is at their sum.  The edge
        # counts for reachability and loop analysis only.
        self.handler_edges = {}
        # Static key of a "leave" template: an instruction with no template
        # of its own gets a synthetic block built from this one instead of
        # declining the whole program, ending the residual program right
        # where the generic interpreter must resume.  -1 means "none".
        self.leave_key = leave_key
        # RPython tuple length is fixed at annotation time, so this can't be
        # rebuilt from a dict's keys -- only copied from a fixed-shape tuple.
        self.state_names = tuple(state_names)
        # Consulted once a program has been generated, with the program and
        # the code it came from; returning False declines it.  Generating is cheap and
        # installing is not -- every installed program is looked at on every
        # trace start, and the JitCodes are carried in the binary -- so which
        # programs are worth that is a judgement about the interpreter, and
        # belongs to whoever wrote it.  None accepts everything.
        self.policy = policy
        # The step function's static argument, under the name it
        # declared.  Every generated block binds the key to it, so
        # the back end never has to know which name that is.
        self.static_name = static_name
        self.unsupported = dict(unsupported or {})
        # The (pc, key) that made the most recent generate() call return
        # None because that instruction has no template, so a caller can say
        # *why* nothing came out instead of only that nothing did.
        # (-1, -1) means "nothing blocked": RPython tuples have no null.
        self.last_blocked = (-1, -1)
        # Set instead of last_blocked when every reachable instruction did
        # specialize but the policy declined the resulting program anyway
        # (the usual case: too small to be worth a JitCode and a guard).
        self.decline_reason = None

    @classmethod
    def from_step_function(cls, translator, step_function, keys, decoder,
                           static_name=None, terminal_values=(-1,),
                           policy=None, graph=None, ref_hole_names=()):
        """Specialize ``step_function`` once per key.

        A key that cannot be specialized is recorded rather than raised on: it
        only matters if a program actually uses that instruction, and whether
        one does is not known until ``generate`` walks it.
        """
        from rpython.translator.backendopt.partialeval import PartialEvaluator
        from rpython.translator.translator import graphof

        if graph is None:
            graph = graphof(translator, step_function)
        if static_name is None:
            static_name = graph.func._pe_static_args_[0]
        split_names = getattr(graph.func, "_pe_split_args_", ())
        state_names = tuple(name for name in split_names[1:])
        evaluator = PartialEvaluator(translator)

        # Keys the interpreter declared off limits: recorded exactly like one
        # that failed to specialize, so a program reaching it is left to the
        # portal and one that never does costs nothing.
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
        """Residual code for ``code``, entered at ``entry_pc``.

        Returns None when the reachable instructions include one with no
        template -- only reachability matters, so an unsupported instruction
        the program never runs costs nothing.

        ``entry_state`` seeds the late-static values that are not the pc, an
        operand-stack depth being the usual one.  Blocks stay keyed by pc: in
        well-formed code the state at a given pc is the same on every path
        reaching it, and a mismatch means that assumption does not hold, so it
        is an error rather than something to paper over.
        """
        from rpython.rtyper.annlowlevel import cast_instance_to_gcref

        blocks = {}
        # Ref holes bound to the code object itself -- "pycode" for pypy --
        # so trace-time ConstPtr pure-folding sees it as a real constant
        # rather than a runtime-loaded green.  Shared by every block: it's
        # the same code for the whole program.  Empty unless the interpreter
        # opted in via ref_hole_names.
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
                # No template for this instruction: fall back to the leave
                # template, if one was configured, rather than declining the
                # whole program.  Its bindings are still this instruction's
                # own -- "instr_start" among them is exactly what the leave
                # template needs to resume the generic loop here.
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
