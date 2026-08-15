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

from rpython.translator.backendopt.partialeval_template import (
    Finish, LinkedResidualProgram, LinkedTemplateBlock)


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
                 unsupported=None, policy=None):
        self.templates = dict(templates)
        self.decoder = decoder
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

    @classmethod
    def from_step_function(cls, translator, step_function, keys, decoder,
                           static_name=None, terminal_values=(-1,),
                           policy=None):
        """Specialize ``step_function`` once per key.

        A key that cannot be specialized is recorded rather than raised on: it
        only matters if a program actually uses that instruction, and whether
        one does is not known until ``generate`` walks it.
        """
        from rpython.translator.backendopt.partialeval import PartialEvaluator
        from rpython.translator.translator import graphof

        graph = graphof(translator, step_function)
        if static_name is None:
            static_name = graph.func._pe_static_args_[0]
        evaluator = PartialEvaluator(translator)

        # Keys the interpreter declared off limits: recorded exactly like one
        # that failed to specialize, so a program reaching it is left to the
        # portal and one that never does costs nothing.
        skipped = getattr(graph.func, "_pe_skip_keys_", ())
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
                   policy=policy)

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
        blocks = {}
        state_names = tuple(sorted(entry_state or ()))
        pending = [(entry_pc, dict(entry_state or {}))]

        while pending:
            pc, state = pending.pop()
            if pc in blocks:
                if blocks[pc].state != state:
                    raise ValueError(
                        "pc %d is reachable with conflicting late-static "
                        "state %r and %r" % (pc, blocks[pc].state, state))
                continue

            key, bindings = self.decoder(code, pc)
            if key not in self.templates:
                return None
            template = self.templates[key]

            bindings = dict(bindings)
            bindings[self.static_name] = key
            bindings.update(state)
            block = LinkedTemplateBlock(pc, key, template, bindings)
            block.state = state
            # Publish before following successors so backedges hit the cache.
            blocks[pc] = block

            for terminator, targets, next_state in zip(
                    template.terminators,
                    template.resolve_targets(bindings),
                    template.resolve_state(bindings)):
                if isinstance(terminator, Finish):
                    block.has_finish = True
                    continue
                if not isinstance(targets, tuple):
                    targets = (targets,)
                for target in targets:
                    if target not in block.successors:
                        block.successors.append(target)
                    if target not in blocks:
                        pending.append((target, next_state))

        program = LinkedResidualProgram(entry_pc, blocks)
        program.state_names = state_names
        program = program.analyze_loops()
        # After the loop analysis, so a policy can ask about loop headers --
        # the usual reason to decline is that there is nothing here the
        # meta-tracer will re-enter often enough to pay for the install.
        if self.policy is not None and not self.policy(program, code):
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
