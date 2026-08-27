"""Install generated programs on a jitdriver's portal.

Set ``PE_DUMP_JITCODE`` to a directory to write each generated program's
JitCode listing there as ``<name>.jitcode``, or to ``-`` for stdout.
"""


class PortalLinker(object):
    """One interpreter's portal, and how a generated program plugs into it.

    ``portal_sources`` says where each residual parameter sits among the
    portal's boxes; ``runtime_names`` names those parameters in the step
    function.  The two are parallel.
    """

    def __init__(self, jitdriver_sd, portal_sources, runtime_names,
                 jit_merge_point_args=(), null_names=(), static_name="opcode",
                 split_names=(), hole_names=(), name="linked"):
        self.jitdriver_sd = jitdriver_sd
        self.portal_sources = tuple(portal_sources)
        self.runtime_names = tuple(runtime_names)
        self.jit_merge_point_args = tuple(jit_merge_point_args)
        self.null_names = tuple(null_names)
        self.static_name = static_name
        self.split_names = tuple(split_names)
        self.hole_names = tuple(hole_names)
        self.name = name

    def install(self, codewriter, program, whole_graph=False, guard=None,
               emitter=None, native_table=None):
        """Compile ``program`` and register it on the portal.

        ``whole_graph`` selects the older back end (one FunctionGraph for
        the whole program) over the default (assemble each template once
        and concatenate).  ``guard`` is ``(pc_index, ref_index)`` into the
        merge point's greens for a portal serving more than one program.
        ``emitter``, with the default back end, is a ``ProgramEmitter``
        whose fragments were already built by ``precompile_fragments``.
        ``native_table``, when given, selects the native pipeline
        (native_pipeline.py) over the SSARepr ``ProgramEmitter.emit``.
        """
        mainjitcode = self.mainjitcode(codewriter)
        if whole_graph:
            lowered = program.lower(
                codewriter, self.name, portal_jd=self.jitdriver_sd,
                runtime_names=self.runtime_names, null_names=self.null_names,
                jit_merge_point_args=self.jit_merge_point_args)
        elif native_table is not None:
            lowered = self._emit_native(codewriter, program, native_table)
        else:
            lowered = self._emit(codewriter, program, emitter=emitter)
        lowered.jitcode.jitdriver_sd = self.jitdriver_sd

        # Order matters: the first program to arrive creates the metadata that
        # every later one registers itself in.
        if mainjitcode.pe_metadata is None:
            program.attach_to_jitcode(mainjitcode, lowered.entry_positions)
        # Lists, not tuples: attach_linked_jitcode's argument_sources
        # parameter sees different lengths across this method's two call
        # sites, and an RPython tuple parameter can't vary in shape.
        linked_program = mainjitcode.pe_metadata.attach_linked_jitcode(
            lowered.jitcode, list(self.portal_sources), [])
        if guard is not None:
            from rpython.translator.backendopt.partialeval_template import (
                sort_ints, uses_compact_entries)
            legit_set = {}
            for pc in program.loop_headers:
                legit_set[pc] = True
            legit_set[program.entry_pc] = True
            legit_entries = []
            for pc in legit_set:
                if pc not in program.leave_pcs:
                    legit_entries.append(pc)
            sort_ints(legit_entries)
            entries = legit_entries
            if not uses_compact_entries(program):
                entries = []
                for pc in lowered.entry_positions:
                    if pc in program.leave_pcs:
                        continue
                    # Reached only by unwinding an exception, which the
                    # interpreter does generically: entering a program at
                    # an except handler's pc miscompiles (krakatau, zip).
                    if pc in program.handler_pcs:
                        continue
                    entries.append(pc)
                sort_ints(entries)
            leave_pcs = list(program.leave_pcs)
            sort_ints(leave_pcs)
            linked_program.set_guard(guard[0], entries, guard[1],
                                     legit_entries,
                                     len(program.loop_headers) > 0,
                                     leave_pcs)
        lowered.jitcode.pe_metadata.attach_linked_jitcode(
            lowered.jitcode, [], [])
        # Either back end plants jit_merge_points when the interpreter named
        # its merge point arguments; without them the metainterp falls back to
        # recognising a jump to the entry position as the loop header.
        lowered.jitcode.pe_metadata.has_merge_points = bool(
            self.jit_merge_point_args)
        lowered.linked_program = linked_program
        self._dump(lowered)
        return lowered

    def _dump(self, lowered):
        """Write the JitCode listing where PE_DUMP_JITCODE asks for it."""
        from rpython.rlib.objectmodel import we_are_translated
        # Debug tooling only, not RPython-legal as written; the
        # we_are_translated() guard keeps it out of real annotation.
        if we_are_translated():
            return
        import os
        where = os.environ.get("PE_DUMP_JITCODE")
        if not where:
            return
        jitcode = lowered.jitcode
        listing = "%s: %d bytes, entry positions %r\n%s" % (
            jitcode.name, len(jitcode.code),
            sorted(lowered.entry_positions.items()), jitcode.dump())
        if where == "-":
            print(listing)
            return
        if not os.path.isdir(where):
            os.makedirs(where)
        path = os.path.join(where, "%s.jitcode" % jitcode.name)
        index = 1
        while os.path.exists(path):
            path = os.path.join(where, "%s.%d.jitcode" % (jitcode.name, index))
            index += 1
        with open(path, "w") as stream:
            stream.write(listing)

    def mainjitcode(self, codewriter):
        """The portal's own JitCode, created on first use."""
        jitdriver_sd = self.jitdriver_sd
        if not hasattr(jitdriver_sd, "mainjitcode"):
            jitdriver_sd.mainjitcode = codewriter.callcontrol.get_jitcode(
                jitdriver_sd.portal_graph)
            jitdriver_sd.mainjitcode.jitdriver_sd = jitdriver_sd
        return jitdriver_sd.mainjitcode

    def _emit(self, codewriter, program, emitter=None):
        from rpython.translator.backendopt.jitcode_emitter import ProgramEmitter
        from rpython.translator.backendopt.partialeval_template import (
            LoweredResidualProgram)

        if emitter is None:
            emitter = ProgramEmitter(
                codewriter, self.jitdriver_sd, self.static_name,
                split_names=self.split_names, hole_names=self.hole_names,
                boundary_names=self.runtime_names,
                jit_merge_point_args=self.jit_merge_point_args,
                null_names=self.null_names)
        jitcode, entry_positions = emitter.emit(program, self.name)
        program.attach_to_jitcode(jitcode, entry_positions)
        lowered = LoweredResidualProgram(None, jitcode, entry_positions)
        lowered.emitter = emitter
        return lowered

    def _emit_native(self, codewriter, program, native_table):
        from rpython.translator.backendopt.native_pipeline import (
            emit_and_assemble_native, NativeAssembler)
        from rpython.translator.backendopt.partialeval_template import (
            LoweredResidualProgram)

        has_merge_points = bool(self.jit_merge_point_args)
        # readonly=True: may run after metainterp_sd froze, so it must never
        # grow the shared insns/descrs tables -- decline instead.
        assembler = NativeAssembler(share_with=codewriter.assembler,
                                    readonly=True)
        jitcode, entry_positions, assembler = emit_and_assemble_native(
            native_table, program, self.name,
            has_merge_points=has_merge_points, assembler=assembler)
        # Resume pcs cast to a signed 16-bit short (resumecode.py); 32767 is
        # the largest surviving pc.  AssemblerError, not assert: a translated
        # binary's assert is fatal, not catchable by the decline path.
        from rpython.jit.codewriter.assembler import AssemblerError
        from rpython.rlib.debug import debug_print
        if len(jitcode.code) > 32767:
            debug_print("runtime cogen: jitcode too large for resume pc "
                       "encoding", len(jitcode.code))
            raise AssemblerError("jitcode too large for resume pc encoding")
        jitcode.own_liveness_info = "".join(assembler.all_liveness)
        program.attach_to_jitcode(jitcode, entry_positions)
        return LoweredResidualProgram(None, jitcode, entry_positions)
