"""Install generated programs on a jitdriver's portal."""


class PortalLinker(object):
    """One interpreter's portal, and how a generated program plugs into it."""

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
        """Compile ``program`` and register it on the portal."""
        mainjitcode = self.mainjitcode(codewriter)
        lowered = self._lower(
            codewriter, program, whole_graph, native_table, emitter)
        lowered.jitcode.jitdriver_sd = self.jitdriver_sd

        # The first program to arrive creates the metadata others register in.
        if mainjitcode.pe_metadata is None:
            program.attach_to_jitcode(mainjitcode, lowered.entry_positions)
        linked_program = mainjitcode.pe_metadata.attach_linked_jitcode(
            lowered.jitcode, list(self.portal_sources), [])
        if guard is not None:
            self._attach_guard(linked_program, program, lowered, guard)
        lowered.jitcode.pe_metadata.attach_linked_jitcode(
            lowered.jitcode, [], [])
        lowered.jitcode.pe_metadata.has_merge_points = bool(
            self.jit_merge_point_args)
        lowered.linked_program = linked_program
        self._dump(lowered)
        return lowered

    def _lower(self, codewriter, program, whole_graph, native_table, emitter):
        """Pick one of the three lowering strategies and run it."""
        if whole_graph:
            return self._lower_whole_graph(codewriter, program)
        if native_table is not None:
            return self._emit_native(codewriter, program, native_table)
        return self._emit(codewriter, program, emitter=emitter)

    def _lower_whole_graph(self, codewriter, program):
        return program.lower(
            codewriter, self.name, portal_jd=self.jitdriver_sd,
            runtime_names=self.runtime_names, null_names=self.null_names,
            jit_merge_point_args=self.jit_merge_point_args)

    def _attach_guard(self, linked_program, program, lowered, guard):
        legit_entries, entries, leave_pcs = program.guard_entries(
            lowered.entry_positions)
        linked_program.set_matcher(guard[0], entries, guard[1],
                                 legit_entries,
                                 len(program.loop_headers) > 0,
                                 leave_pcs)

    def _dump(self, lowered):
        """Write the JitCode listing where PE_DUMP_JITCODE asks for it."""
        from rpython.rlib.objectmodel import we_are_translated
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
                runtime_names=self.runtime_names,
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
        # readonly: may run after metainterp_sd froze; never grow shared tables.
        assembler = NativeAssembler(share_with=codewriter.assembler,
                                    readonly=True)
        jitcode, entry_positions, assembler = emit_and_assemble_native(
            native_table, program, self.name,
            has_merge_points=has_merge_points, assembler=assembler)
        # 32767: resume pcs are a signed 16-bit short (resumecode.py).
        from rpython.jit.codewriter.assembler import AssemblerError
        from rpython.rlib.debug import debug_print
        if len(jitcode.code) > 32767:
            debug_print("runtime cogen: jitcode too large for resume pc "
                       "encoding", len(jitcode.code))
            raise AssemblerError("jitcode too large for resume pc encoding")
        jitcode.own_liveness_info = "".join(assembler.all_liveness)
        program.attach_to_jitcode(jitcode, entry_positions)
        return LoweredResidualProgram(None, jitcode, entry_positions)
