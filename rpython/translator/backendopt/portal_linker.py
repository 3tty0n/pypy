"""Install generated programs on a jitdriver's portal.

Set ``PE_DUMP_JITCODE`` to a directory to write each generated program's
JitCode listing there as ``<name>.jitcode``, or to ``-`` for stdout.  The
listing is the assembled form: byte positions, labels, registers and liveness,
which is what tells apart a program that installed from one that installed and
then jumps somewhere it should not.

``GeneratingExtension`` produces a ``LinkedResidualProgram``; two back ends turn
that into a JitCode.  What comes after is the same for every interpreter: hang
the block map on the portal's JitCode, register the generated one under a guard,
and keep the metadata consistent with which back end ran.  Getting that order
wrong is silent -- the program installs and is simply never entered -- so it
lives here once rather than in each interpreter's offline module.

What differs between interpreters is only the argument table, which a
``PortalLinker`` is constructed with.
"""


class PortalLinker(object):
    """One interpreter's portal, and how a generated program plugs into it.

    ``portal_sources`` says where each residual parameter sits among the
    portal's boxes; ``runtime_names`` names those parameters in the step
    function.  The two are parallel, and the caller should derive both from one
    table rather than keep them in step by hand.
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
               emitter=None):
        """Compile ``program`` and register it on the portal.

        The default back end assembles each residual template once and
        concatenates; ``whole_graph`` selects the older one, which builds a
        single FunctionGraph for the program and hands that to the codewriter.
        The two produce equivalent code, but only the default can run the
        codewriter before a program is known.

        ``guard`` is ``(pc_index, ref_index)`` into the merge point's greens for
        a portal that serves more than one program; None means the portal has
        only this one.

        ``emitter``, only meaningful with the default back end, is a
        ``ProgramEmitter`` whose fragments were already built by
        ``precompile_fragments`` -- passing one runs no codewriter at all.
        None builds a fresh, empty-cached ``ProgramEmitter`` as before.
        """
        mainjitcode = self.mainjitcode(codewriter)
        if whole_graph:
            lowered = program.lower(
                codewriter, self.name, portal_jd=self.jitdriver_sd,
                runtime_names=self.runtime_names, null_names=self.null_names,
                jit_merge_point_args=self.jit_merge_point_args)
        else:
            lowered = self._emit(codewriter, program, emitter=emitter)
        lowered.jitcode.jitdriver_sd = self.jitdriver_sd

        # Order matters: the first program to arrive creates the metadata that
        # every later one registers itself in.
        if mainjitcode.pe_metadata is None:
            program.attach_to_jitcode(mainjitcode, lowered.entry_positions)
        linked_program = mainjitcode.pe_metadata.attach_linked_jitcode(
            lowered.jitcode, self.portal_sources, ())
        if guard is not None:
            # Every block boundary the emitted code has an entry position for
            # -- not just loop headers and the entry -- so a greenkey that
            # goes hot at any block the residual program already covers still
            # matches this program instead of falling back to a generic
            # portal trace for a region that duplicates it.
            entries = sorted(lowered.entry_positions)
            # The loop headers and the entry: exactly the pcs where a trace
            # may legitimately start without duplicating a loop this program
            # already provides.  A stricter subset of `entries` above, used
            # by pe_tick_suppressed (warmstate.py) to tell a genuine loop
            # start apart from a mid-block pc some other trace's tail landed
            # on inside this program's coverage.
            legit_entries = sorted(set(program.loop_headers) |
                                   set([program.entry_pc]))
            linked_program.set_guard(guard[0], entries, guard[1],
                                     legit_entries)
        lowered.jitcode.pe_metadata.attach_linked_jitcode(
            lowered.jitcode, (), ())
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
                jit_merge_point_args=self.jit_merge_point_args)
        jitcode, entry_positions = emitter.emit(program, self.name)
        # Leave the same state ``lower`` does: the block map lives on the
        # JitCode, and the metainterp reads it from there.
        program.attach_to_jitcode(jitcode, entry_positions)
        lowered = LoweredResidualProgram(None, jitcode, entry_positions)
        lowered.emitter = emitter
        return lowered
