"""Baseline compiler tier: compile residual JitCode to machine code
directly, instead of only tracing it (V8-Sparkplug-style method
compilation on top of the existing cogen residual programs).

Background: rpython/translator/backendopt/runtime_cogen.py builds a
residual JitCode for a hot guest code object (see PE_ARCHITECTURE.md).
Today the only consumer of that JitCode is the trace-recording
metainterp (pyjitpl.py); this module is the stub for a second consumer
that runs/compiles it directly, without waiting for the JIT to trace
and optimize it first. The milestones below are meant to be built and
landed in order; each one stands alone.

Design note: residual JitCode contains inline_call_* ops wherever the
generating extension inlined a callee's template into the caller's
program (native_pipeline.py / jitcode_emitter.py decide this at
generation time, independent of this module). The baseline tier does
NOT re-inline anything -- it must not second-guess cogen's inlining
choice. Where inline_call_* appears, baseline emits a real call to the
callee's own compiled (or interpreted) entry point, exactly as the
generic JitCode interpreter would. Teaching baseline to inline residual
calls itself is a later, separate decision -- not part of milestones 1-4.
"""


class BaselineTier(object):
    """Owns the four milestones of the baseline compiler tier.

    Each method below is a stub for one milestone; see its docstring
    for what it must do and which existing module to model it on.
    """

    def run_from_pyjitpl(self, jitcode, position, metainterp):
        """Milestone 1: standalone execution loop for a residual JitCode,
        outside of tracing.

        Model this on rpython/jit/metainterp/blackhole.py: the
        BlackholeInterpreter's per-opcode dispatch loop
        (_run_forever / _resume_mainloop) and its portal entry point
        convert_and_run_from_pyjitpl(), which copies live frame state
        out of the metainterp and then drives execution until the
        program exits. This milestone must interpret (not yet compile)
        the residual JitCode standalone: no tracing, no machine code.
        It is the thing milestone 2 will start replacing block by
        block. Must be fully testable untranslated (unit tests only,
        no RPython translation needed).
        """
        raise NotImplementedError

    def compile_block(self, jitcode, start_position):
        """Milestone 2: compile one residual block to machine code using
        the EXISTING trace backend.

        Each residual block is linear and its boundaries already
        materialize the full frame state (pe_bailout_point ops --
        see PE_ARCHITECTURE.md "pe_bailout_point"), so a block can be
        compiled exactly like a bridge: model this on
        rpython/jit/metainterp/compile.py's bridge compilation path
        (must_compile / the bridge-from-guard-failure code), feeding
        the block's ops through the same optimizer/backend pipeline a
        bridge uses. Chain compiled blocks with plain jumps at their
        boundaries. All values are spilled to the frame at every block
        boundary -- there is no cross-block register allocation in
        this milestone.
        """
        raise NotImplementedError

    def enter_or_run(self, jitdriver_sd, greenargs, redargs):
        """Milestone 3: tier integration into the portal.

        The portal must enter compiled baseline code when a residual
        program exists for the current (code, pc) but no trace/loop
        token has been compiled for it yet, and guard failures coming
        out of traced loops must land back in baseline code instead of
        falling through to the generic JitCode interpreter. Hook
        points to extend: rpython/jit/metainterp/warmstate.py
        (maybe_compile_and_run, execute_assembler -- see where they
        currently choose between "run the loop" and "fall through to
        blackhole/interpreter") and rpython/jit/metainterp/warmspot.py
        (ll_portal_runner, the portal's actual entry function).
        """
        raise NotImplementedError

    def report_stats(self, profiler):
        """Milestone 4: evaluation support -- warmup curves and
        guard-fail tail cost.

        Reuse the counters that already exist rather than inventing
        new ones: the "pe insns generic/portal/residual" counters
        incremented in pyjitpl.py's run_one_step, and the jit-summary
        report built in jitprof.py. Extend both with a baseline-tier
        breakdown (instructions executed in compiled baseline code vs.
        interpreted, and time spent per guard-failure tail) so warmup
        curves and tail cost can be compared against the pure-cogen
        and pure-tracing configurations already measured on this
        branch.
        """
        raise NotImplementedError
