"""Partial-evaluation support for the CPython bytecode interpreter."""

import os
import sys

from pypy.interpreter.pycode import BytecodeCorruption
from pypy.tool.stdlib_opcode import bytecode_spec
from rpython.translator.backendopt.generating_extension import (
    GeneratingExtension)

opcodedesc = bytecode_spec.opcodedesc
HAVE_ARGUMENT = bytecode_spec.HAVE_ARGUMENT
EXTENDED_ARG = opcodedesc.EXTENDED_ARG.index


def opcode_keys():
    """Every opcode index, plus the synthetic PE_LEAVE_OPCODE fallback."""
    from pypy.interpreter.pyopcode import PE_LEAVE_OPCODE

    keys = set(bytecode_spec.opmap.values())
    keys.add(PE_LEAVE_OPCODE)
    return sorted(keys)


def _loop_ends(co_code):
    """For each position, the break target of the innermost enclosing loop.

    A block-stack POP_BLOCK scan is ambiguous for break-in-try (both compile
    to POP_BLOCK); use SETUP_LOOP's jump-offset intervals instead, assigning
    each position the innermost (last-visited) interval containing it.
    """
    n = len(co_code)
    ends = [-1] * n
    setup_loop = opcodedesc.SETUP_LOOP.index
    intervals = []
    position = 0
    extended_arg = 0
    while position < n:
        opcode = ord(co_code[position])
        if opcode >= HAVE_ARGUMENT:
            lo = ord(co_code[position + 1])
            hi = ord(co_code[position + 2])
            oparg = (extended_arg << 16) | (hi << 8) | lo
            length = 3
        else:
            oparg = 0
            length = 1
        if opcode == EXTENDED_ARG:
            extended_arg = oparg
            position += length
            continue
        extended_arg = 0
        if opcode == setup_loop:
            body_start = position + length
            intervals.append((body_start, body_start + oparg))
        position += length
    for body_start, target in intervals:
        pos = body_start
        stop = target if target < n else n
        while pos < stop:
            ends[pos] = target
            pos += 1
    return ends


def decode_instruction(code, pc):
    """Decode one instruction; must match dispatch_bytecode's semantics
    exactly, since the bindings fill the holes its templates left."""
    co_code = code.co_code
    if pc < 0 or pc >= len(co_code):
        # Decline the code object rather than abort the process.
        raise BytecodeCorruption
    opcode = ord(co_code[pc])
    next_instr = pc + 1
    if opcode >= HAVE_ARGUMENT:
        lo = ord(co_code[next_instr])
        hi = ord(co_code[next_instr + 1])
        next_instr += 2
        oparg = (hi * 256) | lo
    else:
        oparg = 0

    while opcode == EXTENDED_ARG:
        opcode = ord(co_code[next_instr])
        if opcode < HAVE_ARGUMENT:
            raise BytecodeCorruption
        lo = ord(co_code[next_instr + 1])
        hi = ord(co_code[next_instr + 2])
        next_instr += 3
        oparg = (oparg * 65536) | (hi * 256) | lo

    if opcode == opcodedesc.BREAK_LOOP.index:
        break_target = _loop_ends(co_code)[pc]
        if break_target < 0:
            # No enclosing SETUP_LOOP: malformed code, decline it.
            raise BytecodeCorruption
    else:
        break_target = -1
    bindings = {
        "pc": next_instr,
        "oparg": oparg,
        "instr_start": pc,
        "break_target": break_target,
    }
    return opcode, bindings


def inline_residualized(translator, graph):
    """Copy of ``graph`` with every ``@pe.residualize`` callee inlined, so
    the partial evaluator sees its body; the generic interpreter still
    calls the original helper untouched."""
    from rpython.flowspace.model import checkgraph, copygraph
    from rpython.translator.backendopt.canraise import RaiseAnalyzer
    from rpython.translator.backendopt.inline import (
        CannotInline, inline_function)
    from rpython.translator.simplify import cleanup_graph, get_graph

    def residualized_callees(g):
        found = {}
        for block in g.iterblocks():
            for op in block.operations:
                if op.opname != "direct_call":
                    continue
                callee = get_graph(op.args[0], translator)
                if callee is not None and \
                        getattr(callee.func, "_pe_residualize_", False):
                    found[callee] = True
        return found

    if not residualized_callees(graph):
        return graph

    copied = copygraph(graph)
    # copygraph drops .annotation on the fresh inputarg Variables; carry it
    # across by position (entry-block arity/order is preserved).
    for old, new in zip(graph.startblock.inputargs,
                        copied.startblock.inputargs):
        new.annotation = old.annotation
    lltype_to_classdef = translator.rtyper.lltype_to_classdef_mapping()
    raise_analyzer = RaiseAnalyzer(translator)
    failed = {}
    for _round in range(10):
        callees = [g for g in residualized_callees(copied) if g not in failed]
        if not callees:
            break
        for callee in callees:
            try:
                inline_function(translator, callee, copied,
                                lltype_to_classdef, raise_analyzer,
                                cleanup=False)
            except CannotInline as error:
                print '[pe] could not inline %s: %s' % (
                    callee.name, error)
                failed[callee] = True
    cleanup_graph(copied)
    checkgraph(copied)
    return copied


def build_generating_extension(translator):
    """Specialize interp_step once per opcode."""
    from pypy.interpreter.pyframe import PyFrame
    from pypy.interpreter.pyopcode import (
        PE_LEAVE, PE_LEAVE_OPCODE, PE_RETURN)

    step = PyFrame.interp_step.im_func
    # Need the AccessDirect variant: the plain one forces the virtualizable
    # on every frame access, which a generated program would then deopt on.
    graph = None
    for candidate in translator.graphs:
        if getattr(candidate, "func", None) is step and \
                "AccessDirect" in candidate.name:
            graph = candidate
            break
    graph_name = graph.name if graph is not None else 'PLAIN (no AccessDirect)'
    print '[pe] step graph:', graph_name
    if graph is not None:
        graph = inline_residualized(translator, graph)
    extension = GeneratingExtension.from_step_function(
        translator, step, opcode_keys(),
        decode_instruction, terminal_values=(PE_LEAVE, PE_RETURN),
        graph=graph, ref_hole_names=("pycode",))
    extension.leave_key = PE_LEAVE_OPCODE
    # The "try each alternative" pattern closes its loop from inside an
    # except clause; the loop header is only recognised via this edge.
    for name in ("SETUP_EXCEPT", "SETUP_FINALLY", "SETUP_WITH"):
        extension.handler_edges[
            getattr(bytecode_spec.opcodedesc, name).index] = ("pc", "oparg")
    return extension


def report_unsupported(extension, out=None):
    """One line per opcode that could not be specialized, and why."""
    from pypy.tool.stdlib_opcode import opcode_method_names

    lines = []
    for key in sorted(extension.unsupported):
        error = extension.unsupported[key]
        message = str(error).splitlines()[0] if str(error) else ""
        name = opcode_method_names[key] if key < len(
            opcode_method_names) else str(key)
        lines.append("[pe] no template for %s: %s: %s" % (
            name, error.__class__.__name__, message))
        if name == "BREAK_LOOP" and hasattr(error, "pe_traceback"):
            lines.append(error.pe_traceback)
    if out is not None:
        for line in lines:
            print >> out, line
    return lines


# Portal arguments (greens then reds); everything below derives from this.
PORTAL_ARGUMENTS = (
    ("next_instr", "pc"),                        # green
    ("is_being_profiled", "is_being_profiled"),  # green
    ("pycode", "pycode"),                        # green
    ("frame", "self"),                           # reds from here on
    ("ec", "ec"),
)
# Bound as constants by the generating extension; the portal never
# supplies these.
LATE_STATIC_ARGUMENTS = ("pc",)

JIT_MERGE_POINT_ARGS = tuple(step for _green, step in PORTAL_ARGUMENTS)
RUNTIME_NAMES = tuple(step for _g, step in PORTAL_ARGUMENTS
                      if step not in LATE_STATIC_ARGUMENTS)
PORTAL_SOURCES = tuple(index for index, (_g, step)
                       in enumerate(PORTAL_ARGUMENTS)
                       if step not in LATE_STATIC_ARGUMENTS)
GREEN_PC_INDEX = JIT_MERGE_POINT_ARGS.index("pc")
GREEN_CODE_INDEX = JIT_MERGE_POINT_ARGS.index("pycode")


def hole_names():
    """Read the declaration, not interp_step's _pe_hole_args_, which does
    not exist yet for a caller that only imported the module."""
    from pypy.interpreter.pyopcode import pedriver
    return pedriver.holes


def portal_linker(jitdriver_sd, name="linked-pypy"):
    from rpython.translator.backendopt.portal_linker import PortalLinker

    return PortalLinker(
        jitdriver_sd, PORTAL_SOURCES, RUNTIME_NAMES,
        jit_merge_point_args=JIT_MERGE_POINT_ARGS,
        null_names=("pycode",), static_name="opcode",
        split_names=LATE_STATIC_ARGUMENTS, hole_names=hole_names(), name=name)


# List holder: a plain module var would fold to a translation constant.
_runtime_cogen_state = [None]

COST_PER_BYTE_NS = 20000
# Cumulative tracing time must cover this many multiples of a generation's
# estimated cost before it is worth doing.
DEFAULT_GATE_K = 4.0

# At most this fraction of the process's measured tracing time may be spent
# generating, bounding the total loss when many per-generation checks pass
# but never individually repay.
GATE_BUDGET_FRACTION = 0.25


class _GateState(object):
    """Holder, not a module float: a prebuilt float folds to its seed
    under translation (see _FoldedLoads in native_pipeline.py)."""
    k = DEFAULT_GATE_K
    env_read = False
    spent_ns = 0.0


_gate_state = _GateState()


def _gate_k():
    """k for the cost-model gate; PYPY_PE_GATE overrides it, read lazily
    once at runtime.  PYPY_PE_GATE=0 disables the gate (always generate)."""
    if not _gate_state.env_read:
        _gate_state.env_read = True
        value = os.environ.get("PYPY_PE_GATE")
        if value:
            try:
                _gate_state.k = float(value)
            except ValueError:
                pass
    return _gate_state.k


# Debugging aid: PYPY_COGEN_EXCLUDE=name,name never generates for these
# code object names, to bisect a misbehaving program at run time.
_exclude_state = [None]


def _excluded(co_name):
    names = _exclude_state[0]
    if names is None:
        value = os.environ.get("PYPY_COGEN_EXCLUDE")
        names = value.split(",") if value else []
        _exclude_state[0] = names
    for name in names:
        if name == co_name:
            return True
    return False


def _gate_allows(profiler, code_size):
    """Has this process traced enough to repay generating a program of
    'code_size' bytes, k times over, within the remaining budget?"""
    from rpython.rlib.jit import Counters

    k = _gate_k()
    if k == 0.0:
        return True
    tracing_ns = (profiler.get_times(Counters.TRACING) +
                  profiler.get_times(Counters.OPTIMIZING)) * 1e9
    cost_ns = code_size * COST_PER_BYTE_NS
    if tracing_ns < k * cost_ns:
        return False
    if _gate_state.spent_ns + cost_ns > GATE_BUDGET_FRACTION * tracing_ns:
        return False
    return True


def _cogen_ns(profiler):
    from rpython.rlib.jit import Counters
    return (profiler.get_times(Counters.PE_COGEN_SCAN) +
            profiler.get_times(Counters.PE_COGEN_INSTALL)) * 1e9


def stamp_after_make_jitcodes(mainjitcode):
    """Runs after codewriter.make_jitcodes(), strictly before finish_setup."""
    state = _runtime_cogen_state[0]
    if state is None:
        return
    from rpython.translator.backendopt.jitcode_emitter import (
        stamp_descr_indices, register_native_insn_coverage)
    codewriter, native_table = state
    stamp_descr_indices(codewriter, native_table)
    register_native_insn_coverage(codewriter, native_table)


def report_template_size(extension, out=None):
    """Total residual operations across all templates."""
    total = 0
    for key in extension.templates:
        total += len(extension.templates[key].operations)
    line = "[pe] template operations: %d over %d templates" % (
        total, len(extension.templates))
    if out is not None:
        print >> out, line
    return line


def report_break_template(extension, out=None):
    """Debug probe: why does BREAK_LOOP's fragment compile find no split
    transition when the template itself carries a Continue?"""
    from pypy.interpreter.pyopcode import opcodedesc
    from rpython.translator.backendopt.partialeval import (
        _find_split_transitions)

    key = opcodedesc.BREAK_LOOP.index
    if key not in extension.templates:
        return
    template = extension.templates[key]
    lines = ["[pe] BREAK terminators: %s" % (
        [t.__class__.__name__ for t in template.terminators],)]
    graph = template.residual_graph
    if graph is not None:
        transitions = _find_split_transitions(graph)
        lines.append("[pe] BREAK residual_graph transitions: %d" %
                     len(transitions))
        lines.append("[pe] BREAK returnblock inputs: %s" %
                     (graph.returnblock.inputargs,))
        for block in graph.iterblocks():
            lines.append("[pe] B %s exits=%d switch=%r" % (
                block, len(block.exits), block.exitswitch))
            for op in block.operations:
                lines.append("[pe]    %s" % (op,))
            for link in block.exits:
                lines.append("[pe]    -> %s args=%s case=%r" % (
                    link.target, link.args, link.exitcase))
    else:
        lines.append("[pe] BREAK residual_graph: None")
    if out is not None:
        for line in lines:
            print >> out, line
    return lines


def report_unresolvable(extension, out=None):
    """Templates whose targets the runtime resolver cannot evaluate."""
    from pypy.tool.stdlib_opcode import opcode_method_names

    from rpython.rlib.rarithmetic import r_uint

    lines = []
    # pc must be r_uint, as interp_step declares it, or resolving a target
    # that mixes it with a signed oparg fails here for an unrelated reason.
    bindings = {"pc": r_uint(0), "oparg": 0}
    for key in sorted(extension.templates):
        try:
            extension.templates[key].resolve_targets(bindings)
        except Exception as error:
            name = opcode_method_names[key] if key < len(
                opcode_method_names) else str(key)
            lines.append("[pe] unresolvable targets in %s: %s: %s" % (
                name, error.__class__.__name__, error))
    if out is not None:
        for line in lines:
            print >> out, line
    return lines


def _install_pe_recover(codewriter, jitdriver_sd, translator):
    """Let the tracer unwind an escaped guest exception in the trace."""
    from pypy.interpreter.error import OperationError
    from pypy.interpreter.pyframe import PyFrame
    from rpython.rtyper.rclass import getclassrepr
    from rpython.translator.translator import graphof

    def helper_jitcode(func):
        jitcode = codewriter.callcontrol.get_jitcode(
            graphof(translator, func))
        # Stands in for the portal as the trace's bottom frame, like a
        # linked program does.
        jitcode.jitdriver_sd = jitdriver_sd
        jitcode.pe_is_linked = True
        return jitcode

    jitdriver_sd.pe_recover_jitcode = helper_jitcode(
        PyFrame.pe_recover.im_func)
    # Only recognised, never a root frame.
    jitdriver_sd.pe_resume_jitcode = codewriter.callcontrol.get_jitcode(
        graphof(translator, PyFrame.pe_resume.im_func))
    classdef = translator.annotator.bookkeeper.getuniqueclassdef(
        OperationError)
    jitdriver_sd.pe_recover_exc_class = getclassrepr(
        translator.rtyper, classdef).getvtable()


def install_runtime_cogen(codewriter, jitdriver_sd, translator):
    """Translation-time entry point: wire runtime cogen onto the portal."""
    from pypy.interpreter.pycode import CO_GENERATOR, PyCode
    from rpython.jit.codewriter.jitcode import (
        PEJitCodeMetadata, dump_jitcode, register_late_jitcode)
    from rpython.rtyper.annlowlevel import cast_gcref_to_instance
    from rpython.translator.backendopt.jitcode_emitter import ProgramEmitter
    from rpython.translator.backendopt.runtime_cogen import (
        generate_for_live_code)

    extension = build_generating_extension(translator)
    linker = portal_linker(jitdriver_sd, "linked-pypy-runtime-cogen")
    _install_pe_recover(codewriter, jitdriver_sd, translator)
    guard = (GREEN_PC_INDEX, GREEN_CODE_INDEX)

    # Fragments are compiled here, at the RPython/translation boundary; the
    # callback below never runs the codewriter, so it can run once translated.
    emitter = ProgramEmitter(
        codewriter, jitdriver_sd, "opcode", LATE_STATIC_ARGUMENTS,
        hole_names(), RUNTIME_NAMES,
        jit_merge_point_args=JIT_MERGE_POINT_ARGS)
    # One template at a time: an opcode the emitter cannot compile becomes a
    # decline for the code objects that use it, not a failed build.
    for key in sorted(extension.templates):
        try:
            emitter.precompile_fragments({key: extension.templates[key]})
        except Exception as error:
            import traceback
            error.pe_traceback = traceback.format_exc()
            extension.unsupported[key] = error
            del extension.templates[key]
    report_unsupported(extension, sys.stdout)
    print >> sys.stdout, report_template_size(extension)
    native_table = emitter.native_table()
    _runtime_cogen_state[0] = (codewriter, native_table)

    # Captured by reference, not by value: reading a field off it here would
    # fold to a translation constant, but .times[] is mutated at run time.
    profiler = jitdriver_sd.warmstate.warmrunnerdesc.metainterp_sd.profiler

    mainjitcode = linker.mainjitcode(codewriter)
    metadata = PEJitCodeMetadata(0, [], [], [], [], [], [])
    metadata.guard_ref_index = GREEN_CODE_INDEX
    metadata.guard_pc_index = GREEN_PC_INDEX
    metadata.cogen_threshold = 32
    metadata.threshold_env_var = "PYPY_COGEN_THRESHOLD"

    def runtime_cogen(gcref):
        from rpython.rlib.debug import debug_print, have_debug_prints
        from rpython.rlib.jit import Counters
        profiler.start_pe_cogen()
        try:
            # Reset on every path, decline included: a stale True from a
            # previous ref's gate defer must never leak into this result.
            metadata.soft_decline = False
            code = cast_gcref_to_instance(PyCode, gcref)
            if code is None:
                return None
            if code.co_flags & CO_GENERATOR:
                # A generator resumes at its suspended pc; a residual
                # program only runs from a block boundary to its own exit.
                return None
            if _excluded(code.co_name):
                return None
            code_size = len(code.co_code)
            if not _gate_allows(profiler, code_size):
                # Decline softly: retried once more tracing accrues, not
                # cached as a permanent decline.
                metadata.soft_decline = True
                if have_debug_prints():
                    tracing = profiler.get_times(Counters.TRACING)
                    optimizing = profiler.get_times(Counters.OPTIMIZING)
                    tracing_ms = int((tracing + optimizing) * 1000)
                    debug_print("pe-cogen gate deferred code_size=%d "
                                "tracing_ms=%d" % (code_size, tracing_ms))
                return None
            if have_debug_prints():
                debug_print("pe-cogen code %s %s:%d" % (
                    code.co_name, code.co_filename, code.co_firstlineno))
            before_ns = _cogen_ns(profiler)
            try:
                program = generate_for_live_code(
                    extension, linker, codewriter, code, guard, gcref,
                    entry_pc=0, native_table=native_table,
                    profiler=profiler)
            finally:
                _gate_state.spent_ns += _cogen_ns(profiler) - before_ns
            if program is None:
                return None
            # Gates execute_frame's residual exception recovery.
            code._pe_has_linked_program = True
            # Assembled after finish_setup() froze liveness and jitcode tables.
            register_late_jitcode(program.jitcode,
                                  program.jitcode.own_liveness_info)
            dump_jitcode(program.jitcode,
                         jitdriver_sd.warmstate.warmrunnerdesc.metainterp_sd)
            return program
        finally:
            profiler.end_pe_cogen()

    metadata.runtime_cogen = runtime_cogen
    mainjitcode.pe_metadata = metadata
    return None
