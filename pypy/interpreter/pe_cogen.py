"""Runtime code generation: specializes interp_step per hot PyCode."""

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
    from pypy.interpreter.pyopcode import PE_LEAVE_OPCODE

    keys = set(bytecode_spec.opmap.values())
    keys.add(PE_LEAVE_OPCODE)
    return sorted(keys)


def _loop_ends(co_code):
    """Break target per position, via SETUP_LOOP intervals, not POP_BLOCK."""
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
    """Decode one instruction; must match dispatch_bytecode exactly."""
    co_code = code.co_code
    if pc < 0 or pc >= len(co_code):
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
    """Copy of graph with every @pe.residualize callee inlined."""
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
    # copygraph drops .annotation on the fresh inputargs; carry it across.
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
    from pypy.interpreter.pyframe import PyFrame
    from pypy.interpreter.pyopcode import (
        PE_LEAVE, PE_LEAVE_OPCODE, PE_RETURN)

    step = PyFrame.interp_step.im_func
    # Need the AccessDirect variant, or every frame access forces it.
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
    # "try each alternative" closes its loop from inside an except clause.
    for name in ("SETUP_EXCEPT", "SETUP_FINALLY", "SETUP_WITH"):
        extension.handler_edges[
            getattr(bytecode_spec.opcodedesc, name).index] = ("pc", "oparg")
    return extension


def report_unsupported(extension, out=None):
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
# Bound as constants by the extension; the portal never supplies these.
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
    """Read the declaration; _pe_hole_args_ doesn't exist until annotated."""
    from pypy.interpreter.pyopcode import pedriver
    return pedriver.holes


def portal_linker(jitdriver_sd, name="linked-pypy"):
    from rpython.translator.backendopt.portal_linker import PortalLinker

    return PortalLinker(
        jitdriver_sd, PORTAL_SOURCES, RUNTIME_NAMES,
        jit_merge_point_args=JIT_MERGE_POINT_ARGS,
        null_names=("pycode",), static_name="opcode",
        split_names=LATE_STATIC_ARGUMENTS, hole_names=hole_names(), name=name)


_runtime_cogen_state = [None]

COST_PER_BYTE_NS = 20000
DEFAULT_GATE_K = 4.0

# Caps total generation time even if every individual gate check passes.
GATE_BUDGET_FRACTION = 0.25


class _GateState(object):
    """Holder: a prebuilt float folds to its seed under translation."""
    k = DEFAULT_GATE_K
    env_read = False
    spent_ns = 0.0


_gate_state = _GateState()


def _gate_k():
    """PYPY_PE_GATE overrides k, read lazily once; 0 disables the gate."""
    if not _gate_state.env_read:
        _gate_state.env_read = True
        value = os.environ.get("PYPY_PE_GATE")
        if value:
            try:
                _gate_state.k = float(value)
            except ValueError:
                pass
    return _gate_state.k


# PYPY_COGEN_EXCLUDE=name,name skips generation, to bisect a bad program.
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
    """Has tracing repaid code_size bytes k times over, within budget?"""
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
    total = 0
    for key in extension.templates:
        total += len(extension.templates[key].operations)
    line = "[pe] template operations: %d over %d templates" % (
        total, len(extension.templates))
    if out is not None:
        print >> out, line
    return line


def report_break_template(extension, out=None):
    """Debug probe for BREAK_LOOP's split-transition search."""
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
    # pc must be r_uint, matching interp_step's declared type.
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
        # Stands in for the portal as the trace's bottom frame.
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

    # Fragments compile here; the callback below never runs the codewriter.
    emitter = ProgramEmitter(
        codewriter, jitdriver_sd, "opcode", LATE_STATIC_ARGUMENTS,
        hole_names(), RUNTIME_NAMES,
        jit_merge_point_args=JIT_MERGE_POINT_ARGS)
    # One template at a time: a bad opcode declines its codes, not the build.
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

    profiler = jitdriver_sd.warmstate.warmrunnerdesc.metainterp_sd.profiler

    mainjitcode = linker.mainjitcode(codewriter)
    metadata = PEJitCodeMetadata(0, [], [], [], [], [], [])
    metadata.match_ref_index = GREEN_CODE_INDEX
    metadata.match_pc_index = GREEN_PC_INDEX
    metadata.cogen_threshold = 32
    metadata.threshold_env_var = "PYPY_COGEN_THRESHOLD"

    def _passes_gate(code, profiler):
        """False declines generation; sets metadata.soft_decline if retryable."""
        from rpython.rlib.debug import debug_print, have_debug_prints
        from rpython.rlib.jit import Counters

        if code.co_flags & CO_GENERATOR:
            # A residual program can't resume at a generator's suspended pc.
            return False
        if _excluded(code.co_name):
            return False
        code_size = len(code.co_code)
        if not _gate_allows(profiler, code_size):
            # Soft decline: retried once more tracing accrues.
            metadata.soft_decline = True
            if have_debug_prints():
                tracing = profiler.get_times(Counters.TRACING)
                optimizing = profiler.get_times(Counters.OPTIMIZING)
                tracing_ms = int((tracing + optimizing) * 1000)
                debug_print("pe-cogen gate deferred code_size=%d "
                            "tracing_ms=%d" % (code_size, tracing_ms))
            return False
        return True

    def _generate_program(code, gcref, profiler):
        from rpython.rlib.debug import debug_print, have_debug_prints

        if have_debug_prints():
            debug_print("pe-cogen code %s %s:%d" % (
                code.co_name, code.co_filename, code.co_firstlineno))
        before_ns = _cogen_ns(profiler)
        try:
            return generate_for_live_code(
                extension, linker, codewriter, code, guard, gcref,
                entry_pc=0, native_table=native_table,
                profiler=profiler)
        finally:
            _gate_state.spent_ns += _cogen_ns(profiler) - before_ns

    def _register_program(code, program):
        # Gates execute_frame's residual exception recovery.
        code._pe_has_linked_program = True
        # Assembled after finish_setup() froze liveness and jitcode tables.
        register_late_jitcode(program.jitcode,
                              program.jitcode.own_liveness_info)
        dump_jitcode(program.jitcode,
                     jitdriver_sd.warmstate.warmrunnerdesc.metainterp_sd)

    def runtime_cogen(gcref):
        profiler.start_pe_cogen()
        try:
            # Reset on every path: a stale True must never leak across refs.
            metadata.soft_decline = False
            code = cast_gcref_to_instance(PyCode, gcref)
            if code is None:
                return None
            if not _passes_gate(code, profiler):
                return None
            program = _generate_program(code, gcref, profiler)
            if program is None:
                return None
            _register_program(code, program)
            return program
        finally:
            profiler.end_pe_cogen()

    metadata.runtime_cogen = runtime_cogen
    mainjitcode.pe_metadata = metadata
    return None
