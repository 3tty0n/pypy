"""Partial-evaluation support for the CPython bytecode interpreter.

Mirrors rpython/jit/tl/tla/offline.py.  ``PyFrame.interp_step`` declares
``opcode`` as offline-static and ``next_instr`` as late-static, so the partial
evaluator can build one residual template per bytecode at translation time.

Unlike PySOM, there is no translation-time variant: a ``PyCode`` is built at
run time by ``compile``, ``exec`` and import, so the set of code objects to
specialize cannot be enumerated when the binary is built.
"""

import sys

from pypy.interpreter.pycode import BytecodeCorruption
from pypy.tool.stdlib_opcode import bytecode_spec
from rpython.translator.backendopt.generating_extension import (
    GeneratingExtension)

opcodedesc = bytecode_spec.opcodedesc
HAVE_ARGUMENT = bytecode_spec.HAVE_ARGUMENT
EXTENDED_ARG = opcodedesc.EXTENDED_ARG.index


def opcode_keys():
    """Every opcode index the interpreter has an implementation for."""
    return sorted(set(bytecode_spec.opmap.values()))


def decode_instruction(code, pc):
    """Decode the instruction of ``code`` starting at ``pc``.

    ``code`` is anything carrying a ``co_code`` string; the linker uses a live
    ``PyCode``.  The decoding must match ``dispatch_bytecode``'s exactly, since
    the bindings returned here fill the holes its residual templates left:
    ``next_instr`` is the position *after* the whole instruction, and an
    EXTENDED_ARG prefix is folded into the argument of what follows it rather
    than reported as an instruction of its own.
    """
    assert pc >= 0
    co_code = code.co_code
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

    bindings = {
        "pc": next_instr,
        "oparg": oparg,
        "instr_start": pc,
    }
    return opcode, bindings


def build_generating_extension(translator):
    """Specialize interp_step once per opcode.

    An opcode with no template is not fatal: only code objects that actually
    reach one are left to the generic dispatch loop.
    """
    from pypy.interpreter.pyframe import PyFrame
    from pypy.interpreter.pyopcode import PE_LEAVE, PE_RETURN

    return GeneratingExtension.from_step_function(
        translator, PyFrame.interp_step.im_func, opcode_keys(),
        decode_instruction, terminal_values=(PE_LEAVE, PE_RETURN))


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
    if out is not None:
        for line in lines:
            print >> out, line
    return lines


# The portal's arguments: greens then reds in jitdriver order, paired with the
# name each carries in interp_step.  Everything below derives from this table,
# so the orders cannot drift apart.
PORTAL_ARGUMENTS = (
    ("next_instr", "pc"),                        # green
    ("is_being_profiled", "is_being_profiled"),  # green
    ("pycode", "pycode"),                        # green
    ("frame", "self"),                           # reds from here on
    ("ec", "ec"),
)
# Bound as constants by the generating extension, so the portal never supplies
# them.
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
    """Read the declaration, not interp_step's _pe_hole_args_: the latter is
    attached when pe_merge_point is annotated, so it does not exist yet when
    a caller only imported the module."""
    from pypy.interpreter.pyopcode import pedriver
    return pedriver.holes


def portal_linker(jitdriver_sd, name="linked-pypy"):
    """How a generated program plugs into the interpreter's portal."""
    from rpython.translator.backendopt.portal_linker import PortalLinker

    return PortalLinker(
        jitdriver_sd, PORTAL_SOURCES, RUNTIME_NAMES,
        jit_merge_point_args=JIT_MERGE_POINT_ARGS,
        null_names=("pycode",), static_name="opcode",
        split_names=LATE_STATIC_ARGUMENTS, hole_names=hole_names(), name=name)


# List holder: a plain module var would fold to a translation constant.
_runtime_cogen_state = [None]


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
    """Total residual operations across all templates.

    The emitted program is these operations plus the pipeline's own markers,
    so this is the part a template-level simplification can move.
    """
    total = 0
    for key in extension.templates:
        total += len(extension.templates[key].operations)
    line = "[pe] template operations: %d over %d templates" % (
        total, len(extension.templates))
    if out is not None:
        print >> out, line
    return line


def report_unresolvable(extension, out=None):
    """Templates whose targets the runtime resolver cannot evaluate.

    The same call the cogen callback makes, with dummy bindings: a late-static
    operation with no dispatch shows up here, at translation time, instead of
    as a decline inside the translated binary.
    """
    from pypy.tool.stdlib_opcode import opcode_method_names

    from rpython.rlib.rarithmetic import r_uint

    lines = []
    # The pc must have the type interp_step declares, or a target mixing it
    # with a signed oparg fails here for a reason the runtime would not have.
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


def install_runtime_cogen(codewriter, jitdriver_sd, translator):
    """Translation-time entry point: wire runtime cogen onto the portal."""
    from pypy.interpreter.pycode import CO_GENERATOR, PyCode
    from rpython.jit.codewriter.jitcode import (
        PEJitCodeMetadata, register_late_jitcode)
    from rpython.rtyper.annlowlevel import cast_gcref_to_instance
    from rpython.translator.backendopt.jitcode_emitter import ProgramEmitter
    from rpython.translator.backendopt.runtime_cogen import (
        generate_for_live_code)

    extension = build_generating_extension(translator)
    linker = portal_linker(jitdriver_sd, "linked-pypy-runtime-cogen")
    guard = (GREEN_PC_INDEX, GREEN_CODE_INDEX)

    # Runtime boundary: fragments compiled here; the callback below never runs
    # the codewriter, which is what lets it run inside a translated binary.
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
            extension.unsupported[key] = error
            del extension.templates[key]
    report_unsupported(extension, sys.stdout)
    native_table = emitter.native_table()
    _runtime_cogen_state[0] = (codewriter, native_table)

    def runtime_cogen(gcref):
        from rpython.rlib.debug import debug_print, have_debug_prints
        code = cast_gcref_to_instance(PyCode, gcref)
        if code is None:
            return None
        if code.co_flags & CO_GENERATOR:
            # A generator's frame suspends at YIELD_VALUE and is resumed
            # later at that pc; a residual program is entered at a block
            # boundary and runs to one of its own exits, which is not the
            # same contract.
            return None
        if have_debug_prints():
            # Name the code object: a residual program that misbehaves is
            # otherwise only identifiable by a raw address.
            debug_print("pe-cogen code %s %s:%d" % (
                code.co_name, code.co_filename, code.co_firstlineno))
        program = generate_for_live_code(
            extension, linker, codewriter, code, guard, gcref,
            entry_pc=0, native_table=native_table)
        if program is None:
            return None
        # Assembled after finish_setup() froze liveness and jitcode tables.
        register_late_jitcode(program.jitcode,
                              program.jitcode.own_liveness_info)
        return program

    mainjitcode = linker.mainjitcode(codewriter)
    metadata = PEJitCodeMetadata(0, [], [], [], [], [], [])
    metadata.guard_ref_index = GREEN_CODE_INDEX
    metadata.runtime_cogen = runtime_cogen
    metadata.cogen_threshold = 32
    metadata.threshold_env_var = "PYPY_COGEN_THRESHOLD"
    mainjitcode.pe_metadata = metadata
    return None
