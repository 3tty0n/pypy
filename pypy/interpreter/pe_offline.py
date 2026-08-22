"""Partial-evaluation support for the CPython bytecode interpreter.

Mirrors rpython/jit/tl/tla/offline.py.  ``PyFrame.interp_step`` declares
``opcode`` as offline-static and ``next_instr`` as late-static, so the partial
evaluator can build one residual template per bytecode at translation time.

Unlike PySOM, there is no translation-time variant: a ``PyCode`` is built at
run time by ``compile``, ``exec`` and import, so the set of code objects to
specialize cannot be enumerated when the binary is built.
"""

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
        "next_instr": next_instr,
        "oparg": oparg,
    }
    return opcode, bindings


def build_generating_extension(translator):
    """Specialize interp_step once per opcode.

    An opcode with no template is not fatal: only code objects that actually
    reach one are left to the generic dispatch loop.
    """
    from pypy.interpreter.pyframe import PyFrame

    return GeneratingExtension.from_step_function(
        translator, PyFrame.interp_step.im_func, opcode_keys(),
        decode_instruction)


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
