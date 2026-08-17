"""Offline partial-evaluation support for tla_refcache -- mirrors
rpython/jit/tl/tla/offline.py exactly, retargeted at tla_refcache.Frame.
"""

from rpython.jit.tl.tla import tla_refcache as refcache
from rpython.translator.backendopt.generating_extension import (
    GeneratingExtension)
from rpython.translator.backendopt.partialeval_template import (
    Finish, ResidualTemplate)


def decode_instruction(bytecode, pc):
    opcode = ord(bytecode[pc])
    next_pc = pc + 1
    if refcache.HASARG[opcode]:
        oparg = ord(bytecode[next_pc])
        next_pc += 1
    else:
        oparg = 0
    bindings = {
        "pc": next_pc,
        "oparg": oparg,
    }
    return opcode, bindings


def build_generating_extension(translator):
    extension = GeneratingExtension.from_step_function(
        translator, refcache.Frame.interp_step.im_func,
        range(len(refcache.OPNAMES)), decode_instruction)
    if extension.unsupported:
        raise ValueError(extension.report(refcache.OPNAMES.__getitem__))

    # See tla/offline.py's identical fix: RETURN is unconditionally
    # terminal in this bytecode format, but RPython's graph iteration can
    # keep the syntactically-following "return pc, None, ..." as an extra
    # residual exit.
    template = extension.templates[refcache.RETURN]
    extension.templates[refcache.RETURN] = ResidualTemplate(
        template.key, template.operations, template.holes,
        tuple(t for t in template.terminators if isinstance(t, Finish)),
        residual_graph=template.residual_graph)
    return extension
