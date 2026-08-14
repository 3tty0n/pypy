"""Offline partial-evaluation support for TLA bytecode."""

from rpython.jit.tl.tla import tla
from rpython.translator.backendopt.generating_extension import (
    GeneratingExtension)
from rpython.translator.backendopt.partialeval_template import (
    Finish, ResidualTemplate)


def decode_instruction(bytecode, pc):
    opcode = ord(bytecode[pc])
    next_pc = pc + 1
    if tla.HASARG[opcode]:
        oparg = ord(bytecode[next_pc])
        next_pc += 1
    else:
        oparg = 0
    bindings = {
        # interp_step receives pc after the complete instruction was decoded.
        "pc": next_pc,
        "oparg": oparg,
        "code": bytecode,
    }
    return opcode, bindings


def build_generating_extension(translator):
    extension = GeneratingExtension.from_step_function(
        translator, tla.Frame.interp_step.im_func,
        range(len(tla.OPNAMES)), decode_instruction)
    if extension.unsupported:
        raise ValueError(extension.report(tla.OPNAMES.__getitem__))

    # RPython keeps the syntactically following ``return pc, None`` as an
    # additional residual exit on some graph iteration orders.  RETURN is
    # unconditionally terminal in the TLA bytecode format.
    template = extension.templates[tla.RETURN]
    extension.templates[tla.RETURN] = ResidualTemplate(
        template.key, template.operations, template.holes,
        tuple(t for t in template.terminators if isinstance(t, Finish)),
        residual_graph=template.residual_graph)
    return extension


def lower_and_install(codewriter, jitdriver_sd, translator, bytecode,
                      whole_graph=False):
    """Build and install the dispatch-free TLA JitCode at translation time."""
    from rpython.translator.backendopt.portal_linker import PortalLinker

    linked = build_generating_extension(translator).generate(bytecode)
    # Portal boxes are (pc, bytecode, self); interp_step expects
    # (self, bytecode, opcode, oparg, pc).
    # The default back end needs merge points: it cannot place the -live-
    # record that position-based back edge detection wants, because the
    # assembler hoists a label above a -live- that follows it.  TLA's
    # whole-graph lowering predates them and stays as it was.
    linker = PortalLinker(jitdriver_sd, (2, 1), ("self", "bytecode"),
                          jit_merge_point_args=()
                          if whole_graph else ("pc", "bytecode", "self"),
                          null_names=("bytecode",),
                          static_name="opcode", split_names=("pc",),
                          hole_names=("oparg",),
                          name="linked-tla-wholegraph"
                          if whole_graph else "linked-tla")
    return linker.install(codewriter, linked, whole_graph=whole_graph)
