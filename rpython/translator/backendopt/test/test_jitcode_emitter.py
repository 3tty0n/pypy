from rpython.translator.translator import TranslationContext, graphof
from rpython.translator.backendopt.generating_extension import (
    GeneratingExtension)
from rpython.translator.backendopt.jitcode_emitter import (
    HOLE_SENTINEL, ProgramEmitter)
from rpython.translator.backendopt.test.test_partialeval_template_lowering \
    import byte_pair_decoder, get_graph


OP_DEC_JUMP = 0
OP_HALT = 1


def interpret_one(opcode, oparg, pc, value):
    if opcode == OP_DEC_JUMP:
        if value > 0:
            return oparg, value - 1
        return pc + 2, value
    return -1, value

interpret_one._pe_static_args_ = ("opcode",)
interpret_one._pe_split_args_ = ("pc",)


def emit(code):
    from rpython.jit.codewriter.codewriter import CodeWriter
    from rpython.jit.codewriter.test.test_codewriter import FakeCPU

    graph, translator = get_graph(interpret_one, [int, int, int, int])
    extension = GeneratingExtension.from_step_function(
        translator, interpret_one, [OP_DEC_JUMP, OP_HALT], byte_pair_decoder)
    program = extension.generate(code)
    codewriter = CodeWriter(FakeCPU(translator.rtyper), [])
    emitter = ProgramEmitter(codewriter, None, "opcode", ("pc",),
                             ("pc", "oparg", "code"), ("value",))
    jitcode, entry_positions = emitter.emit(program, "emitted-mini")
    return program, emitter, jitcode, entry_positions


def test_concatenated_fragments_assemble_into_one_jitcode():
    code = chr(OP_DEC_JUMP) + chr(0) + chr(OP_HALT) + chr(0)
    program, emitter, jitcode, entry_positions = emit(code)

    assert set(entry_positions) == set(program.blocks)
    # The entry block leads, so no prologue goto sits on the entry position --
    # the metainterp reads a jump there as a back edge.
    assert entry_positions[program.entry_pc] == 0

    dump = jitcode.dump()
    assert "int_gt" in dump
    assert "int_sub" in dump
    assert "goto" in dump
    # Decoding is gone: no byte of the program is read at run time.
    assert "strgetitem" not in dump
    assert "int_eq" not in dump
    # Every hole this program supplies was filled, not left at its sentinel.
    assert str(HOLE_SENTINEL) not in dump


def test_each_template_is_assembled_once_per_specialization():
    code = (chr(OP_DEC_JUMP) + chr(0) + chr(OP_DEC_JUMP) + chr(0) +
            chr(OP_HALT) + chr(0))
    program, emitter, jitcode, entry_positions = emit(code)
    # One fragment per reachable specialization point, and no more: the
    # codewriter ran that many times rather than once over a whole program.
    assert len(emitter._fragments) == len(program.blocks)


def test_emitting_for_a_portal_requires_merge_point_arguments():
    """The invariant the PySOM path depends on.

    Without a jit_merge_point the metainterp has to recognise the loop's back
    edge by position, and read a -live- record immediately before it -- which
    cannot be placed, because the assembler hoists a label above a -live- that
    follows it.  Better to say so than to emit a program that installs and then
    crashes on its first guard.
    """
    import py

    code = chr(OP_DEC_JUMP) + chr(0) + chr(OP_HALT) + chr(0)
    graph, translator = get_graph(interpret_one, [int, int, int, int])
    extension = GeneratingExtension.from_step_function(
        translator, interpret_one, [OP_DEC_JUMP, OP_HALT], byte_pair_decoder)
    program = extension.generate(code)

    from rpython.jit.codewriter.codewriter import CodeWriter
    from rpython.jit.codewriter.test.test_codewriter import FakeCPU

    class FakePortal(object):
        pass

    codewriter = CodeWriter(FakeCPU(translator.rtyper), [])
    emitter = ProgramEmitter(codewriter, FakePortal(), "opcode", ("pc",),
                             ("pc", "oparg", "code"), ("value",))
    error = py.test.raises(ValueError, emitter.emit, program)
    assert "jit_merge_point_args" in str(error.value)
