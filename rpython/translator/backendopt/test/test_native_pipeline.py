"""Equivalence gate: native pipeline must byte-match the legacy one."""

from rpython.translator.backendopt.jitcode_emitter import (
    HOLE_SENTINEL, ProgramEmitter)
from rpython.translator.backendopt.generating_extension import (
    GeneratingExtension)
from rpython.translator.backendopt.native_fragments import build_native_table
from rpython.translator.backendopt.native_pipeline import (
    emit_and_assemble_native, NativeAssembler)
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


def _toy_setup(code):
    from rpython.jit.codewriter.codewriter import CodeWriter
    from rpython.jit.codewriter.test.test_codewriter import FakeCPU

    graph, translator = get_graph(interpret_one, [int, int, int, int])
    extension = GeneratingExtension.from_step_function(
        translator, interpret_one, [OP_DEC_JUMP, OP_HALT], byte_pair_decoder)
    program = extension.generate(code)
    codewriter = CodeWriter(FakeCPU(translator.rtyper), [])
    emitter = ProgramEmitter(codewriter, None, "opcode", ("pc",),
                             ("pc", "oparg", "code"), ("value",))
    emitter.precompile_fragments(extension.templates)
    return program, emitter


def test_toy_program_byte_identical():
    code = (chr(OP_DEC_JUMP) + chr(0) + chr(OP_DEC_JUMP) + chr(0) +
            chr(OP_HALT) + chr(0))
    program, emitter = _toy_setup(code)

    original_jitcode, original_positions = emitter.emit(program, "orig")

    native_table = build_native_table(emitter._fragments)
    native_jitcode, native_positions, _asm = emit_and_assemble_native(
        native_table, program, "native", has_merge_points=False)

    assert native_jitcode.code == original_jitcode.code
    assert native_jitcode.constants_i == original_jitcode.constants_i
    assert native_jitcode.constants_r == original_jitcode.constants_r
    assert native_jitcode.constants_f == original_jitcode.constants_f
    assert native_jitcode.num_regs_i() == original_jitcode.num_regs_i()
    assert native_jitcode.num_regs_r() == original_jitcode.num_regs_r()
    assert native_jitcode.num_regs_f() == original_jitcode.num_regs_f()
    assert native_positions == original_positions
    assert str(HOLE_SENTINEL) not in original_jitcode.dump()


def test_toy_program_with_shared_calldescr_byte_identical():
    """Covers an AbstractDescr (calldescr) operand, not just registers/ints."""
    from rpython.jit.codewriter.codewriter import CodeWriter
    from rpython.jit.codewriter.test.test_codewriter import (
        FakeCPU, FakeJitDriverSD)

    OP_ADD = 2

    def helper(x, y):
        return x + y

    def step(opcode, oparg, pc, value):
        if opcode == OP_ADD:
            return pc + 2, helper(value, oparg)
        return -1, value

    step._pe_static_args_ = ("opcode",)
    step._pe_split_args_ = ("pc",)

    class NoInlinePolicy(object):
        def look_inside_graph(self, graph):
            return False

    code = chr(OP_ADD) + chr(5) + chr(OP_ADD) + chr(9) + chr(OP_HALT) + chr(0)
    graph, translator = get_graph(step, [int, int, int, int])
    extension = GeneratingExtension.from_step_function(
        translator, step, [OP_ADD, OP_HALT], byte_pair_decoder)
    program = extension.generate(code)
    codewriter = CodeWriter(FakeCPU(translator.rtyper), [FakeJitDriverSD(graph)])
    codewriter.find_all_graphs(NoInlinePolicy())
    emitter = ProgramEmitter(codewriter, None, "opcode", ("pc",),
                             ("pc", "oparg", "code"), ("value",))
    emitter.precompile_fragments(extension.templates)

    original_jitcode, original_positions = emitter.emit(program, "orig-add")

    native_table = build_native_table(emitter._fragments)
    native_jitcode, native_positions, _asm = emit_and_assemble_native(
        native_table, program, "native-add", has_merge_points=False)

    assert native_jitcode.code == original_jitcode.code
    assert native_jitcode.constants_i == original_jitcode.constants_i
    assert native_jitcode.constants_r == original_jitcode.constants_r
    assert native_positions == original_positions


def test_tla_countdown_byte_identical():
    """Compares both pipelines via a real WarmRunnerDesc/portal setup."""
    from rpython.jit.metainterp.test.support import LLJitMixin
    from rpython.jit.tl.tla import tla
    from rpython.jit.tl.tla import offline as tla_offline
    from rpython.rlib.nonconst import NonConstant

    COUNTDOWN = [
        tla.CONST_INT, 1,
        tla.SUB,
        tla.DUP,
        tla.JUMP_IF, 0,
        tla.RETURN,
    ]
    bytecode = ''.join([chr(x) for x in COUNTDOWN])
    RUNTIME_NAMES = ("self", "bytecode")
    JIT_MERGE_POINT_ARGS = ("pc", "bytecode", "self")

    def interp_w(intvalue):
        w_result = tla.run(NonConstant(bytecode), tla.W_IntObject(intvalue))
        assert isinstance(w_result, tla.W_IntObject)
        return w_result.intvalue

    captured = {}

    def install(codewriter, jitdriver_sd, translator):
        from rpython.jit.codewriter.assembler import Assembler

        extension = tla_offline.build_generating_extension(translator)
        program = extension.generate(bytecode)

        used = dict((key, extension.templates[key]) for key in
                   (tla.CONST_INT, tla.SUB, tla.DUP, tla.JUMP_IF, tla.RETURN))
        emitter = ProgramEmitter(
            codewriter, jitdriver_sd, "opcode", ("pc",), ("oparg",),
            RUNTIME_NAMES, jit_merge_point_args=JIT_MERGE_POINT_ARGS)
        emitter.precompile_fragments(used)

        # codewriter.assembler is pre-populated; swap in a fresh one here.
        emitter.codewriter.assembler = Assembler()
        original_jitcode, original_positions = emitter.emit(program, "orig-tla")

        native_table = build_native_table(emitter._fragments)
        native_jitcode, native_positions, _asm = emit_and_assemble_native(
            native_table, program, "native-tla", has_merge_points=True)

        captured["original"] = original_jitcode
        captured["original_positions"] = original_positions
        captured["native"] = native_jitcode
        captured["native_positions"] = native_positions
        return None

    LLJitMixin().meta_interp(
        interp_w, [42], listops=True, pe_linked_setup=install,
        graph_and_interp_only=True)

    original_jitcode = captured["original"]
    native_jitcode = captured["native"]
    assert native_jitcode.code == original_jitcode.code
    assert native_jitcode.constants_i == original_jitcode.constants_i
    assert native_jitcode.constants_r == original_jitcode.constants_r
    assert native_jitcode.constants_f == original_jitcode.constants_f
    assert native_jitcode.num_regs_i() == original_jitcode.num_regs_i()
    assert native_jitcode.num_regs_r() == original_jitcode.num_regs_r()
    assert native_jitcode.num_regs_f() == original_jitcode.num_regs_f()
    assert captured["native_positions"] == captured["original_positions"]
