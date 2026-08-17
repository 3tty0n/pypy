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


def test_each_template_is_assembled_once_per_opcode():
    code = (chr(OP_DEC_JUMP) + chr(0) + chr(OP_DEC_JUMP) + chr(0) +
            chr(OP_HALT) + chr(0))
    program, emitter, jitcode, entry_positions = emit(code)
    distinct = set(block.key for block in program.blocks.values())
    assert len(program.blocks) == 3
    assert len(distinct) == 2
    assert len(emitter._fragments) == len(distinct)


def test_fragment_identity_is_shared_across_pc():
    """The whole point: two blocks of the same opcode share one fragment."""
    code = (chr(OP_DEC_JUMP) + chr(0) + chr(OP_DEC_JUMP) + chr(0) +
            chr(OP_HALT) + chr(0))
    program, emitter, jitcode, entry_positions = emit(code)
    block_a = program.blocks[0]
    block_b = program.blocks[2]
    assert block_a.key == block_b.key == OP_DEC_JUMP
    assert block_a.bindings != block_b.bindings
    assert emitter.fragment_for(block_a) is emitter.fragment_for(block_b)


def test_shared_fragment_patches_each_instance_with_its_own_values():
    """Sharing patches each instance's own pc/oparg, not the sentinel."""
    from rpython.flowspace.model import Constant

    OP_X = 3
    OP_HALT = 1

    def step(opcode, oparg, pc, value):
        if opcode == OP_X:
            checksum = pc + oparg
            return pc + 2, value + checksum
        return -1, value

    step._pe_static_args_ = ("opcode",)
    step._pe_split_args_ = ("pc",)

    from rpython.jit.codewriter.codewriter import CodeWriter
    from rpython.jit.codewriter.test.test_codewriter import FakeCPU

    code = chr(OP_X) + chr(7) + chr(OP_X) + chr(11) + chr(OP_HALT) + chr(0)
    graph, translator = get_graph(step, [int, int, int, int])
    extension = GeneratingExtension.from_step_function(
        translator, step, [OP_X, OP_HALT], byte_pair_decoder)
    program = extension.generate(code)

    codewriter = CodeWriter(FakeCPU(translator.rtyper), [])
    emitter = ProgramEmitter(codewriter, None, "opcode", ("pc",),
                             ("pc", "oparg", "code"), ("value",))
    jitcode, entry_positions = emitter.emit(program, "emitted-x")

    assert emitter.fragment_for(program.blocks[0]) is \
        emitter.fragment_for(program.blocks[2])
    assert str(HOLE_SENTINEL) not in jitcode.dump()

    sums = []
    for insn in emitter.last_ssarepr.insns:
        if insn[0] == "int_add" and isinstance(insn[1], Constant) and \
                isinstance(insn[2], Constant):
            sums.append((insn[1].value, insn[2].value))
    assert (0, 7) in sums
    assert (2, 11) in sums


def test_precompile_fragments_needs_every_decoder_binding_named():
    """Regression: an undeclared decoder binding needs state_names."""
    from rpython.flowspace.model import Constant

    OP_X = 3
    OP_HALT = 1

    def step(opcode, oparg, pc, value):
        if opcode == OP_X:
            checksum = pc + oparg
            return pc + 2, value + checksum
        return -1, value

    step._pe_static_args_ = ("opcode",)
    step._pe_split_args_ = ("pc",)

    from rpython.jit.codewriter.codewriter import CodeWriter
    from rpython.jit.codewriter.test.test_codewriter import FakeCPU

    code = chr(OP_X) + chr(7) + chr(OP_X) + chr(11) + chr(OP_HALT) + chr(0)
    graph, translator = get_graph(step, [int, int, int, int])
    extension = GeneratingExtension.from_step_function(
        translator, step, [OP_X, OP_HALT], byte_pair_decoder)
    program = extension.generate(code)

    codewriter = CodeWriter(FakeCPU(translator.rtyper), [])
    emitter = ProgramEmitter(codewriter, None, "opcode", ("pc",),
                             ("pc", "code"), ("value",))
    emitter.precompile_fragments(extension.templates, state_names=("oparg",))
    jitcode, entry_positions = emitter.emit(program, "emitted-x-precompiled")

    assert emitter.fragment_for(program.blocks[0]) is \
        emitter.fragment_for(program.blocks[2])
    assert str(HOLE_SENTINEL) not in jitcode.dump()

    sums = []
    for insn in emitter.last_ssarepr.insns:
        if insn[0] == "int_add" and isinstance(insn[1], Constant) and \
                isinstance(insn[2], Constant):
            sums.append((insn[1].value, insn[2].value))
    assert (0, 7) in sums
    assert (2, 11) in sums


def test_shared_fragment_survives_a_foldable_op_on_a_hole():
    """A hole reaching a foldable op (chr) must not be eagerly folded."""
    from rpython.flowspace.model import Constant

    OP_CHR = 4
    OP_HALT = 1

    def step(opcode, oparg, pc, value):
        if opcode == OP_CHR:
            c = chr(oparg)
            return pc + 2, value + ord(c)
        return -1, value

    step._pe_static_args_ = ("opcode",)
    step._pe_split_args_ = ("pc",)

    from rpython.jit.codewriter.codewriter import CodeWriter
    from rpython.jit.codewriter.test.test_codewriter import FakeCPU

    code = (chr(OP_CHR) + chr(65) + chr(OP_CHR) + chr(90) +
            chr(OP_HALT) + chr(0))
    graph, translator = get_graph(step, [int, int, int, int])
    extension = GeneratingExtension.from_step_function(
        translator, step, [OP_CHR, OP_HALT], byte_pair_decoder)
    program = extension.generate(code)

    codewriter = CodeWriter(FakeCPU(translator.rtyper), [])
    emitter = ProgramEmitter(codewriter, None, "opcode", ("pc",),
                             ("pc", "oparg", "code"), ("value",))
    jitcode, entry_positions = emitter.emit(program, "emitted-chr")

    assert emitter.fragment_for(program.blocks[0]) is \
        emitter.fragment_for(program.blocks[2])
    assert str(HOLE_SENTINEL) not in jitcode.dump()

    added = [insn[2].value for insn in emitter.last_ssarepr.insns
             if insn[0] == "int_add" and isinstance(insn[2], Constant)
             and insn[2].value in (65, 90)]
    assert 65 in added
    assert 90 in added


def test_shared_fragment_reuses_calldescr_objects():
    """A calldescr baked into a shared fragment is reused, not rebuilt."""
    from rpython.jit.codewriter.codewriter import CodeWriter
    from rpython.jit.codewriter.test.test_codewriter import FakeCPU, \
        FakeJitDriverSD
    from rpython.jit.metainterp.history import AbstractDescr

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
            # Residual call keeps the calldescr; inlining drops it.
            return False

    code = chr(OP_ADD) + chr(5) + chr(OP_ADD) + chr(9) + chr(OP_HALT) + chr(0)
    graph, translator = get_graph(step, [int, int, int, int])
    extension = GeneratingExtension.from_step_function(
        translator, step, [OP_ADD, OP_HALT], byte_pair_decoder)
    program = extension.generate(code)
    codewriter = CodeWriter(FakeCPU(translator.rtyper),
                            [FakeJitDriverSD(graph)])
    codewriter.find_all_graphs(NoInlinePolicy())
    emitter = ProgramEmitter(codewriter, None, "opcode", ("pc",),
                             ("pc", "oparg", "code"), ("value",))
    emitter.emit(program, "emitted-add")

    descrs = [item for insn in emitter.last_ssarepr.insns for item in insn
             if isinstance(item, AbstractDescr)]
    assert len(descrs) == 2
    assert descrs[0] is descrs[1]


def test_precompile_fragments_runs_the_codewriter_zero_additional_times():
    """Eager fragment table: fragments built once; emit does no compiling."""
    from rpython.jit.codewriter.codewriter import CodeWriter
    from rpython.jit.codewriter.test.test_codewriter import FakeCPU

    code = (chr(OP_DEC_JUMP) + chr(0) + chr(OP_DEC_JUMP) + chr(0) +
            chr(OP_HALT) + chr(0))
    graph, translator = get_graph(interpret_one, [int, int, int, int])
    extension = GeneratingExtension.from_step_function(
        translator, interpret_one, [OP_DEC_JUMP, OP_HALT], byte_pair_decoder)
    codewriter = CodeWriter(FakeCPU(translator.rtyper), [])
    emitter = ProgramEmitter(codewriter, None, "opcode", ("pc",),
                             ("pc", "oparg", "code"), ("value",))

    emitter.precompile_fragments(extension.templates)
    assert len(emitter._fragments) == 2

    calls = []
    real_compile = emitter.compiler.compile
    emitter.compiler.compile = lambda *a, **kw: (calls.append(1) or
                                                 real_compile(*a, **kw))
    try:
        program = extension.generate(code)
        jitcode, entry_positions = emitter.emit(program, "emitted-precompiled")
    finally:
        emitter.compiler.compile = real_compile

    assert calls == []
    assert set(entry_positions) == set(program.blocks)
    assert str(HOLE_SENTINEL) not in jitcode.dump()


def test_precompile_fragments_skips_untemplated_opcodes():
    """An untemplated opcode is simply absent from the fragment table."""
    from rpython.jit.codewriter.codewriter import CodeWriter
    from rpython.jit.codewriter.test.test_codewriter import FakeCPU

    graph, translator = get_graph(interpret_one, [int, int, int, int])
    extension = GeneratingExtension.from_step_function(
        translator, interpret_one, [OP_DEC_JUMP], byte_pair_decoder)
    assert extension.unsupported == {}
    assert OP_HALT not in extension.templates

    codewriter = CodeWriter(FakeCPU(translator.rtyper), [])
    emitter = ProgramEmitter(codewriter, None, "opcode", ("pc",),
                             ("pc", "oparg", "code"), ("value",))
    emitter.precompile_fragments(extension.templates)
    assert set(emitter._fragments) == set([(OP_DEC_JUMP, False)])


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


def test_the_interpreter_decides_what_may_be_specialized():
    """pe.dont_specialize is the interpreter's veto, not the evaluator's."""
    from rpython.rlib import pe

    def step(opcode, oparg, pc, value):
        if opcode == OP_DEC_JUMP:
            if value > 0:
                return oparg, value - 1
            return pc + 2, value
        return -1, value

    pe.PEDriver(static="opcode", split="pc", never=(OP_HALT,)).bind(step)

    _graph, translator = get_graph(step, [int, int, int, int])
    extension = GeneratingExtension.from_step_function(
        translator, step, [OP_DEC_JUMP, OP_HALT], byte_pair_decoder)

    assert extension.handles(OP_DEC_JUMP)
    assert not extension.handles(OP_HALT)
    # A program that reaches the vetoed instruction is declined whole.
    assert extension.generate(chr(OP_DEC_JUMP) + chr(0) +
                              chr(OP_HALT) + chr(0)) is None


def test_the_interpreter_decides_what_is_worth_linking():
    """The declared policy is picked up from the step function itself."""
    from rpython.rlib import pe

    seen = []

    def only_with_loops(program, code):
        seen.append(len(program.loop_headers))
        assert len(code) > 0
        return len(program.loop_headers) > 0

    def step(opcode, oparg, pc, value):
        if opcode == OP_DEC_JUMP:
            if value > 0:
                return oparg, value - 1
            return pc + 2, value
        return -1, value

    pe.PEDriver(static="opcode", split="pc",
                worth_generating=only_with_loops).bind(step)
    _graph, translator = get_graph(step, [int, int, int, int])
    extension = GeneratingExtension.from_step_function(
        translator, step, [OP_DEC_JUMP, OP_HALT], byte_pair_decoder)

    # Jumping to 0 makes a loop; jumping past the end does not.
    looping = extension.generate(chr(OP_DEC_JUMP) + chr(0) +
                                 chr(OP_HALT) + chr(0))
    straight = extension.generate(chr(OP_HALT) + chr(0))
    assert looping is not None
    assert straight is None
    assert seen == [1, 0]


def test_the_merge_point_binds_the_driver_to_its_function():
    """Declaration reaches the evaluator from the call site, like JitDriver."""
    from rpython.rlib.pe import PEDriver

    driver = PEDriver(static="opcode", split="pc", min_size=2)

    def step(opcode, oparg, pc, value):
        driver.pe_merge_point(opcode=opcode, oparg=oparg, pc=pc, value=value)
        if opcode == OP_DEC_JUMP:
            return oparg, value - 1
        return -1, value

    graph, _translator = get_graph(step, [int, int, int, int])
    assert graph.func._pe_static_args_ == ("opcode",)
    assert graph.func._pe_split_args_ == ("pc",)
    assert graph.func._pe_link_policy_ is not None
    # and it leaves nothing behind for the evaluator to strip
    assert not [op for block in graph.iterblocks()
                for op in block.operations if "pe_merge" in op.opname]
