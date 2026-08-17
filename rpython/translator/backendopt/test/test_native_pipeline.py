"""Equivalence gate: native pipeline must byte-match the legacy one."""

import py

from rpython.translator.backendopt.jitcode_emitter import (
    HOLE_SENTINEL, ProgramEmitter, TemplateFragment)
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


def test_stamp_descr_indices_covers_fragment_only_descrs_before_any_assemble():
    """Regression: stamp descr indices before any jitcode is assembled."""
    from rpython.jit.codewriter.codewriter import CodeWriter
    from rpython.jit.codewriter.test.test_codewriter import (
        FakeCPU, FakeJitDriverSD)
    from rpython.translator.backendopt.jitcode_emitter import (
        stamp_descr_indices)

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

    assert codewriter.assembler.descrs == []

    native_table = build_native_table(emitter._fragments)
    stamp_descr_indices(codewriter, native_table)

    assert len(codewriter.assembler.descrs) == 1
    assert codewriter.assembler.descrs[0].pe_descr_index == 0

    from rpython.translator.backendopt.native_fragments import NDescr
    found = 0
    for pair in native_table.values():
        for fragment in pair:
            if fragment is None:
                continue
            for insn in fragment.insns:
                for operand in insn.operands:
                    if isinstance(operand, NDescr):
                        assert operand.descr.pe_descr_index >= 0
                        found += 1
    assert found > 0   # the calldescr really was reached by this walk


def test_readonly_native_assembler_declines_uncovered_insn():
    """readonly=True NativeAssembler (the runtime_cogen path) never grows
    the shared insns/descrs tables: an uncovered (opname, argcodes)
    combination declines the whole program instead of minting a new
    opcode number (see NativeAssembler._insn_number, native_pipeline.py).
    """
    from rpython.jit.codewriter.assembler import AssemblerError

    code = chr(OP_DEC_JUMP) + chr(0) + chr(OP_HALT) + chr(0)
    program, emitter = _toy_setup(code)
    native_table = build_native_table(emitter._fragments)

    # No share_with: an empty, private insns table -- nothing was ever
    # precompiled into it -- so the very first instruction is already
    # uncovered.
    assembler = NativeAssembler(readonly=True)
    py.test.raises(AssemblerError, emit_and_assemble_native,
                   native_table, program, "native-readonly",
                   has_merge_points=False, assembler=assembler)


def test_repeated_helper_call_constants_dedup():
    """Regression: repeated helper-call constant must dedup, not balloon."""
    from rpython.jit.codewriter.codewriter import CodeWriter
    from rpython.jit.codewriter.test.test_codewriter import (
        FakeCPU, FakeJitDriverSD)

    OP_ADD = 2
    N = 60

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

    code = (chr(OP_ADD) + chr(1)) * N + chr(OP_HALT) + chr(0)
    graph, translator = get_graph(step, [int, int, int, int])
    extension = GeneratingExtension.from_step_function(
        translator, step, [OP_ADD, OP_HALT], byte_pair_decoder)
    program = extension.generate(code)
    codewriter = CodeWriter(FakeCPU(translator.rtyper), [FakeJitDriverSD(graph)])
    codewriter.find_all_graphs(NoInlinePolicy())
    emitter = ProgramEmitter(codewriter, None, "opcode", ("pc",),
                             ("pc", "oparg", "code"), ("value",))
    emitter.precompile_fragments(extension.templates)

    original_jitcode, original_positions = emitter.emit(program, "orig-add-N")

    native_table = build_native_table(emitter._fragments)
    native_jitcode, native_positions, _asm = emit_and_assemble_native(
        native_table, program, "native-add-N", has_merge_points=False)

    assert len(original_jitcode.constants_i) == 2   # helper + oparg(1)
    assert len(native_jitcode.constants_i) == 2

    # And, as everywhere else in this file, byte-identical besides.
    assert native_jitcode.code == original_jitcode.code
    assert native_jitcode.constants_i == original_jitcode.constants_i
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


class _FakeSwitchTemplate(object):
    def resolve_targets(self, bindings):
        return []


class _FakeSwitchBlock(object):
    key = "switchop"
    bindings = {}
    template = _FakeSwitchTemplate()


class _FakeSwitchProgram(object):
    entry_pc = 0
    blocks = {0: _FakeSwitchBlock()}


def test_switch_byte_identical():
    """SwitchDictDescr coverage: hand-built, since none occurs naturally."""
    from rpython.jit.codewriter.codewriter import CodeWriter
    from rpython.jit.codewriter.flatten import Register, Label, TLabel
    from rpython.jit.codewriter.jitcode import SwitchDictDescr

    switchdict = SwitchDictDescr()
    switchdict._labels = [(1, TLabel("case1")), (2, TLabel("case2"))]
    insns = [
        ("switch", Register("int", 0), switchdict),
        ("int_return", Register("int", 1)),
        (Label("case1"),),
        ("int_return", Register("int", 1)),
        (Label("case2"),),
        ("int_return", Register("int", 1)),
    ]
    fragment = TemplateFragment(insns, [], {"int": 2}, {})

    codewriter = CodeWriter()
    emitter = ProgramEmitter(codewriter, None, "opcode", (), (), ())
    emitter._fragments[("switchop", False)] = fragment

    program = _FakeSwitchProgram()
    original_jitcode, original_positions = emitter.emit(program, "orig-switch")

    native_table = build_native_table(emitter._fragments)
    native_jitcode, native_positions, _asm = emit_and_assemble_native(
        native_table, program, "native-switch", has_merge_points=False)

    assert native_jitcode.code == original_jitcode.code
    assert native_jitcode.constants_i == original_jitcode.constants_i
    assert native_jitcode.constants_r == original_jitcode.constants_r
    assert native_jitcode.num_regs_i() == original_jitcode.num_regs_i()
    assert native_positions == original_positions
    # Both paths really resolve the switch's targets, not just produce
    # matching bytes by coincidence: fix_labels' SwitchDictDescr.attach()
    # fills a real dict on the *clone* each side builds (never on the
    # shared ``switchdict`` template -- see NSwitchDictOperand). The
    # native side resolves via its own native_switchdictdescrs/
    # NativeSwitchDictDescr/fix_labels override instead.
    [orig_descr] = emitter.codewriter.assembler.switchdictdescrs
    [native_descr] = _asm.native_switchdictdescrs
    assert set(orig_descr.dict) == set(native_descr.dict) == set([1, 2])
    assert orig_descr.dict == native_descr.dict
    assert not hasattr(switchdict, "dict")
