"""Equivalence gate: native pipeline must byte-match the legacy one."""

import py

from rpython.translator.backendopt.jitcode_emitter import (
    HOLE_SENTINEL, ProgramEmitter, TemplateFragment)
from rpython.translator.backendopt.generating_extension import (
    GeneratingExtension)
from rpython.translator.backendopt.native_fragments import build_native_table
from rpython.translator.backendopt.native_pipeline import (
    emit_and_assemble_native, NativeAssembler)


def emit_and_assemble_native_unoptimised(*args, **kwds):
    """The gate checks lowering only; folding has its own test."""
    kwds["optimise"] = False
    return emit_and_assemble_native(*args, **kwds)
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
    native_jitcode, native_positions, _asm = \
        emit_and_assemble_native_unoptimised(
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
    codewriter = CodeWriter(
        FakeCPU(translator.rtyper), [FakeJitDriverSD(graph)])
    codewriter.find_all_graphs(NoInlinePolicy())
    emitter = ProgramEmitter(codewriter, None, "opcode", ("pc",),
                             ("pc", "oparg", "code"), ("value",))
    emitter.precompile_fragments(extension.templates)

    original_jitcode, original_positions = emitter.emit(program, "orig-add")

    native_table = build_native_table(emitter._fragments)
    native_jitcode, native_positions, _asm = \
        emit_and_assemble_native_unoptimised(
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
        stamp_descr_indices, register_native_insn_coverage)

    OP_ADD = 2
    OP_SUB = 3

    def helper(x, y):
        return x + y

    def step(opcode, oparg, pc, value):
        if opcode == OP_ADD:
            return pc + 2, helper(value, oparg)
        if opcode == OP_SUB:
            # Both operands late-static; RPython cannot fold this away.
            return pc + 2, oparg - 5
        return -1, value

    step._pe_static_args_ = ("opcode",)
    step._pe_split_args_ = ("pc",)

    class NoInlinePolicy(object):
        def look_inside_graph(self, graph):
            return False

    code = (chr(OP_ADD) + chr(5) + chr(OP_ADD) + chr(9) +
            chr(OP_SUB) + chr(9) + chr(OP_HALT) + chr(0))
    graph, translator = get_graph(step, [int, int, int, int])
    extension = GeneratingExtension.from_step_function(
        translator, step, [OP_ADD, OP_SUB, OP_HALT], byte_pair_decoder)
    program = extension.generate(code)
    jitdriver_sd = FakeJitDriverSD(graph)
    # handle_jit_marker__pe_bailout_point needs .active/.greens/.index.
    class _FakeJitDriver(object):
        active = True
        greens = []
        numreds = 3
    jitdriver_sd.jitdriver = _FakeJitDriver()
    jitdriver_sd.index = 0
    codewriter = CodeWriter(FakeCPU(translator.rtyper), [jitdriver_sd])
    codewriter.find_all_graphs(NoInlinePolicy())
    emitter = ProgramEmitter(codewriter, jitdriver_sd, "opcode", ("pc",),
                             ("pc", "oparg", "code"), ("value",),
                             jit_merge_point_args=("oparg", "pc", "value"))
    emitter.precompile_fragments(extension.templates)

    assert codewriter.assembler.descrs == []
    assert codewriter.assembler.insns == {}

    native_table = build_native_table(emitter._fragments)
    stamp_descr_indices(codewriter, native_table)
    register_native_insn_coverage(codewriter, native_table)

    assert len(codewriter.assembler.descrs) == 2
    for d in codewriter.assembler.descrs:
        assert d.pe_descr_index >= 0

    assert "pe_bailout_point/cIRFIRF" in codewriter.assembler.insns
    assert "int_sub/cc>i" in codewriter.assembler.insns

    assembler = NativeAssembler(share_with=codewriter.assembler, readonly=True)
    native_jitcode, _positions, _asm = emit_and_assemble_native_unoptimised(
        native_table, program, "native-full", has_merge_points=True,
        assembler=assembler)
    assert len(native_jitcode.code) > 0


def test_readonly_native_assembler_declines_uncovered_insn():
    """readonly NativeAssembler declines an uncovered (opname, argcodes)."""
    from rpython.jit.codewriter.assembler import AssemblerError

    code = chr(OP_DEC_JUMP) + chr(0) + chr(OP_HALT) + chr(0)
    program, emitter = _toy_setup(code)
    native_table = build_native_table(emitter._fragments)

    assembler = NativeAssembler(readonly=True)
    py.test.raises(AssemblerError, emit_and_assemble_native,
                   native_table, program, "native-readonly",
                   has_merge_points=False, assembler=assembler)


def test_emit_native_declines_a_jitcode_too_large_for_resume_pc_encoding():
    """Regression: oversized jitcode is declined, not crash resumecode.py."""
    from rpython.jit.codewriter.assembler import AssemblerError
    from rpython.translator.backendopt import native_pipeline
    from rpython.translator.backendopt.portal_linker import PortalLinker

    code = chr(OP_DEC_JUMP) + chr(0) + chr(OP_HALT) + chr(0)
    program, emitter = _toy_setup(code)
    native_table = build_native_table(emitter._fragments)

    real_emit_and_assemble_native = native_pipeline.emit_and_assemble_native

    def _oversized(*args, **kwargs):
        jitcode, entry_positions, assembler = real_emit_and_assemble_native(
            *args, **kwargs)
        jitcode.code = "x" * 32768
        return jitcode, entry_positions, assembler

    native_pipeline.emit_and_assemble_native = _oversized
    try:
        linker = PortalLinker(None, (), (), static_name="opcode",
                              name="linked-oversized")
        py.test.raises(AssemblerError, linker._emit_native,
                       emitter.codewriter, program, native_table)
    finally:
        native_pipeline.emit_and_assemble_native = \
            real_emit_and_assemble_native


def test_readonly_native_assembler_declines_constant_capacity_overflow():
    """Regression: constant-pool cap overflow raises, no fatal assert."""
    from rpython.jit.codewriter.assembler import AssemblerError

    assembler = NativeAssembler(readonly=True)
    assembler.setup("cap-test")
    for value in range(256):
        assembler.emit_resolved_const(value, "int")
    py.test.raises(AssemblerError, assembler.emit_resolved_const, 256, "int")


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
    codewriter = CodeWriter(
        FakeCPU(translator.rtyper), [FakeJitDriverSD(graph)])
    codewriter.find_all_graphs(NoInlinePolicy())
    emitter = ProgramEmitter(codewriter, None, "opcode", ("pc",),
                             ("pc", "oparg", "code"), ("value",))
    emitter.precompile_fragments(extension.templates)

    original_jitcode, original_positions = emitter.emit(program, "orig-add-N")

    native_table = build_native_table(emitter._fragments)
    native_jitcode, native_positions, _asm = \
        emit_and_assemble_native_unoptimised(
            native_table, program, "native-add-N", has_merge_points=False)

    assert len(original_jitcode.constants_i) == 2
    assert len(native_jitcode.constants_i) == 2

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
        native_jitcode, native_positions, _asm = \
            emit_and_assemble_native_unoptimised(
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
    ref_bindings = {}
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
    native_jitcode, native_positions, _asm = \
        emit_and_assemble_native_unoptimised(
            native_table, program, "native-switch", has_merge_points=False)

    assert native_jitcode.code == original_jitcode.code
    assert native_jitcode.constants_i == original_jitcode.constants_i
    assert native_jitcode.constants_r == original_jitcode.constants_r
    assert native_jitcode.num_regs_i() == original_jitcode.num_regs_i()
    assert native_positions == original_positions
    [orig_descr] = emitter.codewriter.assembler.switchdictdescrs
    [native_descr] = _asm.native_switchdictdescrs
    assert set(orig_descr.dict) == set(native_descr.dict) == set([1, 2])
    assert orig_descr.dict == native_descr.dict
    assert not hasattr(switchdict, "dict")


def test_register_late_jitcode_twice_via_readonly_native_assembler():
    """late_jitcode .index uniqueness comes from a module-global counter."""
    from rpython.jit.codewriter.jitcode import (
        register_late_jitcode, get_late_jitcode, set_late_jitcode_base,
        _late_jitcodes_by_index)
    from rpython.translator.backendopt.jitcode_emitter import (
        stamp_descr_indices, register_native_insn_coverage)

    code = chr(OP_DEC_JUMP) + chr(0) + chr(OP_HALT) + chr(0)
    program, emitter = _toy_setup(code)
    native_table = build_native_table(emitter._fragments)
    # Must run once, after ordinary jitcodes assemble, before readonly use.
    stamp_descr_indices(emitter.codewriter, native_table)
    register_native_insn_coverage(emitter.codewriter, native_table)

    _late_jitcodes_by_index.clear()
    try:
        set_late_jitcode_base(5)

        def assemble_and_register(name):
            assembler = NativeAssembler(
                share_with=emitter.codewriter.assembler, readonly=True)
            jitcode, _, assembler = emit_and_assemble_native_unoptimised(
                native_table, program, name, has_merge_points=False,
                assembler=assembler)
            jitcode.own_liveness_info = "".join(assembler.all_liveness)
            register_late_jitcode(jitcode, jitcode.own_liveness_info)
            return jitcode

        late1 = assemble_and_register("late-1")
        late2 = assemble_and_register("late-2")

        assert late1.index != late2.index
        assert get_late_jitcode(late1.index) is late1
        assert get_late_jitcode(late2.index) is late2
    finally:
        _late_jitcodes_by_index.clear()


# RPython dict iteration order differs from CPython; order must not matter.

import itertools

from rpython.translator.backendopt.native_fragments import (
    NReg, NIntConst, NRefConst)
from rpython.translator.backendopt.native_pipeline import _emit_moves_native


class _FakeSSARepr(object):
    def __init__(self):
        self.insns = []


def _expected_final(sources, destinations):
    expected = {}
    for bname, dest in destinations.items():
        if dest is None:
            continue
        if bname in sources:
            expected[dest] = ("orig", sources[bname])
        elif dest[0] == "ref":
            expected[dest] = ("const-null",)
        else:
            expected[dest] = ("const-int", 0)
    return expected


def _simulate(sources, destinations, scratch, names):
    ssarepr = _FakeSSARepr()
    _emit_moves_native(ssarepr, sources, destinations, scratch, _names=names)

    regs = {}

    def read(reg):
        if reg not in regs:
            regs[reg] = ("orig", reg)
        return regs[reg]

    for insn in ssarepr.insns:
        [source] = insn.operands
        dest_reg = (insn.result.kind, insn.result.index)
        if isinstance(source, NReg):
            value = read((source.kind, source.index))
        elif isinstance(source, NRefConst):
            value = ("const-null",)
        elif isinstance(source, NIntConst):
            value = ("const-int", source.ivalue)
        else:
            raise AssertionError("unexpected operand %r" % (source,))
        regs[dest_reg] = value
    return regs


_NAMES = ["b0", "b1", "b2", "b3", "b4", "b5"]

_SHAPES = {
    "3-cycle + identities": (
        {"b0": ("int", 1), "b1": ("int", 2), "b2": ("int", 0),
         "b3": ("int", 3), "b4": ("int", 4), "b5": ("int", 5)},
        {"b0": ("int", 0), "b1": ("int", 1), "b2": ("int", 2),
         "b3": ("int", 3), "b4": ("int", 4), "b5": ("int", 5)},
    ),
    "3-cycle + chain": (
        {"b0": ("int", 1), "b1": ("int", 2), "b2": ("int", 0),
         "b3": ("int", 4), "b4": ("int", 5)},
        {"b0": ("int", 0), "b1": ("int", 1), "b2": ("int", 2),
         "b3": ("int", 3), "b4": ("int", 4), "b5": ("int", 5)},
    ),
    "two independent chains": (
        {"b0": ("int", 1), "b1": ("int", 2),
         "b3": ("int", 4), "b4": ("int", 5)},
        {"b0": ("int", 0), "b1": ("int", 1), "b2": ("int", 2),
         "b3": ("int", 3), "b4": ("int", 4), "b5": ("int", 5)},
    ),
    "fan-out + 2-cycle": (
        {"b0": ("int", 5), "b1": ("int", 5), "b2": ("int", 5),
         "b3": ("int", 4), "b4": ("int", 3), "b5": ("int", 0)},
        {"b0": ("int", 0), "b1": ("int", 1), "b2": ("int", 2),
         "b3": ("int", 3), "b4": ("int", 4), "b5": ("int", 5)},
    ),
    "self-moves + missing sources": (
        {"b0": ("int", 0), "b3": ("int", 3),
         "b4": ("int", 5), "b5": ("int", 4)},
        {"b0": ("int", 0), "b1": None, "b2": ("int", 2),
         "b3": ("int", 3), "b4": ("int", 4), "b5": ("int", 5)},
    ),
}


def test_emit_moves_native_order_independent():
    scratch = {"int": 6, "ref": 0}
    for shape_name, (sources, destinations) in _SHAPES.items():
        expected = _expected_final(sources, destinations)
        for order in itertools.permutations(_NAMES):
            names = [n for n in order if n in destinations]
            regs = _simulate(sources, destinations, scratch, names)
            for dest, want in expected.items():
                got = regs.get(dest, ("orig", dest))
                assert got == want, (
                    "%s: order %r produced dest %r = %r, want %r" %
                    (shape_name, names, dest, got, want))


def test_operand_argcode_options_and_patch_hole_native_handle_ref_kind():
    """Unit-level: NHole/_patch_hole_native's ref-kind branch."""
    from rpython.rtyper.lltypesystem import lltype, llmemory
    from rpython.translator.backendopt.native_fragments import NHole, NRefConst
    from rpython.translator.backendopt.native_pipeline import (
        _operand_argcode_options, _patch_hole_native)

    hole = NHole("obj", llmemory.GCREF)
    assert hole.kind == "ref"
    assert _operand_argcode_options(hole, allow_short=False) == ["r"]

    S = lltype.GcStruct("S", ("tag", lltype.Signed))
    instance = lltype.malloc(S)
    instance.tag = 42
    ref = lltype.cast_opaque_ptr(llmemory.GCREF, instance)
    patched = _patch_hole_native(hole, 0, {}, {"obj": ref}, is_marker=False)
    assert isinstance(patched, NRefConst)
    assert patched.value == ref


class RefHoleObj(object):
    """Stands in for PyCode: a code-independent hole binds to it directly."""

    def __init__(self, bytecode, tag):
        self.bytecode = bytecode
        self.tag = tag


OP_DEC_JUMP = 0
OP_HALT = 1


def toy_ref_decoder(code, pc):
    opcode = ord(code.bytecode[pc])
    oparg = ord(code.bytecode[pc + 1])
    return opcode, {"pc": pc, "oparg": oparg}


class _RefSinkHolder(object):
    last = None

_REF_SINK = _RefSinkHolder()


def interpret_one_ref(opcode, oparg, pc, value, obj):
    if opcode == OP_DEC_JUMP:
        if value > 0:
            return oparg, value - 1
        return pc + 2, value
    # A real use of obj (setfield_gc), not a call or an is-None check.
    _REF_SINK.last = obj
    return -1, value

interpret_one_ref._pe_static_args_ = ("opcode",)
interpret_one_ref._pe_split_args_ = ("pc",)
interpret_one_ref._pe_hole_args_ = ("obj",)


def _toy_ref_setup(code):
    from rpython.jit.codewriter.codewriter import CodeWriter
    from rpython.jit.codewriter.test.test_codewriter import FakeCPU

    graph, translator = get_graph(interpret_one_ref,
                                  [int, int, int, int, RefHoleObj])
    extension = GeneratingExtension.from_step_function(
        translator, interpret_one_ref, [OP_DEC_JUMP, OP_HALT],
        toy_ref_decoder, ref_hole_names=("obj",))
    program = extension.generate(code)
    codewriter = CodeWriter(FakeCPU(translator.rtyper), [])
    emitter = ProgramEmitter(codewriter, None, "opcode", ("pc",),
                             ("pc", "oparg", "obj"), ("value",))
    emitter.precompile_fragments(extension.templates)
    return program, emitter


def test_ref_hole_reaches_native_constant_pool_as_the_bound_instance():
    from rpython.rtyper.annlowlevel import cast_instance_to_gcref

    bytecode = chr(OP_DEC_JUMP) + chr(0) + chr(OP_HALT) + chr(0)
    code = RefHoleObj(bytecode, tag=99)
    program, emitter = _toy_ref_setup(code)

    native_table = build_native_table(emitter._fragments)
    native_jitcode, _positions, _asm = emit_and_assemble_native_unoptimised(
        native_table, program, "native-ref", has_merge_points=False)

    expected_ref = cast_instance_to_gcref(code)
    assert expected_ref in native_jitcode.constants_r
    # A Ptr-typed hole must never carry the int HOLE_SENTINEL, ever.
    assert str(HOLE_SENTINEL) not in native_jitcode.dump()
