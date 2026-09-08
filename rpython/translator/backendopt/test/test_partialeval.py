from rpython.flowspace.model import checkgraph, summary
from rpython.translator.translator import TranslationContext, graphof
from rpython.rtyper.llinterp import LLInterpreter
from rpython.translator.backendopt.partialeval import (PartialEvaluator,
    specialize_graph, specialize_entry_point, specialize_variant,
    specialize_split_graph, install_split_graph, partial_evaluate,
    make_rtyped_constant)
from rpython.translator.backendopt.partialeval_template import (
    AbsoluteTarget, Branch, Continue, Finish, NextPcHole, PcHole,
    RelativeTarget, ResidualTemplateGenerator, COMPACT_ENTRY_MIN_BLOCKS,
    uses_compact_entries)
from rpython.translator.backendopt.generating_extension import (
    GeneratingExtension)


def test_compact_entries_are_reserved_for_large_programs():
    class Program(object):
        pass

    program = Program()
    program.blocks = dict.fromkeys(range(COMPACT_ENTRY_MIN_BLOCKS - 1))
    assert not uses_compact_entries(program)
    program.blocks[COMPACT_ENTRY_MIN_BLOCKS - 1] = None
    assert uses_compact_entries(program)


def byte_pair_decoder(code, pc):
    opcode = ord(code[pc])
    oparg = ord(code[pc + 1])
    return opcode, {"pc": pc, "oparg": oparg, "code": code}


def get_graph(fn, signature):
    t = TranslationContext()
    t.buildannotator().build_types(fn, signature)
    t.buildrtyper().specialize()
    return graphof(t, fn), t

def run_graph(graph, t, args):
    checkgraph(graph)
    interp = LLInterpreter(t.rtyper)
    return interp.eval_graph(graph, args)

def to_llvalue(t, var, value):
    s_value = t.annotator.binding(var)
    r_value = t.rtyper.getrepr(s_value)
    return r_value.convert_const(value)


def assert_codewriter_accepts(graph, translator):
    from rpython.jit.codewriter.codewriter import CodeWriter
    from rpython.jit.codewriter.test.test_codewriter import (
        FakeCPU, FakeJitDriverSD, FakePolicy)
    jitdriver_sd = FakeJitDriverSD(graph)
    codewriter = CodeWriter(FakeCPU(translator.rtyper), [jitdriver_sd])
    codewriter.find_all_graphs(FakePolicy())
    codewriter.make_jitcodes()
    return jitdriver_sd.mainjitcode


def test_pe_driver_split_metadata():
    from rpython.rlib.pe import PEDriver

    def dispatch(code, pc, x):
        return x

    PEDriver(static="code", split="pc").bind(dispatch)
    assert dispatch._pe_entry_point_
    assert dispatch._pe_static_args_ == ("code",)
    assert dispatch._pe_split_args_ == ("pc",)

def test_specialize_static_integer():
    def f(x, y):
        return x * 2 + y

    f._pe_entry_point_ = True
    f._pe_static_args_ = ("x",)

    graph, t = get_graph(f, [int, int])

    residual = specialize_entry_point(t, graph, {"x": 10})

    assert summary(residual) == {"int_add": 1}


def test_specialize_static_branch():
    def f(x, y):
        if x == 0:
            return y + 1
        else:
            return y * 2

    f._pe_entry_point_ = True
    f._pe_static_args_ = ("x",)

    graph, t = get_graph(f, [int, int])
    residual = specialize_entry_point(t, graph, {"x": 0})
    assert summary(residual) == {"int_add": 1}

    interp = LLInterpreter(t.rtyper)
    assert interp.eval_graph(residual, [999, 10]) == 11


LOAD = 0
ADD = 1
HALT = 2

def test_specialize_static_string_getitem():
    def lookup(code, pc):
        return ord(code[pc])

    lookup._pe_entry_point_ = True
    lookup._pe_static_args_ = ("code", "pc",)

    code = chr(LOAD) + chr(ADD)
    graph, t = get_graph(lookup, [str, int])

    residual = specialize_entry_point(
        t, graph, {"code": code, "pc": 0})

    ops = summary(residual)
    print(ops)

    assert "direct_call" not in ops
    assert "cast_char_to_int" not in ops

def test_specicalize_dispatch_simple_1():
    opcode = chr(LOAD) + chr(ADD)

    def dispatch(opcode, pc, x):
        opcode = ord(opcode[pc])
        if opcode == LOAD:
            return x + 10
        elif opcode == ADD:
            return x + 20
        return -1

    dispatch._pe_entry_point_ = True
    dispatch._pe_static_args_ = ("opcode", "pc",)

    graph, t = get_graph(dispatch, [str, int, int])

    residual = specialize_entry_point(
        t, graph, {"opcode": opcode, "pc": 1})

    ops = summary(residual)
    assert "int_eq" not in ops
    assert ops == {"int_add": 1}

    interp = LLInterpreter(t.rtyper)
    ll_code = to_llvalue(t, graph.startblock.inputargs[0], opcode)
    assert interp.eval_graph(residual, [ll_code, 999, 1]) == 20 + 1


def test_specialize_dispatch_simple_2():
    code = chr(LOAD) + chr(ADD)

    def dispatch(code, pc, x):
        opcode = ord(code[pc])

        if opcode == LOAD:
            return 1, x + 10
        elif opcode == ADD:
            return 2, x + 20

        return -1, x

    dispatch._pe_entry_point_ = True
    dispatch._pe_static_args_ = ("code",)

    graph, t = get_graph(dispatch, [str, int, int])

    residual = specialize_entry_point(
        t,
        graph,
        {"code": code},
    )

    ops = summary(residual)
    assert "int_eq" in ops

    ll_code = to_llvalue(t, graph.startblock.inputargs[0], code)
    interp = LLInterpreter(t.rtyper)
    res = interp.eval_graph(residual, [ll_code, 0, 1])
    assert res.item0 == 1 and res.item1 == 11
    res = interp.eval_graph(residual, [ll_code, 1, 1])
    assert res.item0 == 2 and res.item1 == 21


def test_specialize_dispatch_simple_3():
    code = chr(LOAD) + chr(ADD) + chr(HALT)

    def dispatch(code, pc, x):
        opcode = ord(code[pc])

        if opcode == LOAD:
            return 1, x + 10
        elif opcode == ADD:
            return 2, x + 20
        elif opcode == HALT:
            return -1, x
        return -1, x

    dispatch._pe_entry_point_ = True
    dispatch._pe_static_args_ = ("code",)
    dispatch._pe_split_args_ = ("pc",)

    graph, t = get_graph(dispatch, [str, int, int])

    pe = PartialEvaluator(t)
    residual0 = specialize_variant(
        pe, graph, {"code": code}, {"pc": 0})
    residual1 = specialize_variant(
        pe, graph, {"code": code}, {"pc": 1})

    assert residual0 is not residual1
    assert summary(residual0) == {"int_add": 1, "malloc": 1,
                                  "setfield": 2}
    assert summary(residual1) == {"int_add": 1, "malloc": 1,
                                  "setfield": 2}
    assert specialize_variant(
        pe, graph, {"code": code}, {"pc": 0}) is residual0

    ll_code = to_llvalue(t, graph.startblock.inputargs[0], code)
    interp = LLInterpreter(t.rtyper)
    res = interp.eval_graph(residual0, [ll_code, 999, 1])
    assert res.item0 == 1 and res.item1 == 11
    res = interp.eval_graph(residual1, [ll_code, 999, 1])
    assert res.item0 == 2 and res.item1 == 21

    connected = specialize_split_graph(
        PartialEvaluator(t), graph, {"code": code}, {"pc": 0})
    res = interp.eval_graph(connected, [ll_code, 999, 1])
    assert res.item0 == -1 and res.item1 == 31


def test_connect_split_backedge_reuses_variant():
    code = chr(LOAD) + chr(ADD)

    def dispatch(code, pc, x):
        opcode = ord(code[pc])
        if opcode == LOAD:
            return 1, x + 10
        return 0, x + 20

    dispatch._pe_entry_point_ = True
    dispatch._pe_static_args_ = ("code",)
    dispatch._pe_split_args_ = ("pc",)

    graph, t = get_graph(dispatch, [str, int, int])
    pe = PartialEvaluator(t)
    connected = specialize_split_graph(
        pe, graph, {"code": code}, {"pc": 0})

    checkgraph(connected)
    assert len(pe.cache) == 2
    assert summary(connected) == {"int_add": 2}


def test_connect_all_split_exits_in_dynamic_cfg():
    code = chr(LOAD) + chr(ADD)

    def dispatch(code, pc, x):
        opcode = ord(code[pc])
        if opcode == LOAD:
            if x <= 0:
                return -1, x
            return 1, x - 1
        if x <= 0:
            return -1, x
        return 0, x - 1

    dispatch._pe_entry_point_ = True
    dispatch._pe_static_args_ = ("code",)
    dispatch._pe_split_args_ = ("pc",)

    graph, t = get_graph(dispatch, [str, int, int])
    pe = PartialEvaluator(t)
    connected = specialize_split_graph(
        pe, graph, {"code": code}, {"pc": 0})

    checkgraph(connected)
    assert len(pe.cache) == 2
    ops = summary(connected)
    assert ops["int_le"] == 2
    assert ops["int_sub"] == 2

    ll_code = to_llvalue(t, graph.startblock.inputargs[0], code)
    interp = LLInterpreter(t.rtyper)
    res = interp.eval_graph(connected, [ll_code, 999, 5])
    assert res.item0 == -1 and res.item1 == 0

    assert "goto" in assert_codewriter_accepts(connected, t).dump()


def test_install_split_graph_replaces_registered_entry_graph():
    code = chr(LOAD) + chr(HALT)

    def dispatch(code, pc, x):
        opcode = ord(code[pc])
        if opcode == LOAD:
            return 1, x + 10
        return -1, x

    dispatch._pe_entry_point_ = True
    dispatch._pe_static_args_ = ("code",)
    dispatch._pe_split_args_ = ("pc",)

    graph, t = get_graph(dispatch, [str, int, int])
    original_startblock = graph.startblock
    ll_code = to_llvalue(t, original_startblock.inputargs[0], code)
    graph_count = len(t.graphs)
    installed = install_split_graph(
        PartialEvaluator(t), graph, {"code": code}, {"pc": 0})

    assert installed is graph
    assert graphof(t, dispatch) is graph
    assert graph in t.graphs
    assert len(t.graphs) == graph_count
    assert graph.startblock is not original_startblock
    assert "int_eq" not in summary(graph)

    res = LLInterpreter(t.rtyper).eval_graph(graph, [ll_code, 999, 3])
    assert res.item0 == -1 and res.item1 == 13


def test_split_variants_share_exception_block():
    code = chr(LOAD) + chr(ADD)

    def dispatch(code, pc, x):
        opcode = ord(code[pc])
        if x < 0:
            raise ValueError
        if opcode == LOAD:
            return 1, x - 1
        return 0, x - 1

    dispatch._pe_entry_point_ = True
    dispatch._pe_static_args_ = ("code",)
    dispatch._pe_split_args_ = ("pc",)

    graph, t = get_graph(dispatch, [str, int, int])
    pe = PartialEvaluator(t)
    connected = specialize_split_graph(
        pe, graph, {"code": code}, {"pc": 0})

    checkgraph(connected)
    assert len(pe.cache) == 2
    exitblocks = [block for block in connected.iterblocks()
                  if not block.exits]
    assert exitblocks == [connected.exceptblock]


def test_partial_evaluate_installs_split_entry_point():
    code = chr(LOAD) + chr(HALT)

    def dispatch(code, pc, x):
        if ord(code[pc]) == LOAD:
            return 1, x + 1
        return -1, x

    dispatch._pe_entry_point_ = True
    dispatch._pe_static_args_ = ("code",)
    dispatch._pe_split_args_ = ("pc",)

    graph, t = get_graph(dispatch, [str, int, int])
    ll_code = to_llvalue(t, graph.startblock.inputargs[0], code)
    installed = partial_evaluate(
        t, {dispatch: ({"code": code}, {"pc": 0})})

    assert installed == [graph]
    assert graphof(t, dispatch) is graph
    assert "int_eq" not in summary(graph)
    res = LLInterpreter(t.rtyper).eval_graph(graph, [ll_code, 999, 4])
    assert res.item0 == -1 and res.item1 == 5


def test_residual_template_ir_and_catalog():
    code = chr(LOAD) + chr(HALT)

    def dispatch(code, pc, x):
        if ord(code[pc]) == LOAD:
            return 1, x + 1
        return -1, x

    dispatch._pe_static_args_ = ("code",)
    dispatch._pe_split_args_ = ("pc",)
    graph, t = get_graph(dispatch, [str, int, int])
    pe = PartialEvaluator(t)

    template0 = pe.make_template(
        LOAD, graph, {"code": code}, {"pc": 0})
    template1 = pe.make_template(
        HALT, graph, {"code": code}, {"pc": 1})

    assert isinstance(template0.terminators[0], Continue)
    assert template0.terminators[0].target == 1
    assert isinstance(template1.terminators[0], Finish)
    assert PcHole().kind == "pc"

    extension = GeneratingExtension({LOAD: template0, HALT: template1},
                                    byte_pair_decoder, "opcode")
    assert extension.templates[LOAD] is template0
    assert set(extension.templates) == set([LOAD, HALT])
    assert extension.handles(LOAD) and not extension.handles(LOAD + 99)


def test_symbolic_template_targets_resolve_without_concrete_code():
    generator = ResidualTemplateGenerator()
    fallthrough = generator.symbolic_fallthrough(
        "LOAD_FAST", ("load-fast",), ("frame",), 3)
    absolute = generator.symbolic_absolute_jump(
        "JUMP_ABSOLUTE", (), ("frame",))
    relative = generator.symbolic_relative_jump(
        "JUMP_FORWARD", (), ("frame",), 3)
    branch = generator.symbolic_absolute_branch(
        "POP_JUMP_IF_FALSE", ("is-true",), "condition", ("frame",), 3)
    finish = generator.symbolic_finish(
        "RETURN_VALUE", ("pop",), ("result",))

    assert isinstance(fallthrough.terminators[0].target, NextPcHole)
    assert fallthrough.resolve_targets({"pc": 10}) == [[13]]
    assert isinstance(absolute.terminators[0].target, AbsoluteTarget)
    assert absolute.resolve_targets({"oparg": 24}) == [[24]]
    assert isinstance(relative.terminators[0].target, RelativeTarget)
    assert relative.resolve_targets({"pc": 10, "oparg": 7}) == [[20]]
    assert isinstance(branch.terminators[0], Branch)
    assert branch.terminators[0].condition == "condition"
    assert branch.resolve_targets({"pc": 10, "oparg": 24}) == [[13, 24]]
    assert isinstance(finish.terminators[0], Finish)
    assert finish.resolve_targets({}) == [[]]


def test_offline_pe_of_small_symbolic_interpreter():
    OP_ADD = 0
    OP_JUMP = 1
    OP_HALT = 2

    def interpret_one(opcode, oparg, pc, value):
        if opcode == OP_ADD:
            return pc + 1, value + oparg
        if opcode == OP_JUMP:
            return oparg, value
        return -1, value

    interpret_one._pe_static_args_ = ("opcode",)
    interpret_one._pe_split_args_ = ("pc",)
    graph, t = get_graph(interpret_one, [int, int, int, int])
    original_ops = summary(graph)
    assert original_ops["int_eq"] == 2

    pe = PartialEvaluator(t)
    add = pe.make_symbolic_template(
        OP_ADD, graph, {"opcode": OP_ADD})
    jump = pe.make_symbolic_template(
        OP_JUMP, graph, {"opcode": OP_JUMP})
    halt = pe.make_symbolic_template(
        OP_HALT, graph, {"opcode": OP_HALT})

    assert isinstance(add.holes[0], PcHole)
    assert add.holes[0].name == "pc"

    for opcode in [OP_ADD, OP_JUMP, OP_HALT]:
        residual = pe.specialize(
            graph, {"opcode": opcode}, {"pc": 0})
        assert_codewriter_accepts(residual, t)

    assert not any(op.opname == "int_eq" for op in add.operations)
    assert [op.opname for op in add.operations] == ["int_add"]
    assert add.resolve_targets({"pc": 10, "oparg": 7}) == [[11]]
    assert jump.resolve_targets({"pc": 10, "oparg": 7}) == [[7]]
    assert isinstance(halt.terminators[0], Finish)


def test_link_small_interpreter_dispatch_loop_without_tracing():
    OP_DEC_JUMP = 0
    OP_HALT = 1

    def interpret_one(opcode, oparg, pc, value):
        if opcode == OP_DEC_JUMP:
            if value > 0:
                return oparg, value - 1
            return pc + 2, value
        return -1, value

    def dispatch(code, value):
        pc = 0
        while pc >= 0:
            opcode = ord(code[pc])
            oparg = ord(code[pc + 1])
            pc, value = interpret_one(opcode, oparg, pc, value)
        return value

    interpret_one._pe_static_args_ = ("opcode",)
    interpret_one._pe_split_args_ = ("pc",)
    graph, t = get_graph(interpret_one, [int, int, int, int])
    pe = PartialEvaluator(t)

    extension = GeneratingExtension.from_step_function(
        t, interpret_one, [OP_DEC_JUMP, OP_HALT], byte_pair_decoder)

    for opcode in [OP_DEC_JUMP, OP_HALT]:
        residual = pe.specialize(
            graph, {"opcode": opcode}, {"pc": 0})
        assert_codewriter_accepts(residual, t)

    code = chr(OP_DEC_JUMP) + chr(0) + chr(OP_HALT) + chr(0)
    assert dispatch(code, 5) == 0

    linked = extension.generate(code)
    assert set(linked.blocks) == set([0, 2])
    assert set(linked.blocks[0].successors) == set([0, 2])
    assert 0 in linked.blocks[0].successors
    assert linked.blocks[2].has_finish
    assert linked.loop_headers == [0]
    assert linked.backedges == [(0, 0)]
    assert linked.blocks[0].is_loop_header

    for block in linked.blocks.values():
        assert not any(op.opname == "int_eq"
                       for op in block.template.operations)

    dec_graph = pe.specialize(
        graph, {"opcode": OP_DEC_JUMP}, {"pc": 0})
    jitcode = assert_codewriter_accepts(dec_graph, t)
    linked.attach_to_jitcode(jitcode, {0: 7, 2: 19})
    assert jitcode.pe_metadata.entry_pc == 0
    assert jitcode.pe_metadata.block_pcs == [0, 2]
    assert jitcode.pe_metadata.loop_headers == [0]
    assert jitcode.pe_metadata.backedge_sources == [0]
    assert jitcode.pe_metadata.backedge_targets == [0]
    assert jitcode.pe_metadata.entry_pcs == [0, 2]
    assert jitcode.pe_metadata.entry_positions == [7, 19]

    from rpython.jit.metainterp.pyjitpl import get_pe_trace_start_position
    assert get_pe_trace_start_position(jitcode) == 7


def test_generate_declines_or_falls_back_to_leave_template():
    """No template for an instruction declines, unless leave_key is set."""
    OP_DEC_JUMP = 0
    OP_HALT = 1
    OP_UNSUPPORTED = 2
    OP_LEAVE = 3

    def interpret_one(opcode, oparg, pc, value):
        if opcode == OP_DEC_JUMP:
            if value > 0:
                return oparg, value - 1
            return pc + 2, value
        return -1, value

    interpret_one._pe_static_args_ = ("opcode",)
    interpret_one._pe_split_args_ = ("pc",)
    graph, t = get_graph(interpret_one, [int, int, int, int])

    extension = GeneratingExtension.from_step_function(
        t, interpret_one, [OP_DEC_JUMP, OP_HALT, OP_LEAVE],
        byte_pair_decoder)

    code = chr(OP_DEC_JUMP) + chr(0) + chr(OP_UNSUPPORTED) + chr(0)

    assert extension.leave_key == -1
    assert extension.generate(code) is None
    assert extension.last_blocked == (2, OP_UNSUPPORTED)

    extension.leave_key = OP_LEAVE
    linked = extension.generate(code)
    assert linked is not None
    assert set(linked.blocks) == set([0, 2])
    assert linked.blocks[2].key == OP_LEAVE
    assert linked.blocks[2].has_finish
    assert linked.blocks[2].successors == []
    assert linked.leave_blocks == 1
    assert len(linked.blocks) == 2
    assert linked.leave_blocks != len(linked.blocks)

    entry_unsupported = chr(OP_UNSUPPORTED) + chr(0)
    all_leave = extension.generate(entry_unsupported)
    assert all_leave is not None
    assert all_leave.leave_blocks == len(all_leave.blocks) == 1


def test_meta_interpreter_starts_at_offline_loop_header():
    from rpython.jit.codewriter.jitcode import JitCode, PEJitCodeMetadata
    from rpython.jit.metainterp.pyjitpl import (
        MetaInterp, get_pe_trace_start_position)

    jitcode = JitCode("offline-loop")
    jitcode.pe_metadata = PEJitCodeMetadata(
        10, (10,), (10,), (), (), (10,), (23,))

    class FakeFrame(object):
        def setup_call(self, boxes):
            self.boxes = boxes
            self.pc = 0

    class FakeJitDriverSD(object):
        mainjitcode = jitcode
        num_green_args = 0
        num_red_args = 1

    metainterp = MetaInterp.__new__(MetaInterp)
    metainterp.jitdriver_sd = FakeJitDriverSD()
    class FakeStats(object):
        def pe_metadata_used(self):
            pass
    class FakeStaticData(object):
        stats = FakeStats()
    metainterp.staticdata = FakeStaticData()
    frame = FakeFrame()

    def newframe(jitcode_arg):
        assert jitcode_arg is jitcode
        metainterp.portal_call_depth = 0
        metainterp.framestack.append(frame)
        return frame

    metainterp.newframe = newframe
    metainterp.initialize_withgreenfields = lambda boxes: None
    metainterp.initialize_virtualizable = lambda boxes: None
    boxes = [object()]
    metainterp.initialize_state_from_start(boxes)

    assert frame.boxes == boxes
    assert frame.pc == 23

    jitcode.pe_metadata = PEJitCodeMetadata(
        10, (10,), (10,), (), (), (), ())
    assert get_pe_trace_start_position(jitcode) == 0
    jitcode.pe_metadata = PEJitCodeMetadata(
        10, (10, 20), (20,), (), (), (10,), (23,))
    assert get_pe_trace_start_position(jitcode) == 0


def test_meta_traces_small_interpreter_with_offline_metadata():
    from rpython.rlib.jit import JitDriver
    from rpython.jit.metainterp.test.support import LLJitMixin

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
    graph, t = get_graph(interpret_one, [int, int, int, int])
    pe = PartialEvaluator(t)
    extension = GeneratingExtension.from_step_function(
        t, interpret_one, [OP_DEC_JUMP, OP_HALT], byte_pair_decoder)

    code = chr(OP_DEC_JUMP) + chr(0) + chr(OP_HALT) + chr(0)
    linked = extension.generate(code)
    assert linked.loop_headers == [0]

    driver = JitDriver(greens=["pc"], reds=["value"])

    def dispatch(value):
        pc = 0
        while pc >= 0:
            driver.can_enter_jit(pc=pc, value=value)
            driver.jit_merge_point(pc=pc, value=value)
            opcode = ord(code[pc])
            oparg = ord(code[pc + 1])
            pc, value = interpret_one(opcode, oparg, pc, value)
        return value

    def attach_offline_metadata(jitcode):
        linked.attach_to_jitcode(jitcode, {0: 0, 2: 0})

    runner = LLJitMixin()
    result = runner.meta_interp(
        dispatch, [8], pe_jitcode_setup=attach_offline_metadata,
        enable_opts="")

    assert result == 0
    from rpython.jit.metainterp.warmspot import get_stats
    assert get_stats().pe_metadata_count > 0
    runner.check_trace_count(1)
    runner.check_resops(int_gt=1, int_sub=1,
                        strgetitem=0, int_eq=0)
