
# some unit tests for the bytecode decoding

import py
from rpython.jit.metainterp import pyjitpl
from rpython.jit.metainterp import jitprof
from rpython.jit.metainterp.history import ConstInt
from rpython.jit.metainterp.history import History, IntFrontendOp
from rpython.jit.metainterp.resoperation import ResOperation, rop, InputArgInt
from rpython.jit.metainterp.optimizeopt.util import equaloplists
from rpython.jit.codewriter.jitcode import JitCode


def test_portal_trace_positions():
    py.test.skip("bleh, too direct test, rewrite or kill")
    class jitdriver_sd:
        index = 0

        class warmstate:
            @staticmethod
            def get_unique_id(*args):
                return 0

        class jitdriver:
            is_recursive = True

    jitcode = JitCode("f")
    jitcode.setup(None)
    portal = JitCode("portal")
    portal.jitdriver_sd = jitdriver_sd
    portal.setup(None)
    class FakeStaticData:
        cpu = None
        warmstate = None
        warmrunnerdesc = None
        mainjitcode = portal

    metainterp = pyjitpl.MetaInterp(FakeStaticData(), FakeStaticData())
    metainterp.framestack = []
    class FakeHistory:
        operations = []

        @staticmethod
        def record(*args):
            pass
    history = metainterp.history = FakeHistory()
    metainterp.newframe(portal, "green1")
    history.operations.append(1)
    metainterp.newframe(jitcode)
    history.operations.append(2)
    metainterp.newframe(portal, "green2")
    history.operations.append(3)
    metainterp.popframe()
    history.operations.append(4)
    metainterp.popframe()
    history.operations.append(5)
    metainterp.popframe()
    history.operations.append(6)
    assert metainterp.portal_trace_positions == [("green1", 0), ("green2", 2),
                                                 (None, 3), (None, 5)]
    assert metainterp.find_biggest_function() == "green1"

    metainterp.newframe(portal, "green3")
    history.operations.append(7)
    metainterp.newframe(jitcode)
    history.operations.append(8)
    assert metainterp.portal_trace_positions == [("green1", 0), ("green2", 2),
                                                 (None, 3), (None, 5), ("green3", 6)]
    assert metainterp.find_biggest_function() == "green1"

    history.operations.extend([9, 10, 11, 12])
    assert metainterp.find_biggest_function() == "green3"

def test_remove_consts_and_duplicates():
    class FakeStaticData:
        cpu = None
        all_descrs = []
        warmrunnerdesc = None
    def is_another_box_like(box, referencebox):
        assert box is not referencebox
        assert box.type == referencebox.type
        assert box.getint() == referencebox.getint()
        return True
    metainterp = pyjitpl.MetaInterp(FakeStaticData(), None)
    metainterp.history = History(4, FakeStaticData())
    b1 = IntFrontendOp(1, 1)
    b2 = IntFrontendOp(2, 2)
    c3 = ConstInt(3)
    boxes = [b1, b2, b1, c3]
    dup = {}
    metainterp.history.set_inputargs([b1, b2])
    metainterp.remove_consts_and_duplicates(boxes, 4, dup)
    assert boxes[0] is b1
    assert boxes[1] is b2
    assert is_another_box_like(boxes[2], b1)
    assert is_another_box_like(boxes[3], c3)
    inp, operations = metainterp.history.trace.unpack()
    remap = dict(zip([b1, b2], inp))
    assert equaloplists(operations, [
        ResOperation(rop.SAME_AS_I, [b1]),
        ResOperation(rop.SAME_AS_I, [c3]),
        ], remap=remap)
    assert dup == {b1: None, b2: None}
    #

def test_get_name_from_address():
    class FakeMetaInterpSd(pyjitpl.MetaInterpStaticData):
        def __init__(self):
            pass
    metainterp_sd = FakeMetaInterpSd()
    metainterp_sd.setup_list_of_addr2name([(123, 'a'), (456, 'b')])
    assert metainterp_sd.get_name_from_address(123) == 'a'
    assert metainterp_sd.get_name_from_address(456) == 'b'
    assert metainterp_sd.get_name_from_address(789) == ''


def test_register_late_jitcode_extends_liveness_and_reindexes():
    """register_late_jitcode extends opcode tables/liveness, reindexes."""
    from rpython.jit.codewriter.assembler import Assembler
    from rpython.jit.codewriter.flatten import SSARepr, Register
    from rpython.jit.codewriter.jitcode import enumerate_vars
    from rpython.jit.metainterp.blackhole import BlackholeInterpBuilder

    class FakeMetaInterpSd(pyjitpl.MetaInterpStaticData):
        def __init__(self):
            pass

    asm = Assembler()

    def make_jitcode(name, live_regs, extra_insns=()):
        ssarepr = SSARepr(name)
        ssarepr.insns = [
            ('-live-',) + tuple(Register('int', i) for i in live_regs),
        ]
        ssarepr.insns.extend(extra_insns)
        num_regs = {'int': (max(live_regs) + 1) if live_regs else 0}
        return asm.assemble(ssarepr, num_regs=num_regs)

    def decode_liveness(jitcode, liveness_info):
        offset = jitcode.get_live_vars_info(0, asm.insns['live/'])
        seen = []
        enumerate_vars(offset, liveness_info, seen.append,
                       lambda i: None, lambda i: None, None)
        return seen

    jitcode_a = make_jitcode("a", [0])

    liveness_info = "".join(asm.all_liveness)
    assert decode_liveness(jitcode_a, liveness_info) == [0]

    metainterp_sd = FakeMetaInterpSd()
    metainterp_sd.jitcodes = [jitcode_a]
    jitcode_a.index = 0

    class FakeCodeWriter:
        cpu = None
    codewriter = FakeCodeWriter()
    codewriter.assembler = asm

    metainterp_sd.setup_insns(asm.insns)
    metainterp_sd.blackholeinterpbuilder = BlackholeInterpBuilder(
        codewriter, metainterp_sd)
    opcodes_before = len(metainterp_sd.opcode_implementations)
    bh_opcodes_before = len(metainterp_sd.blackholeinterpbuilder._insns)

    jitcode_b = make_jitcode("b", [0, 1],
                             extra_insns=[('int_return', Register('int', 0))])
    assert len(asm.insns) > opcodes_before

    metainterp_sd.register_late_jitcode(jitcode_b, codewriter)

    assert jitcode_b.index == 1
    assert metainterp_sd.jitcodes == [jitcode_a, jitcode_b]
    assert len(metainterp_sd.opcode_implementations) == len(asm.insns)
    assert len(metainterp_sd.blackholeinterpbuilder._insns) == len(asm.insns)
    assert len(metainterp_sd.opcode_implementations) > opcodes_before
    assert len(metainterp_sd.blackholeinterpbuilder._insns) > bh_opcodes_before
    assert metainterp_sd.opcode_names[:opcodes_before] == \
        metainterp_sd.blackholeinterpbuilder._insns[:opcodes_before]
    assert decode_liveness(jitcode_a, metainterp_sd.liveness_info) == [0]
    assert decode_liveness(jitcode_b, metainterp_sd.liveness_info) == [0, 1]
