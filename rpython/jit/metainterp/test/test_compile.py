from rpython.config.translationoption import get_combined_translation_config
from rpython.jit.metainterp.history import ConstInt, History, Stats
from rpython.jit.metainterp.history import INT
from rpython.jit.metainterp.compile import compile_loop
from rpython.jit.metainterp.compile import compile_tmp_callback
from rpython.jit.metainterp import jitexc
from rpython.rlib.rjitlog import rjitlog as jl
from rpython.jit.metainterp import jitprof, compile
from rpython.jit.metainterp.optimizeopt.test.test_util import LLtypeMixin
from rpython.jit.tool.oparser import parse, convert_loop_to_trace
from rpython.jit.metainterp.optimizeopt import ALL_OPTS_DICT
from rpython.jit.metainterp.resoperation import rop, ResOperation, InputArgRef

class FakeCPU(object):
    supports_guard_gc_type = True

    class Storage:
        pass

    class tracker:
        pass

    def __init__(self):
        self.seen = []

    def compile_loop(self, inputargs, operations, token, jd_id=0,
                     unique_id=0, log=True, name='',
                     logger=None):
        token.compiled_loop_token = self.Storage()
        self.seen.append((inputargs, operations, token))

class FakeLogger(object):
    def log_loop(self, inputargs, operations, number=0, type=None, ops_offset=None, name='', memo=None):
        pass

    def log_loop_from_trace(self, *args, **kwds):
        pass

    def repr_of_resop(self, op):
        return repr(op)

class FakeState(object):
    enable_opts = ALL_OPTS_DICT.copy()
    enable_opts.pop('unroll')
    pureop_historylength = 16

    def attach_unoptimized_bridge_from_interp(*args):
        pass

    def get_unique_id(*args):
        return 0

    def get_location_str(self, args):
        return 'location'

class FakeGlobalData(object):
    pass

class FakeMetaInterpStaticData(object):
    all_descrs = []
    logger_noopt = FakeLogger()
    logger_ops = FakeLogger()
    config = get_combined_translation_config(translating=True)
    jitlog = jl.JitLogger()

    stats = Stats(None)
    profiler = jitprof.EmptyProfiler()
    warmrunnerdesc = None
    def log(self, msg, event_kind=None):
        pass

class FakeMetaInterp:
    call_pure_results = {}
    box_names_memo = {}
    class jitdriver_sd:
        index = 0
        warmstate = FakeState()
        virtualizable_info = None
        vec = False


def test_genext_pure_arithmetic_shortcut_rejects_unsupported_ops():
    assert compile.GENEXT_PURE_COMPILE_SHORTCUT is False
    assert compile._is_pure_arithmetic_trace_op(rop.INT_ADD)
    assert compile._is_pure_arithmetic_trace_op(rop.FLOAT_TRUEDIV)
    assert compile._is_pure_arithmetic_trace_op(rop.GUARD_NO_OVERFLOW)
    assert compile._is_pure_arithmetic_trace_op(rop.INT_LT)
    assert compile._is_pure_arithmetic_trace_op(rop.UINT_GE)
    assert compile._is_pure_arithmetic_trace_op(rop.FLOAT_EQ)
    assert compile._is_pure_arithmetic_trace_op(rop.FLOAT_NE)
    assert compile._is_pure_arithmetic_trace_op(rop.INT_IS_TRUE)
    assert compile._is_pure_arithmetic_trace_op(rop.INT_IS_ZERO)
    assert compile._is_pure_arithmetic_trace_op(rop.GUARD_TRUE)
    assert compile._is_pure_arithmetic_trace_op(rop.GUARD_FALSE)
    assert not compile._is_pure_arithmetic_trace_op(rop.UINT_MUL_HIGH)


def test_genext_pure_arithmetic_shortcut_rejects_ref_failargs():
    guard_op = ResOperation(rop.GUARD_NO_OVERFLOW, [])
    guard_op.setfailargs([InputArgRef()])
    assert compile._has_ref_failargs(guard_op)

    int_guard_op = ResOperation(rop.GUARD_NO_OVERFLOW, [])
    int_guard_op.setfailargs([ConstInt(42), None])
    assert not compile._has_ref_failargs(int_guard_op)

def test_compile_loop():
    cpu = FakeCPU()
    staticdata = FakeMetaInterpStaticData()
    staticdata.all_descrs = LLtypeMixin.cpu.setup_descrs()
    staticdata.cpu = cpu
    staticdata.jitlog = jl.JitLogger(cpu)
    staticdata.jitlog.trace_id = 1
    #
    loop = parse('''
    [p1]
    i1 = getfield_gc_i(p1, descr=valuedescr)
    i2 = int_add(i1, 1)
    p2 = new_with_vtable(descr=nodesize)
    setfield_gc(p2, i2, descr=valuedescr)
    jump(p2)
    ''', namespace=LLtypeMixin.__dict__.copy())
    #
    metainterp = FakeMetaInterp()
    metainterp.staticdata = staticdata
    metainterp.cpu = cpu
    metainterp.history = History(len(loop.inputargs), staticdata)
    t = convert_loop_to_trace(loop, staticdata)
    metainterp.history.inputargs = t.inputargs
    metainterp.history.trace = t
    #
    greenkey = 'faked'
    target_token = compile_loop(
        metainterp, greenkey, (0, 0, 0, 0, 0), t.inputargs,
        [t._mapping[x] for x in loop.operations[-1].getarglist()],
        use_unroll=False)
    jitcell_token = target_token.targeting_jitcell_token
    assert jitcell_token == target_token.original_jitcell_token
    assert jitcell_token.target_tokens == [target_token]
    assert jitcell_token.number == 2
    #
    assert len(cpu.seen) == 1
    assert cpu.seen[0][2] == jitcell_token
    #
    del cpu.seen[:]


def test_compile_tmp_callback():
    from rpython.jit.backend.llgraph import runner
    from rpython.rtyper.lltypesystem import lltype, llmemory
    from rpython.rtyper.annlowlevel import llhelper
    from rpython.rtyper.llinterp import LLException
    #
    cpu = runner.LLGraphCPU(None)
    FUNC = lltype.FuncType([lltype.Signed]*4, lltype.Signed)
    def ll_portal_runner(g1, g2, r3, r4):
        assert (g1, g2, r3, r4) == (12, 34, -156, -178)
        if raiseme:
            raise raiseme
        else:
            return 54321
    #
    class FakeJitDriverSD:
        portal_runner_ptr = llhelper(lltype.Ptr(FUNC), ll_portal_runner)
        portal_runner_adr = llmemory.cast_ptr_to_adr(portal_runner_ptr)
        portal_calldescr = cpu.calldescrof(FUNC, FUNC.ARGS, FUNC.RESULT, None)
        portal_finishtoken = compile.DoneWithThisFrameDescrInt()
        propagate_exc_descr = compile.PropagateExceptionDescr()
        num_red_args = 2
        result_type = INT
    #
    loop_token = compile_tmp_callback(cpu, FakeJitDriverSD(),
                                      [ConstInt(12), ConstInt(34)], "ii")
    #
    raiseme = None
    # only two arguments must be passed in
    deadframe = cpu.execute_token(loop_token, -156, -178)
    fail_descr = cpu.get_latest_descr(deadframe)
    assert fail_descr is FakeJitDriverSD().portal_finishtoken
    #
    EXC = lltype.GcStruct('EXC')
    llexc = lltype.malloc(EXC)
    raiseme = LLException("exception class", llexc)
    deadframe = cpu.execute_token(loop_token, -156, -178)
    fail_descr = cpu.get_latest_descr(deadframe)
    assert isinstance(fail_descr, compile.PropagateExceptionDescr)
    got = cpu.grab_exc_value(deadframe)
    assert lltype.cast_opaque_ptr(lltype.Ptr(EXC), got) == llexc
    #
    class FakeMetaInterpSD:
        pass
    FakeMetaInterpSD.cpu = cpu
    deadframe = cpu.execute_token(loop_token, -156, -178)
    fail_descr = cpu.get_latest_descr(deadframe)
    try:
        fail_descr.handle_fail(deadframe, FakeMetaInterpSD(), None)
    except jitexc.ExitFrameWithExceptionRef as e:
        assert lltype.cast_opaque_ptr(lltype.Ptr(EXC), e.value) == llexc
    else:
        assert 0, "should have raised"


def test_hbp_should_throttle_dual_gate():
    from rpython.jit.metainterp.compile import (
        ResumeGuardDescr, HBP_BRIDGE_STORM_THRESHOLD)

    class FakeJitCode(object):
        genext_hbp_candidate = True

    class FakeJitDriverSD(object):
        mainjitcode = FakeJitCode()

    class FakeLoopToken(object):
        n_compiled_bridges = 0
        outermost_jitdriver_sd = FakeJitDriverSD()

    class FakeCLT(object):
        def __init__(self, lt):
            self._lt = lt
        def loop_token_wref(self):
            return self._lt

    class DeadCLT(object):
        def loop_token_wref(self):
            return None

    lt = FakeLoopToken()
    d = object.__new__(ResumeGuardDescr)

    # The throttle is dual-gated: dynamic bridge-storm confirmation plus the
    # static genext HBP-candidate classification.

    # gate absent: rd_loop_token defaults to None
    assert d._hbp_should_throttle() is False

    d.rd_loop_token = FakeCLT(lt)
    # at/below the storm threshold
    lt.n_compiled_bridges = HBP_BRIDGE_STORM_THRESHOLD
    assert d._hbp_should_throttle() is False

    # both sub-gates open -> throttle
    lt.n_compiled_bridges = HBP_BRIDGE_STORM_THRESHOLD + 1
    assert d._hbp_should_throttle() is True

    # static sub-gate closed -> no throttle
    FakeJitCode.genext_hbp_candidate = False
    assert d._hbp_should_throttle() is False
    FakeJitCode.genext_hbp_candidate = True

    # dead weakref -> no throttle
    d.rd_loop_token = DeadCLT()
    assert d._hbp_should_throttle() is False
