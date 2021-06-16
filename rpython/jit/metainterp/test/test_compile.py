from rpython.config.translationoption import get_combined_translation_config
from rpython.jit.codewriter.effectinfo import EffectInfo
from rpython.jit.metainterp.resoperation import rop
from rpython.jit.metainterp.optimizeopt.util import equaloplists
from rpython.jit.metainterp.history import ConstInt, History, Stats
from rpython.jit.metainterp.history import INT
from rpython.jit.metainterp.compile import (
    compile_loop, compile_simple_and_split, compile_tmp_callback, make_jitcell_token)
from rpython.jit.metainterp import jitexc
from rpython.rlib.rjitlog import rjitlog as jl
from rpython.jit.metainterp import jitprof, compile
from rpython.jit.metainterp.optimizeopt.test.test_util import LLtypeMixin
from rpython.jit.tool.oparser import op_parser, parse, parse_with_vars, convert_loop_to_trace
from rpython.jit.metainterp.optimizeopt import ALL_OPTS_DICT
from rpython.rtyper.annlowlevel import llhelper
from rpython.rtyper.lltypesystem import lltype

from pprint import pprint

class JitCode(object):
    def __init__(self, index):
        self.index = index

class FakeCPU(object):
    supports_guard_gc_type = True

    class Storage:
        pass

    class tracker:
        pass

    def __init__(self):
        self.seen = []

    def calldescrof(self, FUNC, ARGS, RESULT, effect_info):
        from rpython.jit.backend.llgraph.runner import CallDescr
        return CallDescr(RESULT, ARGS, effect_info)

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

class FakeFrame(object):
    parent_snapshot = None

    def __init__(self, pc, jitcode, boxes):
        self.pc = pc
        self.jitcode = jitcode
        self.boxes = boxes

    def get_list_of_active_boxes(self, flag, new_array, encode):
        a = new_array(len(self.boxes))
        for i, box in enumerate(self.boxes):
            a[i] = encode(box)
        return a

def unpack_snapshot(t, op, pos):
    op.framestack = []
    si = t.get_snapshot_iter(op.rd_resume_position)
    virtualizables = si.unpack_array(si.vable_array)
    vref_boxes = si.unpack_array(si.vref_array)
    for snapshot in si.framestack:
        jitcode, pc = si.unpack_jitcode_pc(snapshot)
        boxes = si.unpack_array(snapshot.box_array)
        op.framestack.append(FakeFrame(JitCode(jitcode), pc, boxes))
    op.virtualizables = virtualizables
    op.vref_boxes = vref_boxes

def unpack(t):
    iter = t.get_iter()
    l = []
    try:
        while not iter.done():
            op = iter.next()
            if op.is_guard():
                unpack_snapshot(iter, op, op.rd_resume_position)
            l.append(op)
    except Exception:
        pass
    return iter.inputargs, l

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
    metainterp.history = History()
    t = convert_loop_to_trace(loop, staticdata)
    metainterp.history.inputargs = t.inputargs
    metainterp.history.trace = t
    #
    greenkey = 'faked'
    target_token = compile_loop(
        metainterp, greenkey, (0, 0, 0), t.inputargs,
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


def test_compile_simple_loop_and_split():
    from rpython.jit.metainterp.support import ptr2int
    from rpython.jit.metainterp.pyjitpl import MetaInterpStaticData
    from rpython.rtyper.annlowlevel import llhelper
    from rpython.rtyper.lltypesystem import lltype, llmemory

    cpu = FakeCPU()

    class FakeMetaInterpStaticData(MetaInterpStaticData):
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

        def __init__(self):
            pass

    def merge(dic1, dic2):
        new_dic = dic1.copy()
        new_dic.update(dic2)
        return new_dic

    Ptr = lltype.Ptr
    FuncType = lltype.FuncType
    FPTR = Ptr(FuncType([lltype.Char], lltype.Char))
    def cut_here(c):
        return c

    func_ptr = llhelper(FPTR, cut_here)
    cutheredescr = cpu.calldescrof(FPTR.TO, (lltype.Char,), lltype.Char,
                                       EffectInfo.MOST_GENERAL)

    staticdata = FakeMetaInterpStaticData()
    staticdata.all_descrs = LLtypeMixin.cpu.setup_descrs()
    staticdata.cpu = cpu
    staticdata.jitlog = jl.JitLogger(cpu)
    staticdata.jitlog.trace_id = 2
    staticdata.setup_list_of_addr2name([(ptr2int(func_ptr), 'cut_here')])

    metainterp = FakeMetaInterp()
    metainterp.staticdata = staticdata
    metainterp.cpu = cpu
    metainterp.history = History()

    namespace = merge(LLtypeMixin.__dict__.copy(), locals().copy())

    loop = parse('''
    [p1]
    i1 = getfield_gc_i(p1, descr=valuedescr)
    i2 = int_add(i1, 1)
    i3 = int_gt(i2, 0)
    i4 = call_i(ConstClass(func_ptr), descr=cutheredescr) # calling cut_here pseudo function
    i5 = getfield_gc_i(p1, descr=valuedescr)
    i6 = int_add(i5, 2)
    jump(i6)
    ''', namespace=namespace)

    loop_before = parse('''
    [p1]
    i1 = getfield_gc_i(p1, descr=valuedescr)
    i2 = int_add(i1, 1)
    i3 = int_gt(i2, 0)
    i4 = call_i(ConstClass(func_ptr), descr=cutheredescr)
    # TODO: jump instruction here
    ''', namespace=namespace)

    loop_after = parse('''
    [p1]
    i5 = getfield_gc_i(p1, descr=valuedescr)
    i6 = int_add(i5, 2)
    jump(i6)
    ''', namespace=namespace)

    t = convert_loop_to_trace(loop, staticdata)
    metainterp.history.trace = t
    metainterp.history.inputargs = t.inputargs

    raiseme = None
    greenkey = 'faked'
    t_after_cutted, t_before_cutted = compile_simple_and_split(
        metainterp, greenkey, t, t.inputargs,
        metainterp.jitdriver_sd.warmstate.enable_opts,
        (0, 0, 0))

    t_after = convert_loop_to_trace(loop_after, staticdata)
    i0, ops = t_after.unpack()
    i0_c, ops_c = unpack(t_after_cutted)
    assert ops_c != []
    assert len(i0) == len(i0_c)
    assert len(ops) == len(ops_c)

    t_before = convert_loop_to_trace(loop_before, staticdata)
    i1, ops = t_before.unpack()
    i1_c, ops_c = t_before_cutted.unpack()
    assert ops_c != []
    assert len(i1) == len(i1_c)
    assert len(ops) == len(ops_c)


def test_compile_simple_loop_and_split2():
    from rpython.jit.metainterp.history import BasicFailDescr
    from rpython.jit.metainterp.support import ptr2int
    from rpython.jit.metainterp.pyjitpl import MetaInterpStaticData
    from rpython.rtyper.annlowlevel import llhelper
    from rpython.rtyper.lltypesystem import lltype, llmemory

    cpu = FakeCPU()

    class FakeMetaInterpStaticData(MetaInterpStaticData):
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

        def __init__(self):
            pass

    def merge(dic1, dic2):
        new_dic = dic1.copy()
        new_dic.update(dic2)
        return new_dic

    Ptr = lltype.Ptr
    FuncType = lltype.FuncType
    FPTR = Ptr(FuncType([lltype.Char], lltype.Char))
    def cut_here(c):
        return c
    cuthere_ptr = llhelper(FPTR, cut_here)
    cuthereescr = cpu.calldescrof(FPTR.TO, (lltype.Char,), lltype.Char,
                                  EffectInfo.MOST_GENERAL)
    def func(x):
        return x
    func_ptr = llhelper(FPTR, func)
    calldescr = cpu.calldescrof(FPTR.TO, (lltype.Number,), lltype.Number,
                                EffectInfo.MOST_GENERAL)

    faildescr = BasicFailDescr(1)
    faildescr2 = BasicFailDescr(2)
    faildescr3 = BasicFailDescr(3)

    staticdata = FakeMetaInterpStaticData()
    staticdata.all_descrs = LLtypeMixin.cpu.setup_descrs()
    staticdata.cpu = cpu
    staticdata.jitlog = jl.JitLogger(cpu)
    staticdata.jitlog.trace_id = 2
    staticdata.setup_list_of_addr2name([(ptr2int(cuthere_ptr), 'cut_here')])

    metainterp = FakeMetaInterp()
    metainterp.staticdata = staticdata
    metainterp.cpu = cpu
    metainterp.history = History()

    namespace = merge(LLtypeMixin.__dict__.copy(), locals().copy())

    # simplified version without guard
    trace_str = """
    [p0]
    debug_merge_point(0, 0, '0: DUP ')
    p1 = getfield_gc_i(p0, descr=valuedescr)
    i3 = strgetitem(p1, 0)
    i7 = call_i(ConstClass(func_ptr), p0, 1, descr=calldescr)
    debug_merge_point(0, 0, '1: CONST_INT 1')
    i12 = call_i(ConstClass(func_ptr), p0, 2, descr=calldescr)
    debug_merge_point(0, 0, '3: LT ')
    i16 = call_i(ConstClass(func_ptr), p0, 4, descr=calldescr)
    debug_merge_point(0, 0, '4: JUMP_IF 8')
    i18 = getfield_gc_i(p0, descr=valuedescr)
    i20 = int_sub(i18, 1)
    p21 = getfield_gc_i(p0, descr=valuedescr)
    p22 = getarrayitem_gc_i(p21, i20, descr=arraydescr)
    setarrayitem_gc(p21, i20, ConstPtr(nullptr), descr=arraydescr)
    i25 = call_i(ConstClass(func_ptr), p0, p22, descr=calldescr)
    setfield_gc(p0, i20, descr=valuedescr)
    debug_merge_point(0, 0, '6: JUMP 13')
    i28 = call_i(ConstClass(cuthere_ptr), 6, descr=cuthereescr) # splitting point
    debug_merge_point(0, 0, '6: JUMP 13')
    debug_merge_point(0, 0, '13: EXIT ')
    i31 = int_sub(i20, 1)
    p32 = getarrayitem_gc_i(p21, i31, descr=arraydescr)
    setarrayitem_gc(p21, i31, ConstPtr(nullptr), descr=valuedescr)
    leave_portal_frame(0)
    setfield_gc(p0, i31, descr=valuedescr)
    finish(p32)
    """

    trace_before = parse("""
    [p0]
    debug_merge_point(0, 0, '0: DUP ')
    p1 = getfield_gc_i(p0, descr=valuedescr)
    i3 = strgetitem(p1, 0)
    i7 = call_i(ConstClass(func_ptr), p0, 1, descr=calldescr)
    debug_merge_point(0, 0, '1: CONST_INT 1')
    i12 = call_i(ConstClass(func_ptr), p0, 2, descr=calldescr)
    debug_merge_point(0, 0, '3: LT ')
    i16 = call_i(ConstClass(func_ptr), p0, 4, descr=calldescr)
    debug_merge_point(0, 0, '4: JUMP_IF 8')
    i18 = getfield_gc_i(p0, descr=valuedescr)
    i20 = int_sub(i18, 1)
    p21 = getfield_gc_i(p0, descr=valuedescr)
    p22 = getarrayitem_gc_i(p21, i20, descr=arraydescr)
    setarrayitem_gc(p21, i20, ConstPtr(nullptr), descr=arraydescr)
    i25 = call_i(ConstClass(func_ptr), p0, p22, descr=calldescr)
    setfield_gc(p0, i20, descr=valuedescr)
    debug_merge_point(0, 0, '6: JUMP 13')
    i28 = call_i(ConstClass(cuthere_ptr), 6, descr=cuthereescr)
    """, namespace=namespace)

    trace_after = parse("""
    [p0, p21, i20]
    debug_merge_point(0, 0, '6: JUMP 13')
    debug_merge_point(0, 0, '13: EXIT ')
    i31 = int_sub(i20, 1)
    p32 = getarrayitem_gc_i(p21, i31, descr=arraydescr)
    setarrayitem_gc(p21, i31, ConstPtr(nullptr), descr=valuedescr)
    leave_portal_frame(0)
    setfield_gc(p0, i31, descr=valuedescr)
    finish(p32)
    """, namespace=namespace)

    trace = parse(trace_str, namespace=namespace)

    t = convert_loop_to_trace(trace, staticdata)
    metainterp.history.trace = t
    metainterp.history.inputargs = t.inputargs

    # test version of copying ops from then branch
    total_count = t._count
    pos, count, index = t.cut_point_by_fname("cut_here")
    i_t, ops_t = unpack(t)

    t_ops_before = ops_t[:count - 1]
    t_ops_after = ops_t[count - 1:]
    l = []
    for i in range(count, total_count):
        op = ops_t[i - 1]
        args = op.getarglist()
        for arg in args:
            if arg in t_ops_before:
                if arg not in l:
                    l.append(arg)
    t_ops_after = l + t_ops_after
    pprint(t_ops_after)


    raiseme = None
    greenkey = 'faked'
    t_after_cutted, t_before_cutted = compile_simple_and_split(
        metainterp, greenkey, t, t.inputargs,
        metainterp.jitdriver_sd.warmstate.enable_opts,
        (0, 0, 0))

    t_before = convert_loop_to_trace(trace_before, staticdata)
    i0, ops = t_before.unpack()
    i0_c, ops_c = unpack(t_before_cutted)
    assert ops_c != []
    assert len(i0) == len(i0_c)
    assert len(ops) == len(ops_c)
