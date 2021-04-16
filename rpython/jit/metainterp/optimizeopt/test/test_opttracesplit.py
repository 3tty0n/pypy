import py
import sys
import re
from rpython.rlib.rarithmetic import intmask
from rpython.rlib.rarithmetic import LONG_BIT
from rpython.rlib.rjitlog import rjitlog as jl
from rpython.rtyper import rclass
from rpython.rtyper.lltypesystem import lltype
from rpython.rtyper.annlowlevel import llhelper
from rpython.jit.codewriter.effectinfo import EffectInfo
from rpython.jit.backend.llgraph import runner
from rpython.jit.metainterp import pyjitpl
from rpython.jit.metainterp.jitprof import EmptyProfiler
from rpython.jit.metainterp.optimize import InvalidLoop
from rpython.jit.metainterp.optimizeopt.test.test_util import (
    BaseTest, convert_old_style_to_targets)
from rpython.jit.metainterp.history import (
    JitCellToken, ConstInt, get_const_ptr_for_string)
from rpython.jit.metainterp import executor, compile
from rpython.jit.metainterp.resoperation import (
    rop, ResOperation, InputArgInt, OpHelpers, InputArgRef)
from rpython.jit.metainterp.support import ptr2int
from rpython.jit.metainterp.optimizeopt.intdiv import magic_numbers
from rpython.jit.metainterp.test.test_resume import (
    ResumeDataFakeReader, MyMetaInterp)
from rpython.jit.metainterp.optimizeopt.test import test_util
from rpython.jit.tool.oparser import parse, convert_loop_to_trace

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

# ____________________________________________________________


def merge_dicts(*dict_args):
    """
    Given any number of dictionaries, shallow copy and merge into a new dict,
    precedence goes to key-value pairs in latter dictionaries.
    """
    result = {}
    for dictionary in dict_args:
        result.update(dictionary)
    return result


class FakeMetaInterpStaticData(pyjitpl.MetaInterpStaticData):
    all_descrs = []

    def __init__(self, cpu):
        self.cpu = cpu
        self.profiler = EmptyProfiler()
        self.options = test_util.Fake()
        self.globaldata =test_util.Fake()
        self.config = test_util.get_combined_translation_config(translating=True)
        self.jitlog = jl.JitLogger()
        self.callinfocollection = test_util.FakeCallInfoCollection()

    class logger_noopt:
        @classmethod
        def log_loop(*args, **kwds):
            pass

        @classmethod
        def log_loop_from_trace(*args, **kwds):
            pass

    class logger_ops:
        repr_of_resop = repr

    class warmrunnerdesc:
        class memory_manager:
            retrace_limit = 5
            max_retrace_guards = 15
        jitcounter = test_util.DeterministicJitCounter()

    def get_name_from_address(self, ptr):
        try:
            return repr(ptr)
        except AttributeError:
            return ""

class BaseTestTraceSplit(BaseTest):

    enable_opts = "intbounds:rewrite:virtualize:string:earlyforce:pure:heap:tracesplit"

    def optimize_loop(self, ops, optops, call_pure_results=None):
        cpu = runner.LLGraphCPU(None)
        Ptr = lltype.Ptr
        FuncType = lltype.FuncType
        FPTR = Ptr(FuncType([lltype.Char], lltype.Char))

        def cut_here(c):
            return c

        func_ptr = llhelper(FPTR, cut_here)
        calldescr = cpu.calldescrof(FPTR.TO, (lltype.Char,), lltype.Char,
                                EffectInfo.MOST_GENERAL)

        namespace = merge_dicts(test_util.LLtypeMixin.__dict__.copy(), locals().copy())
        self.namespace = namespace
        self.metainterp_sd = FakeMetaInterpStaticData(cpu)

        loop = self.parse(ops)
        token = JitCellToken()
        if loop.operations[-1].getopnum() == rop.JUMP:
            loop.operations[-1].setdescr(token)
        exp = parse(optops, namespace=namespace)
        expected = convert_old_style_to_targets(exp, jump=True)
        call_pure_results = self._convert_call_pure_results(call_pure_results)
        trace = convert_loop_to_trace(loop, self.metainterp_sd)
        compile_data = compile.SimpleCompileData(
            trace, call_pure_results=call_pure_results,
            enable_opts=self.enable_opts)
        info, ops = compile_data.optimize_trace(self.metainterp_sd, None, {})
        label_op = ResOperation(rop.LABEL, info.inputargs)
        loop.inputargs = info.inputargs
        loop.operations = [label_op] + ops
        self.loop = loop
        self.assert_equal(loop, expected)

class TestOptTraceSplit(BaseTestTraceSplit):

    def test_trace_split(self):
        ops ="""
        [p0]
        i1 = getfield_gc_i(p0, descr=valuedescr)
        i2 = call_i(ConstClass(func_ptr), descr=calldescr)
        i3 = int_add(i1, 1)
        i4 = call_i(p0, descr=plaincalldescr)
        jump(i3)
        """
        expected ="""
        [p0]
        i1 = getfield_gc_i(p0, descr=valuedescr)
        i3 = int_add(i1, 1)
        i4 = call_i(p0, descr=plaincalldescr)
        jump(i3)
        """
        self.optimize_loop(ops, expected)
