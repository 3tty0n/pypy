import py
import pytest
from rpython.jit import metainterp
import sys
from rpython.rlib.rarithmetic import intmask
from rpython.rlib.rarithmetic import LONG_BIT
from rpython.rtyper import rclass
from rpython.rtyper.lltypesystem import lltype
from rpython.jit.metainterp.optimize import InvalidLoop
from rpython.jit.metainterp.optimizeopt.foldarray import OptFoldAarray
from rpython.jit.metainterp.optimizeopt.test.test_util import (
    BaseTest, convert_old_style_to_targets, FakeJitDriverStaticData)
from rpython.jit.metainterp.history import (
    JitCellToken, ConstInt, get_const_ptr_for_string)
from rpython.jit.metainterp import executor, compile, resume
from rpython.jit.metainterp.resoperation import (
    rop, ResOperation, InputArgInt, OpHelpers, InputArgRef)
from rpython.jit.metainterp.test.test_resume import (
    ResumeDataFakeReader, MyMetaInterp)
from rpython.jit.tool.oparser import parse, convert_loop_to_trace

class BaseTestOptFoldArray(BaseTest):

    enable_opts = "foldarray"

    def analyze(self, ops, optops, call_pure_results=None):
        loop = self.parse(ops)
        token = JitCellToken()
        if loop.operations[-1].getopnum() == rop.JUMP:
            loop.operations[-1].setdescr(token)
        exp = parse(optops, namespace=self.namespace.copy())
        expected = convert_old_style_to_targets(exp, jump=True)
        call_pure_results = self._convert_call_pure_results(call_pure_results)
        trace = convert_loop_to_trace(loop, self.metainterp_sd)
        jitdriver_sd = FakeJitDriverStaticData()
        compile_data = compile.SimpleCompileData(
            trace, call_pure_results=call_pure_results,
            enable_opts=self.enable_opts)
        info, ops = compile_data.optimize_trace(self.metainterp_sd, jitdriver_sd, {})
        label_op = ResOperation(rop.LABEL, info.inputargs)
        loop.inputargs = info.inputargs
        loop.operations = [label_op] + ops
        self.loop = loop
        self.assert_equal(loop, expected)


class TestOptFoldArray(BaseTestOptFoldArray):

    def test_remove_setarray_1(self):
        ops = """
        [p0, p1]
        p2 = getarrayitem_gc_r(p0, 0, descr=arraydescr)
        p3 = getarrayitem_gc_r(p1, 0, descr=arraydescr)
        setarrayitem_gc(p1, 1, p2)
        p4 = getarrayitem_gc_r(p1, 1, descr=arraydescr)
        """

        optops = """
        [p0, p1]
        p2 = getarrayitem_gc_r(p0, 0, descr=arraydescr)
        p3 = getarrayitem_gc_r(p1, 0, descr=arraydescr)
        setarrayitem_gc(p1, 1, p2) # should be removed later
        p4 = getarrayitem_gc_r(p0, 0, descr=arraydescr)
        """

        self.analyze(ops, optops)
