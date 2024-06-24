import py
import pytest
from rpython.jit import metainterp
import sys
from rpython.rlib.rarithmetic import intmask
from rpython.rlib.rarithmetic import LONG_BIT
from rpython.rtyper import rclass
from rpython.rtyper.lltypesystem import lltype
from rpython.jit.metainterp.optimize import InvalidLoop
from rpython.jit.metainterp.optimizeopt.foldarray import (
    OptFoldArray, DependencyGraph, optimize_foldarray)
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

class TestDependencyGraph(BaseTest):
    def _setup(self, ops, optops, call_pure_results=None):
        loop = self.parse(ops)
        token = JitCellToken()
        if loop.operations[-1].getopnum() == rop.JUMP:
            loop.operations[-1].setdescr(token)
        exp = parse(optops, namespace=self.namespace.copy())
        expected = convert_old_style_to_targets(exp, jump=True)
        call_pure_results = self._convert_call_pure_results(call_pure_results)
        trace = convert_loop_to_trace(loop, self.metainterp_sd)
        jitdriver_sd = FakeJitDriverStaticData()

        optimize_foldarray(trace, self.metainterp_sd, jitdriver_sd, {})
        return loop, exp

    def test_dependency_graph_1(self):
        ops = """
        [p0, p1, i2]
        p3 = getarrayitem_gc_r(p0, 0, descr=arraydescr)
        i3 = int_add(i2, 1)
        i4 = int_add(i3, 2)
        setarrayitem_gc(p0, i4, p3, descr=arraydescr)
        p4 = getarrayitem_gc_r(p0, i4, descr=arraydescr) # p4 == p3
        p5 = getfield_gc_r(p4, descr=valuedescr)
        jump(p0, p5, i4)
        """

        optops = """
        [p0, p1, i2]
        p3 = getarrayitem_gc_r(p0, 0, descr=arraydescr)
        p5 = getfield_gc_r(p3, descr=valuedescr)
        jump(p0, p5, i2)
        """

        self._setup(ops, optops)


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
        compile_data = compile.SimpleFoldArrayLoopData(
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
        p5 = getfield_gc_r(p4, descr=valuedescr)
        jump(p1, p5)
        """

        optops = """
        [p0, p1]
        p2 = getarrayitem_gc_r(p0, 0, descr=arraydescr)
        p5 = getfield_gc_r(p2, descr=valuedescr)
        jump(p1, p5)
        """

        self.analyze(ops, optops)

    def test_remove_setarray_2(self):
        ops = """
        [p0, p1, i2]
        p3 = getarrayitem_gc_r(p0, 0, descr=arraydescr)
        i3 = int_add(i2, 1)
        setarrayitem_gc(p1, i3, p3)
        p4 = getarrayitem_gc_r(p1, i3, descr=arraydescr)
        p5 = getfield_gc_r(p4, descr=valuedescr)
        jump(p1, p5, i2)
        """

        optops = """
        [p0, p1, i2]
        p2 = getarrayitem_gc_r(p0, 0, descr=arraydescr)
        i3 = int_add(i2, 1)
        p5 = getfield_gc_r(p2, descr=valuedescr)
        jump(p1, p5, i3)
        """

        self.analyze(ops, optops)
