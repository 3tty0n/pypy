"""
Tests for direct ResOperation generation (DirectTraceBuilder)
"""
import py
from rpython.jit.codewriter.flatten import SSARepr, Label, Register
from rpython.jit.codewriter.jitcode import JitCode
from rpython.jit.codewriter.assembler import Assembler
from rpython.jit.codewriter.genextension import GenExtension, DirectTraceBuilder
from rpython.jit.metainterp.history import ConstInt, ConstPtr, ConstFloat
from rpython.jit.metainterp.resoperation import rop
from rpython.rtyper.lltypesystem import lltype, llmemory
from rpython.flowspace.model import Constant


class FakeMetaInterp:
    def __init__(self):
        self.recorded_ops = []

    def execute_and_record(self, opnum, descr, *args):
        """Record the operation that was executed"""
        self.recorded_ops.append((opnum, descr, args))
        # Return a fake result box
        if opnum in (rop.INT_ADD, rop.INT_SUB, rop.INT_MUL, rop.INT_FLOORDIV,
                     rop.INT_MOD, rop.INT_AND, rop.INT_OR, rop.INT_XOR,
                     rop.INT_LSHIFT, rop.INT_RSHIFT):
            result_val = 42  # Fake result
            return ConstInt(result_val)
        elif opnum in (rop.INT_LT, rop.INT_LE, rop.INT_EQ, rop.INT_NE,
                       rop.INT_GT, rop.INT_GE):
            return ConstInt(1)  # Fake boolean result
        return None


class FakeMIFrame:
    def __init__(self, jitcode):
        self.jitcode = jitcode
        self.pc = 0
        self.metainterp = FakeMetaInterp()
        self.registers_i = [ConstInt(0)] * 256
        self.registers_r = [ConstPtr(lltype.nullptr(llmemory.GCREF.TO))] * 256
        self.registers_f = [ConstFloat(0.0)] * 256


def test_direct_trace_builder_created():
    ssarepr = SSARepr("test", genextension=True)
    i0, i1 = Register('int', 0), Register('int', 1)
    ssarepr.insns = [
        ('int_add', i0, i1, '->', i1),
        ('int_return', i1),
    ]

    assembler = Assembler()
    jitcode = assembler.assemble(ssarepr, num_regs={'int': 2})

    assert jitcode.genext_trace_builder is not None
    assert jitcode.genext_function is None
    assert isinstance(jitcode.genext_trace_builder, DirectTraceBuilder)


def test_old_code_generation_still_works():
    ssarepr = SSARepr("test", genextension=True)
    i0, i1 = Register('int', 0), Register('int', 1)
    ssarepr.insns = [
        ('int_add', i0, i1, '->', i1),
        ('int_return', i1),
    ]

    assembler = Assembler()
    jitcode = assembler.assemble(ssarepr, num_regs={'int': 2})
    # Manually call generate with use_direct_ops=False
    GenExtension(assembler, ssarepr, jitcode).generate(use_direct_ops=False)

    assert jitcode.genext_function is not None
    assert jitcode.genext_trace_builder is None
    assert callable(jitcode.genext_function)


def test_direct_trace_builder_has_work_list():
    ssarepr = SSARepr("test", genextension=True)
    i0, i1 = Register('int', 0), Register('int', 1)
    ssarepr.insns = [
        ('int_add', i0, i1, '->', i1),
        ('int_return', i1),
    ]

    assembler = Assembler()
    jitcode = assembler.assemble(ssarepr, num_regs={'int': 2})

    trace_builder = jitcode.genext_trace_builder
    assert trace_builder.work_list is not None
    assert hasattr(trace_builder.work_list, 'get_specializer_at_pc')


def test_work_list_get_specializer_at_pc():
    ssarepr = SSARepr("test", genextension=True)
    i0, i1 = Register('int', 0), Register('int', 1)
    ssarepr.insns = [
        ('int_add', i0, i1, '->', i1),
        ('int_return', i1),
    ]

    assembler = Assembler()
    jitcode = assembler.assemble(ssarepr, num_regs={'int': 2})

    trace_builder = jitcode.genext_trace_builder
    work_list = trace_builder.work_list

    # Should find specializer at pc=0 (int_add instruction)
    spec = work_list.get_specializer_at_pc(0)
    assert spec is not None
    assert spec.name == 'int_add'


def test_specializer_has_execute_direct_method():
    ssarepr = SSARepr("test", genextension=True)
    i0, i1 = Register('int', 0), Register('int', 1)
    ssarepr.insns = [
        ('int_add', i0, i1, '->', i1),
        ('int_return', i1),
    ]

    assembler = Assembler()
    jitcode = assembler.assemble(ssarepr, num_regs={'int': 2})

    trace_builder = jitcode.genext_trace_builder
    spec = trace_builder.work_list.get_specializer_at_pc(0)

    assert hasattr(spec, 'execute_direct')
    assert callable(spec.execute_direct)


def test_direct_int_add_specialized_handler_exists():
    ssarepr = SSARepr("test", genextension=True)
    i0, i1 = Register('int', 0), Register('int', 1)
    ssarepr.insns = [
        ('int_add', i0, i1, '->', i1),
        ('int_return', i1),
    ]

    assembler = Assembler()
    jitcode = assembler.assemble(ssarepr, num_regs={'int': 2})

    trace_builder = jitcode.genext_trace_builder
    spec = trace_builder.work_list.get_specializer_at_pc(0)

    # Should have a direct handler for int_add
    handler = getattr(spec, '_direct_specialized_int_add', None)
    assert handler is not None
    assert callable(handler)


def test_direct_int_binary_op_helper_exists():
    """Test that the generic binary op helper exists"""
    ssarepr = SSARepr("test", genextension=True)
    i0, i1 = Register('int', 0), Register('int', 1)
    ssarepr.insns = [
        ('int_add', i0, i1, '->', i1),
        ('int_return', i1),
    ]

    assembler = Assembler()
    jitcode = assembler.assemble(ssarepr, num_regs={'int': 2})

    trace_builder = jitcode.genext_trace_builder
    spec = trace_builder.work_list.get_specializer_at_pc(0)

    # Should have the generic helper
    helper = getattr(spec, '_direct_int_binary_op', None)
    assert helper is not None
    assert callable(helper)


def test_both_approaches_produce_compatible_jitcode():
    ssarepr1 = SSARepr("test", genextension=True)
    i0, i1 = Register('int', 0), Register('int', 1)
    ssarepr1.insns = [
        ('int_add', i0, i1, '->', i1),
        ('int_return', i1),
    ]

    # Direct ops approach
    assembler1 = Assembler()
    jitcode_direct = assembler1.assemble(ssarepr1, num_regs={'int': 2})

    # Code generation approach - use same SSARepr structure
    ssarepr2 = SSARepr("test", genextension=True)
    ssarepr2.insns = [
        ('int_add', i0, i1, '->', i1),
        ('int_return', i1),
    ]

    assembler2 = Assembler()
    jitcode_codegen = assembler2.assemble(ssarepr2, num_regs={'int': 2})
    jitcode_codegen.genext_trace_builder = None
    GenExtension(assembler2, ssarepr2, jitcode_codegen).generate(use_direct_ops=False)

    assert jitcode_direct.code == jitcode_codegen.code

    assert (jitcode_direct.genext_trace_builder is not None) != (
        jitcode_codegen.genext_trace_builder is not None)


def test_direct_operations_coverage():
    operations = [
        'int_add', 'int_sub', 'int_mul', 'int_floordiv', 'int_mod',
        'int_and', 'int_or', 'int_xor', 'int_lshift', 'int_rshift',
        'int_lt', 'int_le', 'int_eq', 'int_ne', 'int_gt', 'int_ge'
    ]

    for op_name in operations:
        ssarepr = SSARepr("test_" + op_name, genextension=True)
        i0, i1 = Register('int', 0), Register('int', 1)
        i2 = Register('int', 2)

        ssarepr.insns = [
            (op_name, i0, i1, '->', i2),
            ('int_return', i2),
        ]

        assembler = Assembler()
        jitcode = assembler.assemble(ssarepr, num_regs={'int': 3})

        trace_builder = jitcode.genext_trace_builder
        spec = trace_builder.work_list.get_specializer_at_pc(0)

        # Check that the handler exists
        handler_name = '_direct_specialized_' + op_name
        handler = getattr(spec, handler_name, None)
        assert handler is not None, "Missing handler for %s" % op_name
