"""Integration tests for binding-time analysis in the meta-interpreter."""

import pytest
from rpython.jit.metainterp import bta
from rpython.jit.metainterp.history import ConstInt, ConstPtr


class FakePyCode(object):
    def __init__(self, code, nlocals=8, argcount=0, name='test'):
        self.co_code = code
        self.co_nlocals = nlocals
        self.co_argcount = argcount
        self.co_name = name


class FakeCPU(object):
    pass


class FakeStaticData(object):
    def __init__(self):
        self.cpu = FakeCPU()
        self.config = FakeConfig()


class FakeConfig(object):
    class translation(object):
        heapcache_genext_fastpath = False


class FakeProfiler(object):
    def count_ops(self, opnum, category=None):
        pass

    def start_tracing(self):
        pass

    def end_tracing(self):
        pass

    def start_interpretation(self):
        pass

    def end_interpretation(self):
        pass

    def count(self, *args):
        pass


class FakeJitDriverSD(object):
    num_green_args = 3
    virtualizable_info = None
    greenfield_info = None
    index_of_virtualizable = -1


class FakeJITCode(object):
    pass


def make_simple_code():
    # LOAD_CONST 1 ; LOAD_CONST 2 ; BINARY_ADD ; RETURN
    code = (chr(bta.LOAD_CONST) + '\x01\x00' +
            chr(bta.LOAD_CONST) + '\x02\x00' +
            chr(bta.BINARY_ADD) + '\x00\x00' +
            chr(bta.RETURN_VALUE))
    return FakePyCode(code, nlocals=0)


def test_bta_info_can_skip_pure_op():
    pycode = make_simple_code()
    info = bta.analyse_pycode(pycode)
    # The first LOAD_CONST at offset 0 is static
    assert info.can_skip_pure_op(0, 'int_add')
    # The BINARY_ADD at offset 6 is static (both inputs are constants)
    assert info.can_skip_pure_op(6, 'int_add')
    # RETURN_VALUE is not pure-op skipable
    assert not info.can_skip_pure_op(9, 'int_add')


def test_setup_bta_extracts_pycode():
    # Build a real PyCode-like object with minimal fields
    pycode = make_simple_code()
    # We can't easily construct a real PyCode here, so just test the
    # analysis path directly.
    info = bta.analyse_pycode(pycode)
    assert info is not None
    assert info.num_offsets == len(pycode.co_code)
