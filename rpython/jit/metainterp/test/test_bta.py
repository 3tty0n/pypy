from rpython.jit.metainterp import bta

class FakePyCode(object):
    def __init__(self, code, nlocals=8, argcount=0):
        self.co_code = code
        self.co_nlocals = nlocals
        self.co_argcount = argcount
        self.co_name = 'test'


def make_load_const(idx):
    return chr(bta.LOAD_CONST) + chr(idx & 0xFF) + chr((idx >> 8) & 0xFF)


def make_load_fast(idx):
    return chr(bta.LOAD_FAST) + chr(idx & 0xFF) + chr((idx >> 8) & 0xFF)


def make_store_fast(idx):
    return chr(bta.STORE_FAST) + chr(idx & 0xFF) + chr((idx >> 8) & 0xFF)


def make_simple_op(opcode):
    return chr(opcode)

def make_binop(opcode):
    if opcode >= bta.HAVE_ARGUMENT:
        return chr(opcode) + '\x00\x00'
    return chr(opcode)


def make_return():
    return chr(bta.RETURN_VALUE)


def test_load_const_is_static():
    # LOAD_CONST 5
    code = make_load_const(5) + make_return()
    pycode = FakePyCode(code, nlocals=0)
    info = bta.analyse_pycode(pycode)
    assert info.offset_bt[0] == bta.BT_STATIC
    assert info.stack_top_bt[0] == bta.BT_STATIC


def test_load_fast_unknown():
    # LOAD_FAST 0
    code = make_load_fast(0) + make_return()
    pycode = FakePyCode(code, nlocals=1)
    info = bta.analyse_pycode(pycode)
    assert info.offset_bt[0] == bta.BT_DYNAMIC


def test_store_fast_then_load_is_static():
    # LOAD_CONST 7 ; STORE_FAST 0 ; LOAD_FAST 0 ; RETURN
    code = (make_load_const(7) +
            make_store_fast(0) +
            make_load_fast(0) +
            make_return())
    pycode = FakePyCode(code, nlocals=1)
    info = bta.analyse_pycode(pycode)
    # offset of LOAD_FAST should see the local as static
    load_fast_off = len(make_load_const(7) + make_store_fast(0))
    assert info.get_local_bt(load_fast_off, 0) == bta.BT_STATIC
    assert info.offset_bt[load_fast_off] == bta.BT_STATIC


def test_binary_add_of_constants():
    # LOAD_CONST 1 ; LOAD_CONST 2 ; BINARY_ADD ; RETURN
    code = (make_load_const(1) +
            make_load_const(2) +
            make_binop(bta.BINARY_ADD) +
            make_return())
    pycode = FakePyCode(code, nlocals=0)
    info = bta.analyse_pycode(pycode)
    add_off = len(make_load_const(1) + make_load_const(2))
    assert info.offset_bt[add_off] == bta.BT_STATIC
    assert info.stack_top_bt[add_off] == bta.BT_STATIC


def test_binary_add_mixed():
    # LOAD_FAST 0 ; LOAD_CONST 2 ; BINARY_ADD ; RETURN
    code = (make_load_fast(0) +
            make_load_const(2) +
            make_binop(bta.BINARY_ADD) +
            make_return())
    pycode = FakePyCode(code, nlocals=1)
    info = bta.analyse_pycode(pycode)
    add_off = len(make_load_fast(0) + make_load_const(2))
    assert info.offset_bt[add_off] == bta.BT_DYNAMIC


def test_dup_top():
    # LOAD_CONST 1 ; DUP_TOP ; RETURN
    code = (make_load_const(1) +
            make_binop(bta.DUP_TOP) +
            make_return())
    pycode = FakePyCode(code, nlocals=0)
    info = bta.analyse_pycode(pycode)
    dup_off = len(make_load_const(1))
    assert info.offset_bt[dup_off] == bta.BT_STATIC


def test_pop_top():
    # LOAD_CONST 1 ; POP_TOP ; LOAD_CONST 2 ; RETURN
    code = (make_load_const(1) +
            make_binop(bta.POP_TOP) +
            make_load_const(2) +
            make_return())
    pycode = FakePyCode(code, nlocals=0)
    info = bta.analyse_pycode(pycode)
    pop_off = len(make_load_const(1))
    load2_off = pop_off + len(make_binop(bta.POP_TOP))
    assert info.offset_bt[pop_off] == bta.BT_UNKNOWN
    assert info.offset_bt[load2_off] == bta.BT_STATIC
