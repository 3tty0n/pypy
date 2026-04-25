"""Unit tests for the persistent trace cache (Idea 3)."""

import os

import pytest

from rpython.jit.metainterp import tracecache as tc


class FakeConst(object):
    """Minimal stand-in for a greenkey Const; fingerprint_greenkey inspects
    `type` and tries a few value accessors."""

    def __init__(self, type_, value):
        self.type = type_
        self.value = value

    def __repr__(self):
        return '<FakeConst %s=%r>' % (self.type, self.value)


class FakeBox(object):
    """Minimal stand-in for a ResOperation-produced box. Implements the
    getint/getopnum/getarglist/getdescr/is_constant surface used by the
    tracecache builder.
    """

    def __init__(self, type_='i', is_const=False, value=None):
        self.type = type_
        self._is_const = is_const
        self._value = value

    def is_constant(self):
        return self._is_const

    def getint(self):
        return int(self._value)

    def getfloatstorage(self):
        return float(self._value)

    def getref_base(self):
        return self._value


class FakeOp(object):
    def __init__(self, opnum, args, type_='v', descr=None):
        self.opnum = opnum
        self.args = args
        self.type = type_
        self.descr = descr

    def getopnum(self):   return self.opnum
    def getarglist(self): return list(self.args)
    def getdescr(self):   return self.descr


class FakeLoop(object):
    def __init__(self, inputargs, operations):
        self.inputargs = inputargs
        self.operations = operations


def _enable(monkeypatch, tmp_path):
    path = str(tmp_path / 'cache')
    os.makedirs(path)
    monkeypatch.setattr(tc, 'CACHE_DIR', path)
    return path


def test_disabled_by_default(monkeypatch):
    monkeypatch.setattr(tc, 'CACHE_DIR', '')
    assert not tc.is_enabled()
    assert tc.load('anything') is None
    entry = tc.TraceCacheEntry([], [])
    assert tc.store('anything', entry) is False


def test_fingerprint_greenkey_stable():
    gk1 = [FakeConst('i', 1), FakeConst('i', 2)]
    gk2 = [FakeConst('i', 1), FakeConst('i', 2)]
    gk3 = [FakeConst('i', 1), FakeConst('i', 3)]
    assert tc.fingerprint_greenkey(gk1) == tc.fingerprint_greenkey(gk2)
    assert tc.fingerprint_greenkey(gk1) != tc.fingerprint_greenkey(gk3)
    assert tc.fingerprint_greenkey(None) == tc.fingerprint_greenkey(None)


def test_fingerprint_type_profile_order_independent():
    a = {'x': 'int', 'y': 'float'}
    b = {'y': 'float', 'x': 'int'}
    assert tc.fingerprint_type_profile(a) == tc.fingerprint_type_profile(b)
    c = {'x': 'int', 'y': 'str'}
    assert tc.fingerprint_type_profile(a) != tc.fingerprint_type_profile(c)


def test_make_key_includes_jit_version(monkeypatch):
    gk = [FakeConst('i', 1)]
    tp = {'x': 'int'}
    k1 = tc.make_key(gk, tp)
    monkeypatch.setattr(tc, 'JIT_VERSION', 'X' * 16)
    k2 = tc.make_key(gk, tp)
    assert k1 != k2


# --- wire format roundtrip -------------------------------------------------

def test_wire_roundtrip_simple():
    builder = tc._Builder()
    b0 = FakeBox('i')
    b1 = FakeBox('i')
    c_int = FakeBox('i', is_const=True, value=42)
    r0 = FakeOp(7, [b0, c_int], type_='i')
    r1 = FakeOp(11, [r0, b1], type_='v')
    entry = builder.build([b0, b1], [r0, r1])
    payload = tc.encode_entry(entry)
    back = tc.decode_entry(payload)
    assert len(back.inputargs) == 2
    assert len(back.operations) == 2
    op0 = back.operations[0]
    assert op0.opnum == 7
    assert op0.result_type == ord('i')
    assert op0.args[0].tag == tc.TAG_BOX_REF
    assert op0.args[1].tag == tc.TAG_CONST_INT
    assert op0.args[1].value == 42
    op1 = back.operations[1]
    assert op1.args[0].tag == tc.TAG_BOX_REF
    # First op's result id must match op1's first arg (box identity).
    assert op1.args[0].value == op0.result_box_id


def test_wire_roundtrip_float_const():
    builder = tc._Builder()
    a = FakeBox('f')
    c = FakeBox('f', is_const=True, value=1.5)
    op = FakeOp(3, [a, c], type_='f')
    entry = builder.build([a], [op])
    back = tc.decode_entry(tc.encode_entry(entry))
    assert back.operations[0].args[1].tag == tc.TAG_CONST_FLT
    assert abs(back.operations[0].args[1].value - 1.5) < 1e-9


def test_wire_rejects_truncated():
    with pytest.raises(ValueError):
        tc.decode_entry(b'\x00\x00\x00\x01')  # incomplete


def test_store_and_load_roundtrip(monkeypatch, tmp_path):
    _enable(monkeypatch, tmp_path)
    builder = tc._Builder()
    a = FakeBox('i'); b = FakeBox('i')
    op = FakeOp(9, [a, b], type_='i')
    entry = builder.build([a, b], [op])
    entry.meta = {'num_ops': 1}
    key = tc.make_key([FakeConst('i', 1)], {'x': 'int'})
    assert tc.store(key, entry)
    loaded = tc.load(key)
    assert loaded is not None
    assert len(loaded.operations) == 1
    assert loaded.operations[0].opnum == 9
    assert loaded.meta == {'num_ops': 1}


def test_version_mismatch_rejects(monkeypatch, tmp_path):
    _enable(monkeypatch, tmp_path)
    entry = tc._Builder().build([], [])
    key = tc.make_key([FakeConst('i', 1)], {'x': 'int'})
    tc.store(key, entry)
    monkeypatch.setattr(tc, 'JIT_VERSION', 'X' * 16)
    assert tc.load(key) is None


def test_assumption_validation(monkeypatch, tmp_path):
    _enable(monkeypatch, tmp_path)
    entry = tc._Builder().build([], [])
    records = [{'kind': 'quasi_immutable', 'id': 'obj#1'}]
    entry.assumptions = records
    key = tc.make_key([FakeConst('i', 1)], {'x': 'int'})
    tc.store(key, entry, assumption_records=records)

    def reject(assumptions):
        return False
    assert tc.load(key, validate=reject) is None

    def accept(assumptions):
        # Round-trips as Assumption(kind_tag, payload_str); kind 0x01 is
        # ASSUMP_QUASI_IMMUT and payload is the string-ified id.
        assert len(assumptions) == 1
        a = assumptions[0]
        assert a.kind == tc.ASSUMP_QUASI_IMMUT
        assert a.payload == 'obj#1'
        return True
    got = tc.load(key, validate=accept)
    assert got is not None


def test_invalidate_removes_file(monkeypatch, tmp_path):
    path = _enable(monkeypatch, tmp_path)
    entry = tc._Builder().build([], [])
    key = tc.make_key([FakeConst('i', 1)], {})
    tc.store(key, entry)
    assert os.path.exists(os.path.join(path, key + '.tc'))
    tc.invalidate(key)
    assert not os.path.exists(os.path.join(path, key + '.tc'))


def test_corrupt_file_returns_none(monkeypatch, tmp_path):
    path = _enable(monkeypatch, tmp_path)
    key = 'abc123'
    with open(os.path.join(path, key + '.tc'), 'wb') as f:
        f.write(b'not a valid cache file')
    assert tc.load(key) is None


# --- replay path -----------------------------------------------------------

def test_replay_structural():
    builder = tc._Builder()
    a = FakeBox('i'); b = FakeBox('i')
    c = FakeBox('i', is_const=True, value=7)
    op0 = FakeOp(4, [a, c], type_='i')
    op1 = FakeOp(5, [op0, b], type_='v')
    entry = builder.build([a, b], [op0, op1])
    # Roundtrip through bytes.
    entry2 = tc.decode_entry(tc.encode_entry(entry))
    inputs, ops = tc.replay_to_operations(entry2)
    assert len(inputs) == 2
    assert len(ops) == 2
    # op1's first arg must be the same reconstructed box as op0's result.
    assert ops[1].getarglist()[0] is ops[0]
    # Const survives as a stub holding the value.
    const_arg = ops[0].getarglist()[1]
    assert const_arg.value == 7


def test_replay_with_descr_resolver():
    class MyDescr(object):
        def __repr__(self): return 'MyDescr()'
    d = MyDescr()
    builder = tc._Builder()
    a = FakeBox('i')
    op = FakeOp(3, [a], type_='i', descr=d)
    entry = builder.build([a], [op])
    entry2 = tc.decode_entry(tc.encode_entry(entry))

    calls = []

    def resolver(descr_id, repr_str):
        calls.append((descr_id, repr_str))
        return 'RESOLVED-' + repr_str
    inputs, ops = tc.replay_to_operations(entry2, descr_resolver=resolver)
    assert ops[0].getdescr() == 'RESOLVED-MyDescr()'
    assert len(calls) == 1


def test_build_entry_from_loop():
    a = FakeBox('i'); b = FakeBox('i')
    op = FakeOp(2, [a, b], type_='i')
    loop = FakeLoop([a, b], [op])
    entry = tc.build_entry_from_loop(loop, meta={'num_ops': 1})
    assert entry.meta == {'num_ops': 1}
    assert len(entry.inputargs) == 2
    assert len(entry.operations) == 1
