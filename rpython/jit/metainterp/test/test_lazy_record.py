"""Unit tests for lazy op recording (Idea 1A)."""

from rpython.jit.metainterp import lazy_record as lr


def _make_recorder():
    return lr.LazyRecorder()


def test_pure_chain_fully_dead():
    # a = int_add(x, 1); b = int_mul(a, 2); c = int_sub(b, 3)
    # Nothing is ever used -> all three stay virtual.
    r = _make_recorder()
    a = r.record('int_add', [1, 1])
    b = r.record('int_mul', [a, 2])
    r.record('int_sub', [b, 3])
    total, kept = r.count()
    assert total == 3
    assert kept == 0
    sink = lr.MaterializeSink()
    r.materialize(sink)
    assert sink.ops == []


def test_side_effect_cascades_to_operands():
    # setfield(obj, int_add(x, 1)) -> both escape.
    r = _make_recorder()
    a = r.record('int_add', [1, 1])
    obj = 'OBJ'  # opaque const token
    r.record('setfield_gc', [obj, a])
    _, kept = r.count()
    assert kept == 2
    sink = lr.MaterializeSink()
    r.materialize(sink)
    assert len(sink.ops) == 2
    assert sink.ops[0][1] == 'int_add'
    assert sink.ops[1][1] == 'setfield_gc'


def test_guard_forces_resume_vars():
    # x = int_add(a, 1); guard_true(cond) with live_vars=[x] -> x escapes
    r = _make_recorder()
    x = r.record('int_add', [1, 1])
    cond = r.record('int_lt', [x, 10])
    r.record_guard('guard_true', [cond], live_vars=[x])
    _, kept = r.count()
    # cond forced via guard args cascade; x forced via live_vars; guard itself.
    assert kept == 3


def test_loop_tail_forces_livevars():
    r = _make_recorder()
    x = r.record('int_add', [1, 1])
    y = r.record('int_mul', [x, 2])
    r.record_loop_tail([y])
    _, kept = r.count()
    # y escapes (tail); x cascades because y references it.
    assert kept == 2


def test_reduction_ratio_matches_hand_count():
    r = _make_recorder()
    # simulate Bolz's 11200 recorded / 22 kept: mostly pure, a handful escape
    survivors = []
    for i in range(100):
        a = r.record('int_add', [i, 1])
        if i % 50 == 0:
            survivors.append(a)
    r.record_loop_tail(survivors)
    total, kept = r.count()
    assert total == 100
    # Two survivors at i=0, 50.
    assert kept == 2
    assert abs(r.reduction_ratio() - 0.98) < 1e-9


def test_materialize_preserves_program_order_and_arg_refs():
    r = _make_recorder()
    a = r.record('int_add', [1, 1])
    b = r.record('int_mul', [a, 2])
    r.record('setfield_gc', ['OBJ', b])
    sink = lr.MaterializeSink()
    r.materialize(sink)
    assert [op[1] for op in sink.ops] == ['int_add', 'int_mul', 'setfield_gc']
    # Arg of setfield must point at b's emitted handle (second emit).
    setf = sink.ops[2]
    mul = sink.ops[1]
    assert setf[2][1] is mul


def test_call_may_force_is_side_effect():
    r = _make_recorder()
    x = r.record('int_add', [1, 1])
    r.record('call_may_force_n', [x])
    _, kept = r.count()
    assert kept == 2


def test_virtualop_repr_sane():
    r = _make_recorder()
    v = r.record('int_add', [1, 1])
    s = repr(v)
    assert 'V1' in s and 'int_add' in s
