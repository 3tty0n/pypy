"""Unit tests for snapshot delta encoding (Idea 1D)."""

from rpython.jit.metainterp import snapshot_delta as sd


def test_full_roundtrip():
    store = sd.SnapshotStore()
    s = store.emit_full([1, 2, 3, 4])
    assert store.resolve(s) == [1, 2, 3, 4]
    assert s.kind == sd.Snapshot.KIND_FULL


def test_single_slot_delta():
    store = sd.SnapshotStore()
    s0 = store.emit_full([10, 20, 30, 40, 50, 60, 70, 80])
    s1 = store.emit_delta(s0, [10, 20, 30, 40, 99, 60, 70, 80])
    assert s1.kind == sd.Snapshot.KIND_DELTA
    assert s1.changes == [(4, 99)]
    assert store.resolve(s1) == [10, 20, 30, 40, 99, 60, 70, 80]


def test_delta_chain_resolves_correctly():
    store = sd.SnapshotStore(chain_bound=4)
    base = store.emit_full([0, 0, 0, 0, 0, 0, 0, 0])
    s1 = store.emit_delta(base, [0, 1, 0, 0, 0, 0, 0, 0])
    s2 = store.emit_delta(s1, [0, 1, 2, 0, 0, 0, 0, 0])
    s3 = store.emit_delta(s2, [0, 1, 2, 3, 0, 0, 0, 0])
    assert store.resolve(s3) == [0, 1, 2, 3, 0, 0, 0, 0]
    assert s3.kind == sd.Snapshot.KIND_DELTA


def test_falls_back_to_full_when_half_changed():
    store = sd.SnapshotStore()
    base = store.emit_full([0, 0, 0, 0])  # size 4
    # Changing 3 slots: delta cost >= full cost threshold.
    s = store.emit_delta(base, [1, 2, 3, 0])
    assert s.kind == sd.Snapshot.KIND_FULL


def test_chain_bound_forces_full():
    store = sd.SnapshotStore(chain_bound=2)
    base = store.emit_full([0, 0, 0, 0, 0, 0])
    s1 = store.emit_delta(base, [9, 0, 0, 0, 0, 0])   # delta (depth 1)
    s2 = store.emit_delta(s1,   [9, 8, 0, 0, 0, 0])   # delta (depth 2)
    # Next one would exceed bound -> must flip to FULL.
    s3 = store.emit_delta(s2, [9, 8, 7, 0, 0, 0])
    assert s3.kind == sd.Snapshot.KIND_FULL
    assert store.resolve(s3) == [9, 8, 7, 0, 0, 0]


def test_size_report_shows_savings():
    store = sd.SnapshotStore()
    # 20 snapshots, only 1 slot out of 20 changes each time.
    cur = list(range(20))
    prev = store.emit_full(cur)
    for i in range(19):
        cur = list(cur)
        cur[0] = cur[0] + 1
        prev = store.emit_delta(prev, cur)
    total, all_full, savings = store.size_report()
    # Delta path should be substantially smaller.
    assert total < all_full
    assert savings > 0.5


def test_resolves_unchanged_deep_chain():
    store = sd.SnapshotStore(chain_bound=12)
    base = store.emit_full([7] * 16)
    cur = base
    v = [7] * 16
    for i in range(10):
        v = list(v)
        v[i] = 100 + i
        cur = store.emit_delta(cur, v)
    assert store.resolve(cur) == v
