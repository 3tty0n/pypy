"""Unit tests for the dispatch-pair profiler (Idea 1)."""

import json
import os

import pytest

from rpython.jit.metainterp import dispatch_profile


def _fresh(monkeypatch):
    """Return a fresh, enabled profiler; restore state afterwards."""
    monkeypatch.setattr(dispatch_profile, 'DISPATCH_PROFILE_ENABLED', True)
    monkeypatch.setattr(dispatch_profile.DispatchProfiler, '_instance', None)
    return dispatch_profile.DispatchProfiler.get_instance()


def test_disabled_is_noop(monkeypatch):
    # When the env flag is off, record_pair must not populate anything.
    monkeypatch.setattr(dispatch_profile, 'DISPATCH_PROFILE_ENABLED', False)
    monkeypatch.setattr(dispatch_profile.DispatchProfiler, '_instance', None)
    dispatch_profile.record_pair('jc', 0, 1)
    # Touching the getter still returns an instance, but counts stay empty.
    prof = dispatch_profile.DispatchProfiler.get_instance()
    assert prof.pair_counts == {}


def test_record_pair_counts(monkeypatch):
    prof = _fresh(monkeypatch)
    for _ in range(3):
        dispatch_profile.record_pair('jc', 1, 2)
    dispatch_profile.record_pair('jc', 1, 3)
    assert prof.pair_counts[('jc', 1, 2)] == 3
    assert prof.pair_counts[('jc', 1, 3)] == 1
    assert prof.total_pair_hits() == 4


def test_record_entry_counts(monkeypatch):
    prof = _fresh(monkeypatch)
    dispatch_profile.record_entry('jc')
    dispatch_profile.record_entry('jc')
    dispatch_profile.record_entry('other')
    assert prof.jitcode_entries == {'jc': 2, 'other': 1}


def test_top_pairs_resolves_names(monkeypatch):
    prof = _fresh(monkeypatch)
    names = ['op_zero', 'op_one', 'op_two', 'op_three']
    for _ in range(5):
        dispatch_profile.record_pair('jc', 1, 2)
    dispatch_profile.record_pair('jc', 0, 3)
    top = prof.top_pairs(10, opcode_names=names)
    # Sorted by frequency desc.
    assert top[0] == ('jc', 'op_one', 'op_two', 5)
    assert top[1] == ('jc', 'op_zero', 'op_three', 1)


def test_dump_json_roundtrip(tmp_path, monkeypatch):
    prof = _fresh(monkeypatch)
    names = ['a', 'b', 'c']
    for _ in range(2):
        dispatch_profile.record_pair('jc', 0, 1)
    dispatch_profile.record_pair('jc', 1, 2)
    out = tmp_path / 'profile.json'
    prof.dump_json(str(out), names)
    payload = json.loads(out.read_text())
    assert payload['total_pair_hits'] == 3
    assert payload['opcode_names'] == names
    pairs = sorted(payload['pairs'], key=lambda p: -p['count'])
    assert pairs[0]['count'] == 2
    assert pairs[0]['prev'] == 0 and pairs[0]['cur'] == 1


def test_load_hot_pairs_from_file(tmp_path, monkeypatch):
    prof = _fresh(monkeypatch)
    names = ['a', 'b', 'c']
    dispatch_profile.record_pair('jc', 0, 1)
    out = tmp_path / 'profile.json'
    prof.dump_json(str(out), names)
    loaded = dispatch_profile.load_hot_pairs_from_file(str(out))
    assert ('jc', 'a', 'b') in loaded


def test_register_and_lookup_hot_pair():
    dispatch_profile.clear_hot_pairs()
    assert dispatch_profile.lookup_hot_pair('op_a', 'op_b') is None
    marker = object()
    dispatch_profile.register_hot_pair('op_a', 'op_b', marker)
    assert dispatch_profile.lookup_hot_pair('op_a', 'op_b') is marker
    dispatch_profile.clear_hot_pairs()
    assert dispatch_profile.lookup_hot_pair('op_a', 'op_b') is None


def test_triple_counts_from_sliding_window(monkeypatch):
    # (add, lt, guard_true) is the canonical hot triple.
    monkeypatch.setattr(dispatch_profile, 'DISPATCH_PROFILE_ENABLED', True)
    monkeypatch.setattr(dispatch_profile.DispatchProfiler, '_instance', None)
    prof = dispatch_profile.DispatchProfiler.get_instance()
    dispatch_profile.record_entry('jc')
    # Feed pairs (0,1), (1,2) as the window slides: that's triple (0,1,2).
    dispatch_profile.record_pair('jc', 0, 1)
    dispatch_profile.record_pair('jc', 1, 2)
    # Next window: (2,0), (0,1) -> triple (2,0,1).
    dispatch_profile.record_pair('jc', 2, 0)
    dispatch_profile.record_pair('jc', 0, 1)
    assert prof.triple_counts[('jc', 0, 1, 2)] == 1
    assert prof.triple_counts[('jc', 2, 0, 1)] == 1
    assert prof.total_triple_hits() == 3  # also (1,2,0)


def test_triple_window_resets_on_entry(monkeypatch):
    monkeypatch.setattr(dispatch_profile, 'DISPATCH_PROFILE_ENABLED', True)
    monkeypatch.setattr(dispatch_profile.DispatchProfiler, '_instance', None)
    prof = dispatch_profile.DispatchProfiler.get_instance()
    dispatch_profile.record_entry('jc')
    dispatch_profile.record_pair('jc', 0, 1)
    dispatch_profile.record_entry('jc')  # new frame: window cleared
    dispatch_profile.record_pair('jc', 1, 2)
    # No triple should have formed across the frame boundary.
    assert prof.total_triple_hits() == 0
