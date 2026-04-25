"""Unit + integration tests for super-jitcode fused dispatch (Idea 1)."""

import json

import pytest

from rpython.jit.metainterp import super_jitcode
from rpython.jit.metainterp import dispatch_profile


class FakeStaticData(object):
    """Just enough of a MetaInterpStaticData surface for super_jitcode to
    install and drive a super-op table on."""

    def __init__(self, opcode_names, implementations):
        assert len(opcode_names) == len(implementations)
        self.opcode_names = list(opcode_names)
        self.opcode_implementations = list(implementations)
        self.super_op_table = None
        self.super_op_N = 0


def _make_impl(log, tag):
    def impl(self, pc):
        log.append(tag)
        self.pc = pc + 1
    impl.__name__ = 'impl_' + tag
    return impl


def test_build_fused_handler_calls_both():
    log = []
    impl_a = _make_impl(log, 'a')
    impl_b = _make_impl(log, 'b')
    fused = super_jitcode.build_fused_handler(impl_a, impl_b)

    class F:
        pc = 0
    frame = F()
    fused(frame, 0)
    assert log == ['a', 'b']
    assert frame.pc == 2  # each impl bumps pc by 1


def test_build_table_stores_op2_opnum():
    impls = [_make_impl([], 'zero'),
             _make_impl([], 'one'),
             _make_impl([], 'two')]
    table = super_jitcode.build_table(3, [(0, 1), (1, 2)], impls)
    assert table[0 * 3 + 1] == 1
    assert table[1 * 3 + 2] == 2
    # Unregistered pair stays -1 (EMPTY_SLOT).
    assert table[2 * 3 + 0] == super_jitcode.EMPTY_SLOT


def test_build_table_skips_oob_pairs():
    impls = [_make_impl([], 'a'), _make_impl([], 'b')]
    # (5, 1) is out of range for a 2-opcode system.
    table = super_jitcode.build_table(2, [(0, 1), (5, 1)], impls)
    assert table[0 * 2 + 1] == 1


def test_install_pairs_and_resolve_names():
    impls = [_make_impl([], 'x'), _make_impl([], 'y')]
    sd = FakeStaticData(['opX', 'opY'], impls)
    installed = super_jitcode.install_pairs(sd, [('opX', 'opY')])
    assert installed == 1
    assert sd.super_op_N == 2
    assert sd.super_op_table[0 * 2 + 1] == 1


def test_install_pairs_drops_unknown_names():
    impls = [_make_impl([], 'x'), _make_impl([], 'y')]
    sd = FakeStaticData(['opX', 'opY'], impls)
    installed = super_jitcode.install_pairs(
        sd, [('opX', 'opY'), ('opX', 'opZ'), ('nope', 'opY')])
    assert installed == 1


def test_build_table_from_profile_roundtrip(tmp_path, monkeypatch):
    # Fresh profiler to avoid cross-test bleed.
    monkeypatch.setattr(dispatch_profile, 'DISPATCH_PROFILE_ENABLED', True)
    monkeypatch.setattr(dispatch_profile.DispatchProfiler, '_instance', None)
    prof = dispatch_profile.DispatchProfiler.get_instance()
    names = ['add', 'lt', 'goto', 'live']
    # Pretend we saw many (add, lt) and a few (lt, goto) pairs.
    for _ in range(50):
        dispatch_profile.record_pair('jc0', 0, 1)
    for _ in range(10):
        dispatch_profile.record_pair('jc0', 1, 2)
    out = tmp_path / 'p.json'
    prof.dump_json(str(out), names)

    # Load via super_jitcode and confirm the installed table reflects both.
    impls = [_make_impl([], n) for n in names]
    sd = FakeStaticData(names, impls)
    n_installed = super_jitcode.build_table_from_profile(sd, str(out),
                                                        top_k=10)
    assert n_installed == 2
    assert sd.super_op_table[0 * 4 + 1] == 1   # add -> lt
    assert sd.super_op_table[1 * 4 + 2] == 2   # lt -> goto


def test_super_op_dispatch_skips_outer_loop_plumbing():
    """Simulate the dispatch-loop fast path end to end: op1 runs, then the
    super-op table dispatches op2 inline. Verify both impls ran exactly
    once and the outer bytecodes_counter was incremented only for op2 in
    the fast path (matching pyjitpl's run_one_step semantics)."""
    log = []

    def impl1(frame, pc):
        log.append(('op1', pc))
        frame.pc = pc + 1

    def impl2(frame, pc):
        log.append(('op2', pc))
        frame.pc = pc + 1

    impls = [impl1, impl2]
    sd = FakeStaticData(['op1', 'op2'], impls)
    super_jitcode.install_pairs(sd, [('op1', 'op2')])

    class FakeFrame(object):
        def __init__(self):
            self.pc = 0
            self.bytecodes_counter = 0

    frame = FakeFrame()
    # Manually drive the fast-path exactly like run_one_step would.
    op1 = 0
    pc = frame.pc
    sd.opcode_implementations[op1](frame, pc)
    pc = frame.pc
    next_op = 1
    slot = sd.super_op_table[op1 * sd.super_op_N + next_op]
    assert slot >= 0  # fusable
    frame.bytecodes_counter += 1
    sd.opcode_implementations[slot](frame, pc)
    assert log == [('op1', 0), ('op2', 1)]
    assert frame.pc == 2
    assert frame.bytecodes_counter == 1
