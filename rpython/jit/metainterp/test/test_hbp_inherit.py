"""Tests for HBP parent-state inheritance - pypy/pypy#5184.

Phase A: guard-fact inheritance.  The bridgeopt encoding gains optional
sections carrying integer bounds and nullness-only PtrInfo facts.  Writes are
gated on warmstate.hbp_inherit >= 1 and enable_hot_bridge_promotion; the
default arm emits no extra bytes.

Smoking-gun pattern from #5184 (verbatim from the issue body):

    +1088: guard_true(i94, .) [...]
    +1104: guard_false(i100, descr=<Guard0x...>) [i0, i94, ...]
    # bridge out of Guard 0x...:
    +222: guard_true(i1, .) [...]            # i1 == i94 redundantly re-guarded

The bridge re-guards i1 because the parent's IntBound on i94 was dropped
across the bridge boundary by bridgeopt.  Section 4 carries it across.
"""

from rpython.rlib.jit import JitDriver, dont_look_inside, promote, set_param
from rpython.jit.metainterp.test.support import LLJitMixin
from rpython.jit.metainterp.optimizeopt.bridgeopt import (
    serialize_optimizer_knowledge, deserialize_optimizer_knowledge)
from rpython.jit.metainterp.optimizeopt.info import (
    InstancePtrInfo, NonNullPtrInfo)
from rpython.jit.metainterp.optimizeopt.intutils import IntBound, MININT, MAXINT
from rpython.jit.metainterp.history import ConstInt
from rpython.jit.metainterp.resoperation import (
    InputArgInt, InputArgRef, ResOperation, rop)
from rpython.jit.metainterp.resume import NumberingState
from rpython.jit.metainterp.resumecode import unpack_numbering
from rpython.jit.metainterp.warmspot import get_stats


# --- unit-level: section 4 round-trip on the bridgeopt wire format ---


class _FakeCPU(object):
    def cls_of_box(self, box):
        return None


class _FakeWarmState(object):
    def __init__(self, hbp_inherit, enable_hot_bridge_promotion=True,
                 hbp_inherit_bool=True):
        self.hbp_inherit = hbp_inherit
        self.hbp_inherit_bool = hbp_inherit_bool
        self.enable_hot_bridge_promotion = enable_hot_bridge_promotion
        self.enable_hbp_value_promotion = True
        self.enable_hbp_ref_value_promotion = True
        self.enable_hbp_float_value_promotion = True
        self.enable_hbp_bool_promotion = False
        self.enable_hbp_guard_bool_promotion = False
        self.enable_hbp_class_promotion = False
        self.hot_bridge_global_max_variants = 0
        self.hbp_global_variant_count = 0
        self.hbp_inherit_max_liveboxes = 0


class _FakeJD(object):
    def __init__(self, hbp_inherit, enable_hot_bridge_promotion=True,
                 hbp_inherit_bool=True):
        self.warmstate = _FakeWarmState(hbp_inherit,
                                        enable_hot_bridge_promotion,
                                        hbp_inherit_bool)


class _FakeOptimizer(object):
    metainterp_sd = None
    optheap = None
    optrewrite = None

    def __init__(self, hbp_inherit=0, enable_hot_bridge_promotion=True,
                 hbp_inherit_bool=True):
        self.cpu = _FakeCPU()
        self.intbounds = {}
        self.classes = {}
        self.nonnulls = set()
        self.jitdriver_sd = _FakeJD(hbp_inherit,
                                    enable_hot_bridge_promotion,
                                    hbp_inherit_bool)

    def make_constant_class(self, box, cls):
        self.classes[box] = cls

    def setintbound(self, box, bound):
        self.intbounds[box] = bound

    def getintbound(self, box):
        fw = box.get_forwarded()
        if isinstance(fw, IntBound):
            return fw
        return IntBound.unbounded()

    def make_nonnull(self, box):
        self.nonnulls.add(box)


class _FakeClass(object):
    pass


class _FakeStorage(object):
    def __init__(self, numb):
        self.rd_numb = numb


def _serialise(opt, bound_specs):
    """Build a NumberingState carrying one InputArgInt per (lower, upper),
    each with its IntBound preset, run the serialiser, return the writer
    and the liveboxes list."""
    liveboxes = []
    for lower, upper in bound_specs:
        box = InputArgInt()
        box.set_forwarded(IntBound(lower=lower, upper=upper))
        liveboxes.append(box)
    numb_state = NumberingState(4)
    numb_state.append_int(1)                          # resume-block sentinel
    serialize_optimizer_knowledge(opt, numb_state, liveboxes, {}, None)
    return numb_state, liveboxes


def test_intbound_section_off_emits_length_zero():
    """hbp_inherit=0: no phase-A sections are emitted, preserving the
    baseline bridgeopt byte stream."""
    numb_state, _ = _serialise(_FakeOptimizer(hbp_inherit=0),
                               [(1, 42), (-5, 5)])
    decoded = unpack_numbering(numb_state.create_numbering())
    assert decoded == [1, 0, 0, 0]


def test_hbp_disabled_does_not_emit_intbound_section():
    """The snapshot is tied to HBP.  Even with hbp_inherit=1, disabling HBP
    leaves the baseline bridgeopt byte stream unchanged."""
    numb_state, _ = _serialise(
        _FakeOptimizer(hbp_inherit=1, enable_hot_bridge_promotion=False),
        [(1, 42)])
    decoded = unpack_numbering(numb_state.create_numbering())
    assert decoded == [1, 0, 0, 0]


def test_intbound_section_round_trip():
    """hbp_inherit=1: an IntBound on an int livebox round-trips through
    serialise -> deserialise."""
    numb_state, _ = _serialise(_FakeOptimizer(hbp_inherit=1), [(1, 42)])
    rbox = InputArgInt()
    after = _FakeOptimizer(hbp_inherit=1)
    deserialize_optimizer_knowledge(
        after, _FakeStorage(numb_state.create_numbering()), [rbox], [rbox],
        use_hbp_inherit=True)
    assert rbox in after.intbounds
    ib = after.intbounds[rbox]
    assert ib.lower == 1
    assert ib.upper == 42


def test_intbound_section_not_restored_for_plain_bridge():
    """Phase-B restore is limited to the promoted-bridge path."""
    numb_state, _ = _serialise(_FakeOptimizer(hbp_inherit=1), [(1, 42)])
    rbox = InputArgInt()
    after = _FakeOptimizer(hbp_inherit=1)
    deserialize_optimizer_knowledge(
        after, _FakeStorage(numb_state.create_numbering()), [rbox], [rbox])
    assert rbox not in after.intbounds


def test_intbound_section_round_trip_multiple_entries():
    """Several IntBounds round-trip in order; entries are matched by
    livebox index."""
    numb_state, _ = _serialise(_FakeOptimizer(hbp_inherit=1),
                               [(0, 1), (-10, 10), (100, 200)])
    rboxes = [InputArgInt(), InputArgInt(), InputArgInt()]
    after = _FakeOptimizer(hbp_inherit=1)
    deserialize_optimizer_knowledge(
        after, _FakeStorage(numb_state.create_numbering()), rboxes, rboxes,
        use_hbp_inherit=True)
    assert (after.intbounds[rboxes[0]].lower,
            after.intbounds[rboxes[0]].upper) == (0, 1)
    assert (after.intbounds[rboxes[1]].lower,
            after.intbounds[rboxes[1]].upper) == (-10, 10)
    assert (after.intbounds[rboxes[2]].lower,
            after.intbounds[rboxes[2]].upper) == (100, 200)


def test_hbp_inherit_livebox_cap_skips_extra_sections():
    """A configured livebox cap can avoid broad parent-state snapshots on
    large guards while preserving the baseline bridgeopt sections."""
    opt = _FakeOptimizer(hbp_inherit=1)
    opt.jitdriver_sd.warmstate.hbp_inherit_max_liveboxes = 1
    numb_state, _ = _serialise(opt, [(1, 42), (2, 43)])
    decoded = unpack_numbering(numb_state.create_numbering())
    assert decoded == [1, 0, 0, 0]


def test_intbound_section_skips_current_guard_argument():
    """Do not inherit success-path facts about the guard that is currently
    failing.  A promoted bridge may still inherit facts about other boxes
    that were established earlier in the parent."""
    opt = _FakeOptimizer(hbp_inherit=1)
    tested = InputArgInt()
    tested.set_forwarded(IntBound(lower=2, upper=2))
    parent_fact = InputArgInt()
    parent_fact.set_forwarded(IntBound(lower=1, upper=255))
    guard_op = ResOperation(rop.GUARD_VALUE, [tested, ConstInt(2)])
    numb_state = NumberingState(4)
    numb_state.append_int(1)
    serialize_optimizer_knowledge(
        opt, numb_state, [tested, parent_fact], {}, None, guard_op)
    r_tested = InputArgInt()
    r_parent_fact = InputArgInt()
    after = _FakeOptimizer(hbp_inherit=1)
    deserialize_optimizer_knowledge(
        after, _FakeStorage(numb_state.create_numbering()),
        [r_tested, r_parent_fact], [r_tested, r_parent_fact],
        use_hbp_inherit=True)
    assert r_tested not in after.intbounds
    assert (after.intbounds[r_parent_fact].lower,
            after.intbounds[r_parent_fact].upper) == (1, 255)


def test_bool_guard_can_skip_hbp_inherit_section():
    """The benchmark profile can keep integer-value inheritance while
    avoiding extra resume-data sections on bool-derived HBP guards."""
    opt = _FakeOptimizer(hbp_inherit=1, hbp_inherit_bool=False)
    opt.jitdriver_sd.warmstate.enable_hbp_bool_promotion = True
    tested = InputArgInt()
    tested.set_forwarded(IntBound(lower=0, upper=1))
    parent_fact = InputArgInt()
    parent_fact.set_forwarded(IntBound(lower=1, upper=42))
    guard_op = ResOperation(rop.GUARD_VALUE, [tested, ConstInt(1)])
    numb_state = NumberingState(4)
    numb_state.append_int(1)
    serialize_optimizer_knowledge(
        opt, numb_state, [tested, parent_fact], {}, None, guard_op)
    decoded = unpack_numbering(numb_state.create_numbering())
    assert decoded == [1, 0, 0, 0]


def test_intbound_section_skips_universe_range():
    """A box whose IntBound is MININT..MAXINT carries no information;
    serialise must skip it so no phase-A section is emitted."""
    opt = _FakeOptimizer(hbp_inherit=1)
    box = InputArgInt()
    box.set_forwarded(IntBound(lower=MININT, upper=MAXINT))
    numb_state = NumberingState(4)
    numb_state.append_int(1)
    serialize_optimizer_knowledge(opt, numb_state, [box], {}, None)
    decoded = unpack_numbering(numb_state.create_numbering())
    assert decoded == [1, 0, 0, 0]


def test_intbound_section_skips_out_of_short_range():
    """HBP intbound inheritance keeps resume data compact by storing only
    signed-short sized bounds; wider ranges are rediscovered in the bridge."""
    numb_state, _ = _serialise(_FakeOptimizer(hbp_inherit=1),
                               [(-50000, 50000)])
    rbox = InputArgInt()
    after = _FakeOptimizer(hbp_inherit=1)
    deserialize_optimizer_knowledge(
        after, _FakeStorage(numb_state.create_numbering()), [rbox], [rbox],
        use_hbp_inherit=True)
    assert rbox not in after.intbounds


# --- nullness (section 5) round-trip --------------------------------------


def _serialise_refs(opt, boxes):
    """Build a NumberingState carrying the given ref boxes (with whatever
    PtrInfo they already have forwarded), run the serialiser."""
    numb_state = NumberingState(4)
    numb_state.append_int(1)
    serialize_optimizer_knowledge(opt, numb_state, list(boxes), {}, None)
    return numb_state


def test_nullness_section_off_emits_length_zero():
    """hbp_inherit=0: no nullness section is emitted."""
    opt = _FakeOptimizer(hbp_inherit=0)
    box = InputArgRef()
    box.set_forwarded(NonNullPtrInfo())
    numb_state = _serialise_refs(opt, [box])
    decoded = unpack_numbering(numb_state.create_numbering())
    assert decoded == [1, 0, 0, 0, 0]


def test_nullness_section_round_trip():
    """hbp_inherit=1: a NonNullPtrInfo-only ref box round-trips through
    section 5 and lands on the bridge optimizer via make_nonnull."""
    opt = _FakeOptimizer(hbp_inherit=1)
    box = InputArgRef()
    box.set_forwarded(NonNullPtrInfo())
    numb_state = _serialise_refs(opt, [box])
    rbox = InputArgRef()
    after = _FakeOptimizer(hbp_inherit=1)
    deserialize_optimizer_knowledge(
        after, _FakeStorage(numb_state.create_numbering()), [rbox], [rbox],
        use_hbp_inherit=True)
    assert rbox in after.nonnulls


def test_nullness_section_skipped_when_class_known():
    """When section 1 already captures a known class for a ref box
    (which implies non-null), phase A must NOT duplicate the bit."""
    cls = _FakeClass()
    opt = _FakeOptimizer(hbp_inherit=1)
    box = InputArgRef()
    box.set_forwarded(InstancePtrInfo(known_class=cls))
    numb_state = _serialise_refs(opt, [box])
    decoded = unpack_numbering(numb_state.create_numbering())
    assert decoded == [1, 0b100000, 0, 0, 0]


def test_nullness_section_off_does_not_install_on_bridge():
    """hbp_inherit=0: even with a non-null parent livebox, the bridge
    optimizer must not have make_nonnull called."""
    opt = _FakeOptimizer(hbp_inherit=0)
    box = InputArgRef()
    box.set_forwarded(NonNullPtrInfo())
    numb_state = _serialise_refs(opt, [box])
    rbox = InputArgRef()
    after = _FakeOptimizer(hbp_inherit=0)
    deserialize_optimizer_knowledge(
        after, _FakeStorage(numb_state.create_numbering()), [rbox], [rbox],
        use_hbp_inherit=True)
    assert rbox not in after.nonnulls


def test_nullness_section_mixed_with_intbound():
    """Phase B should not regress phase A: a trace mixing int and ref
    liveboxes round-trips both sections in one pass."""
    opt = _FakeOptimizer(hbp_inherit=1)
    int_box = InputArgInt()
    int_box.set_forwarded(IntBound(lower=0, upper=99))
    ref_box = InputArgRef()
    ref_box.set_forwarded(NonNullPtrInfo())
    numb_state = NumberingState(4)
    numb_state.append_int(1)
    serialize_optimizer_knowledge(
        opt, numb_state, [int_box, ref_box], {}, None)
    r_int = InputArgInt()
    r_ref = InputArgRef()
    after = _FakeOptimizer(hbp_inherit=1)
    deserialize_optimizer_knowledge(
        after, _FakeStorage(numb_state.create_numbering()),
        [r_int, r_ref], [r_int, r_ref], use_hbp_inherit=True)
    assert r_int in after.intbounds
    assert (after.intbounds[r_int].lower,
            after.intbounds[r_int].upper) == (0, 99)
    assert r_ref in after.nonnulls


def test_intbound_section_off_does_not_install_on_bridge():
    """hbp_inherit=0 must not deserialise any IntBound on the bridge
    side, even if the parent had narrowed-bound boxes."""
    numb_state, _ = _serialise(_FakeOptimizer(hbp_inherit=0), [(1, 42)])
    rbox = InputArgInt()
    after = _FakeOptimizer(hbp_inherit=0)
    deserialize_optimizer_knowledge(
        after, _FakeStorage(numb_state.create_numbering()), [rbox], [rbox],
        use_hbp_inherit=True)
    assert rbox not in after.intbounds


# --- integration: smoke test for HBP with inheritance enabled ---


class TestHBPInherit(LLJitMixin):
    """End-to-end: meta_interp must not crash with HBP and hbp_inherit
    both on, and the answer must match the un-jitted reference."""

    def test_hbp_inherit_smoke(self):
        myjitdriver = JitDriver(greens=[], reds=['i', 'N', 's'])

        def run(N):
            # Same HBP-enabling preamble as test_hbp_cardinality_gate_*.
            set_param(None, 'threshold', 3)
            set_param(None, 'trace_eagerness', 1)
            set_param(None, 'retrace_limit', 5)
            set_param(None, 'enable_hot_bridge_promotion', 1)
            set_param(None, 'hot_bridge_threshold', 1)
            set_param(None, 'hot_bridge_guard_threshold', 1)
            set_param(None, 'hbp_inherit', 1)
            s = 0
            i = 0
            while i < N:
                myjitdriver.jit_merge_point(N=N, i=i, s=s)
                # Two consecutive guards on related conditions, the second
                # of which fails ~50% of times the first passes - the
                # #5184 shape, scaled to fit a JIT-test timing budget.
                x = i & 0xff
                if x > 0:
                    if x < 0x80:
                        s += x
                    else:
                        s += 0x80
                i += 1
            return s

        def ref(N):
            s = 0
            i = 0
            while i < N:
                x = i & 0xff
                if x > 0:
                    if x < 0x80:
                        s += x
                    else:
                        s += 0x80
                i += 1
            return s

        res = self.meta_interp(run, [400])
        assert res == ref(400)

    def test_5184_bridge_inherits_intbound_guard_count(self):
        """Trace shape from pypy/pypy#5184: the parent proves x > 0 before a
        hot promoted guard; the promoted bridge must not re-prove x != 0."""
        myjitdriver = JitDriver(greens=[], reds=['i', 'N', 's'])

        @dont_look_inside
        def opaque(v):
            return v

        def run(N, inherit):
            set_param(None, 'threshold', 3)
            set_param(None, 'trace_eagerness', 1)
            set_param(None, 'retrace_limit', 5)
            set_param(None, 'enable_hot_bridge_promotion', 1)
            set_param(None, 'hot_bridge_threshold', 0)
            set_param(None, 'hot_bridge_guard_threshold', 1)
            set_param(None, 'hbp_inherit', inherit)
            s = 0
            i = 0
            while i < N:
                myjitdriver.jit_merge_point(N=N, i=i, s=s)
                x = i & 0xff
                if x > 0:
                    s = opaque(s)
                    tag = promote(x & 3)
                    if tag:
                        if x > 0:
                            s += 256 // x
                        else:
                            s -= 99
                    else:
                        s += x
                i += 1
            return s

        def ref(N):
            s = 0
            i = 0
            while i < N:
                x = i & 0xff
                if x > 0:
                    tag = x & 3
                    if tag:
                        if x > 0:
                            s += 256 // x
                        else:
                            s -= 99
                    else:
                        s += x
                i += 1
            return s

        def count_int_gt():
            count = 0
            for loop in get_stats().get_all_loops():
                for op in loop._all_operations():
                    if op.getopname() == 'int_gt':
                        count += 1
            return count

        res = self.meta_interp(run, [400, 0])
        assert res == ref(400)
        int_gt_without = count_int_gt()

        res = self.meta_interp(run, [400, 1])
        assert res == ref(400)
        int_gt_with = count_int_gt()

        assert int_gt_with < int_gt_without

    def test_hbp_inherit_off_smoke(self):
        """Mirror of test_hbp_inherit_smoke with hbp_inherit=0 - confirms
        the default arm still works (no regression)."""
        myjitdriver = JitDriver(greens=[], reds=['i', 'N', 's'])

        def run(N):
            set_param(None, 'threshold', 3)
            set_param(None, 'trace_eagerness', 1)
            set_param(None, 'retrace_limit', 5)
            set_param(None, 'enable_hot_bridge_promotion', 1)
            set_param(None, 'hot_bridge_threshold', 1)
            set_param(None, 'hot_bridge_guard_threshold', 1)
            set_param(None, 'hbp_inherit', 0)
            s = 0
            i = 0
            while i < N:
                myjitdriver.jit_merge_point(N=N, i=i, s=s)
                x = i & 0xff
                if x > 0:
                    if x < 0x80:
                        s += x
                    else:
                        s += 0x80
                i += 1
            return s

        def ref(N):
            s = 0
            i = 0
            while i < N:
                x = i & 0xff
                if x > 0:
                    if x < 0x80:
                        s += x
                    else:
                        s += 0x80
                i += 1
            return s

        res = self.meta_interp(run, [400])
        assert res == ref(400)
