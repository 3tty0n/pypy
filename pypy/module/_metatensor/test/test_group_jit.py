"""What the meta-tracer makes of the leave rule, checked on the JIT itself.

The same core methods the app-level Group and Aggregate use, driven from an
RPython loop so the traces can be inspected: one step is two workers'
gradients, two membership checks and a commit.  A leave is armed for step K,
after the loop is hot, so it arrives in compiled code.
"""

import os
os.environ.setdefault('RTENSOR_CPU', '1')

from rpython.jit.metainterp.test.support import LLJitMixin
from rpython.rlib.jit import JitDriver
from rpython.metatensor import core, kernels, nn, ops, runtime
from pypy.module._metatensor.interp_group import W_Group

kernels.init_device()

STEPS = 40
K = 30
XA = [1.0, -2.0, 3.0, 0.5]
XB = [2.0, 1.0, -1.0, 4.0]


def reference(dropped_at, revoke):
    w = [0.0] * 4
    for step in range(STEPS):
        s = float(1 + step % 4)
        ga = [max(x * s, 0.0) + x for x in XA]
        gb = [max(x * s, 0.0) + x for x in XB]
        b_in = (dropped_at == 'never' or step < K or
                (step == K and dropped_at == 'after' and not revoke))
        g = [a + b for a, b in zip(ga, gb)] if b_in else ga
        w = [wi - 0.125 * gi for wi, gi in zip(w, g)]
    return sum([wi * (i + 1) for i, wi in enumerate(w)])


driver = JitDriver(greens=[], reds=['step', 'w', 'xa', 'xb', 'g'])


def program(revoke, when, reason):
    g = W_Group(['A', 'B'])
    g.declare('preempt', 'retain')
    if revoke:
        g.declare('fault', 'revoke')
    if when > 0:
        g.arm('B', reason, 3 * K + when)
    xa = nn.Tensor(core.from_list(XA))
    xb = nn.Tensor(core.from_list(XB))
    w = nn.Tensor(core.from_list([0.0] * 4))
    step = 0
    while step < STEPS:
        driver.jit_merge_point(step=step, w=w, xa=xa, xb=xb, g=g)
        s = nn.Tensor(runtime.scalar_of(float(1 + step % 4), core.F64))
        agg = g.aggregate()
        agg.sync()
        agg.contribute('A', xa.mul(s).relu().add(xa))
        agg.sync()
        agg.contribute('B', xb.mul(s).relu().add(xb))
        w = g.commit(agg, w, 0.125)
        step += 1
    return ops.item(ops.sum(ops.mul(w.t, core.from_list([1.0, 2.0, 3.0,
                                                         4.0])), -1))


def structural_guard_virtuals():
    """For each guard_not_invalidated in the loop body, the virtual objects
    its resume data would rebuild, counted by struct name."""
    from rpython.jit.metainterp.test.support import get_stats
    from rpython.jit.metainterp import resume
    ops = get_stats().get_all_loops()[0].operations
    start = [k for k, op in enumerate(ops) if op.getopname() == 'label'][-1]
    out = []
    for op in ops[start:]:
        if op.getopname() != 'guard_not_invalidated':
            continue
        d = op.getdescr()
        while not hasattr(d, 'rd_virtuals'):
            d = d.prev
        kinds = {}
        for v in d.rd_virtuals or []:
            if v is None:
                continue
            name = type(v).__name__
            if isinstance(v, resume.AbstractVirtualStructInfo):
                name = str(v.fielddescrs[0]).split('GcStruct ')[-1]
                name = name.split(' ')[0].split('.')[-1]
            kinds[name] = kinds.get(name, 0) + 1
        out.append(kinds)
    return out


class TestLeaveJit(LLJitMixin):

    def run(self, revoke, when, reason):
        def f(revoke_i, when_i, fault_i):
            return program(revoke_i == 1, when_i,
                           'fault' if fault_i == 1 else 'preempt')
        return self.meta_interp(f, [int(revoke), when,
                                    int(reason == 'fault')])

    def test_steady_state_is_one_kernel_and_allocates_nothing(self):
        # With parts retained for a possible revoke, a step is still one
        # launch - w - lr*(g_A + g_B), both gradients inside it - and the
        # aggregate, its parts and the tensors in them are never allocated.
        # The other two call_r are the scalar lookups for s and lr; the
        # three call_n are the membership polls, each followed by the
        # structural guard (guard_not_invalidated; one more at loop entry).
        res = self.run(True, 0, 'fault')
        assert res == reference('never', False)
        self.check_simple_loop(new_with_vtable=0, call_r=3, call_n=3,
                               guard_not_invalidated=4)

    def test_retention_costs_nothing_in_the_steady_state(self):
        # Retain-only policy: nothing is kept, and the loop is the same.
        res = self.run(False, 0, 'fault')
        assert res == reference('never', False)
        self.check_simple_loop(new_with_vtable=0, call_r=3, call_n=3,
                               guard_not_invalidated=4)

    def test_h1_preempt_in_compiled_code(self):
        res = self.run(True, 3, 'preempt')
        assert res == reference('after', False)

    def test_h2_fault_in_compiled_code_revokes(self):
        res = self.run(True, 3, 'fault')
        assert res == reference('after', True)

    def test_h3_leave_before_contribution(self):
        res = self.run(True, 2, 'fault')
        assert res == reference('before', True)

    def test_resume_data_holds_parts_only_while_the_rule_can_read_them(self):
        # The four structural guards of a step: loop entry, before A's
        # contribution, before B's, and in the commit.  Under a policy with
        # a revoke the commit's guard can rebuild both contributors' parts;
        # the fused total is there either way, without who contributed what.
        self.run(True, 0, 'fault')
        kept = structural_guard_virtuals()
        self.run(False, 0, 'fault')
        none = structural_guard_virtuals()
        assert [k.get('Part', 0) for k in kept] == [0, 0, 1, 2]
        assert [k.get('Part', 0) for k in none] == [0, 0, 0, 0]
        assert kept[3].get('VTensorInfo') == none[3].get('VTensorInfo') > 0
        # nothing survives the commit: the next step's first guard holds w
        assert kept[0] == none[0] == {'Tensor': 1}


    def mutant(self, name, value):
        old = os.environ.get(name)
        os.environ[name] = value
        try:
            return self.run(True, 3, 'fault')
        finally:
            if old is None:
                del os.environ[name]
            else:
                os.environ[name] = old

    def test_erasure_mutant_commits_the_revoked_contribution(self):
        # No parts kept: the revoke has nothing to leave out, and step K
        # commits g_A + g_B as if B had been preempted.
        res = self.mutant('MOTION_ERASE', '1')
        assert res != reference('after', True)
        assert res == reference('after', False)

    def test_order_mutant_commits_before_revoking(self):
        res = self.mutant('MOTION_ORDER', 'late')
        assert res != reference('after', True)
        assert res == reference('after', False)
