import os
os.environ.setdefault('RTENSOR_CPU', '1')


class AppTestGroup(object):
    spaceconfig = dict(usemodules=['_metatensor'])

    def setup_class(cls):
        # One step: sync before A (poll 3k+1), sync before B (3k+2), and the
        # sync that begins the commit (3k+3).
        cls.w_run = cls.space.appexec([], """():
            def run(policy, event, steps=6, k=3):
                import _metatensor as mt
                g = mt.group(['A', 'B'])
                for reason, action in policy:
                    g.declare(reason, action)
                if event is not None:
                    who, reason, when = event
                    g.arm(who, reason, 3 * k + when)
                xa = mt.tensor([1.0, -2.0, 3.0, 0.5])
                xb = mt.tensor([2.0, 1.0, -1.0, 4.0])
                w = mt.tensor([0.0, 0.0, 0.0, 0.0])
                for step in range(steps):
                    s = mt.scalar(float(1 + step % 4), 'float64')
                    agg = g.aggregate()
                    agg.sync()
                    agg.contribute('A', (xa * s).relu() + xa)
                    agg.sync()
                    agg.contribute('B', (xb * s).relu() + xb)
                    w = g.commit(agg, w, 0.125)
                return w.tolist(), g.commits(), g.log(), g.retains()
            return run
        """)
        # The reference, by hand on plain floats: B's gradient is in every
        # step before k, and in step k only if it arrived and was retained.
        cls.w_expected = cls.space.appexec([], """():
            def expected(dropped_at, revoke, steps=6, k=3):
                xa = [1.0, -2.0, 3.0, 0.5]
                xb = [2.0, 1.0, -1.0, 4.0]
                w = [0.0] * 4
                for step in range(steps):
                    s = float(1 + step % 4)
                    ga = [max(x * s, 0.0) + x for x in xa]
                    gb = [max(x * s, 0.0) + x for x in xb]
                    b_in = step < k or (step == k and dropped_at == 'after'
                                        and not revoke)
                    g = [a + b for a, b in zip(ga, gb)] if b_in else ga
                    w = [wi - 0.125 * gi for wi, gi in zip(w, g)]
                return w
            return expected
        """)

    def test_policy_derives_retention(self):
        _, _, _, kept = self.run([('preempt', 'retain')], None)
        assert kept is False
        _, _, _, kept = self.run([('preempt', 'retain'), ('fault', 'revoke')],
                                 None)
        assert kept is True

    def test_h1_preempt_after_contribution_keeps_it(self):
        w, commits, log, _ = self.run(
            [('preempt', 'retain'), ('fault', 'revoke')], ('B', 'preempt', 3))
        assert w == self.expected('after', False)
        assert commits == 6
        assert log == ['leave B preempt']

    def test_h2_fault_after_contribution_revokes_it(self):
        w, commits, log, _ = self.run(
            [('preempt', 'retain'), ('fault', 'revoke')], ('B', 'fault', 3))
        assert w == self.expected('after', True)
        assert commits == 6
        assert log == ['leave B fault', 'revoke B']

    def test_h3_leave_before_contribution_has_nothing_to_revoke(self):
        w, commits, log, _ = self.run(
            [('preempt', 'retain'), ('fault', 'revoke')], ('B', 'fault', 2))
        assert w == self.expected('before', True)
        assert log == ['leave B fault']
