"""Negative control for the leave rule's audit: the same rule, written by
someone who knows how the aggregate is represented and how recovery works.
audit_rules.py must reject it; it is never imported."""

from pypy.module._metatensor.motion_iface import REVOKE
from pypy.module._metatensor.interp_group import W_Aggregate
from rpython.jit.metainterp import resume


def leave(group, pending, p, reason):
    group.drop_member(p)
    group.version += 1
    if group.action(reason) == REVOKE:
        for agg in pending:
            cur = agg.parts
            while cur is not None:
                if cur.p == p:
                    agg.total = agg.total.sub(cur.t)
                cur = cur.next
            agg.pending.append(resume.snapshot)
