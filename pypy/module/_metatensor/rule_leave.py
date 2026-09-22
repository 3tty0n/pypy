"""Change rule for a participant leaving, for a given reason.

p no longer takes part.  If the application declared REVOKE for this
reason, p's uncommitted contributions are left out of every pending
aggregate; if it declared RETAIN, uncommitted contributions are untouched.
"""

from pypy.module._metatensor.motion_iface import REVOKE


def leave(group, pending, p, reason):
    group.drop_member(p)
    action = group.action(reason)
    if action == REVOKE:
        for uncommitted in pending:
            uncommitted.exclude(p)
