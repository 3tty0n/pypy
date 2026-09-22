"""The vocabulary a change rule may use, and nothing else.

A change rule states what an event means for the program: which premise it
invalidates, and what the application's declared policy does to work that
has not been committed yet.  It is written against these two abstract
classes.  They name membership, policy and uncommitted contributions; they
name no kernel, no fused region, no guard and no resume data, so a rule that
imports only this module cannot know how the state it talks about is
represented once the JIT has specialised it.

benchmark/paper/audit_rules.py checks the rule file against that promise.
"""

from pypy.interpreter.baseobjspace import W_Root

RETAIN = 'retain'
REVOKE = 'revoke'


class Membership(W_Root):
    """Who takes part, and what the application declared for each way of
    leaving."""

    def drop_member(self, p):
        """p no longer takes part; anything that assumed it did is void."""
        raise NotImplementedError

    def action(self, reason):
        """The declared action for a leave with this reason: RETAIN or
        REVOKE."""
        raise NotImplementedError


class Uncommitted(W_Root):
    """Contributions received but not committed yet."""

    def exclude(self, p):
        """Leave p's contributions out of what will be committed."""
        raise NotImplementedError
