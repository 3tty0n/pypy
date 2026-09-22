"""Membership, uncommitted aggregates and the leave event, around the rule.

Participants contribute tensors to an aggregate that a commit applies to the
model.  A participant can leave while the aggregate is uncommitted, and the
application declares per reason whether its received contributions are
retained or revoked.  What a leave *means* is the change rule,
rule_leave.leave, written against motion_iface alone.  This module is the
interpreter around it: it holds the reference semantics, and it is the only
hand-written code MOTION needs besides the rule and the application's policy
declaration.

What the meta-tracer makes of it, with nothing written for the purpose:

  structural guard   membership (names, version) is quasi-immutable.  A
                     trace specialises on it, a leave notice bumps the
                     version, and the compiled code fails at
                     guard_not_invalidated.
  retention          contribute() keeps a per-contributor part only when
                     `retains` is set.  The parts live in the aggregate,
                     which never escapes a step, so in a trace they are
                     virtual: fusion folds them into the commit's kernel and
                     they exist only as entries in the resume data of the
                     guards inside the step.  A commit drops them.
  ordered transition the guard sits in the sync that begins a commit, so the
                     resumed interpreter applies the rule and then commits.

`retains` itself is derived, not declared: declare() runs the rule on a
recording stand-in for each declared reason and keeps parts exactly when some
reason makes the rule read them.  With RETAIN everywhere nothing is kept.

Revocation rebuilds the committed value from the surviving parts in
contribution order rather than subtracting the revoked one, so a revoked step
is bitwise the step that never saw the contribution.

Two mutants, for the checker to reject, are switched on by environment
variables read when a group is created: MOTION_ERASE=1 keeps no parts
whatever the rule reads, MOTION_ORDER=late takes the value before the sync.
"""

import os

from rpython.rlib import jit
from pypy.interpreter.error import oefmt
from pypy.interpreter.gateway import interp2app, unwrap_spec
from pypy.interpreter.typedef import TypeDef
from rpython.metatensor import nn, ops, runtime
from pypy.module._metatensor.motion_iface import (Membership, Uncommitted,
                                                  RETAIN, REVOKE)
from pypy.module._metatensor import rule_leave
from pypy.module._metatensor.interp_tensor import W_Tensor


def _env_set(name):
    v = os.environ.get(name)
    return v is not None and v != '' and v != '0'


def _without(names, p):
    """A fixed-size copy of names without p: the membership list is
    quasi-immutable, so it is replaced, never resized."""
    k = 0
    for n in names:
        if n != p:
            k += 1
    out = [''] * k
    j = 0
    for n in names:
        if n != p:
            out[j] = n
            j += 1
    return out


class Part(object):
    _immutable_fields_ = ['p', 't', 'next']

    def __init__(self, p, t, next):
        self.p = p
        self.t = t
        self.next = next


class W_Group(Membership):
    _immutable_fields_ = ['names?[*]', 'version?', 'retains?', 'erase',
                          'late']

    def __init__(self, names):
        self.names = _without(names, None)
        self.version = 0
        self.retains = False
        self.reasons = []
        self.actions = []
        self.armed_at = []
        self.armed_p = []
        self.armed_reason = []
        self.polls = 0
        self.notices_p = []
        self.notices_reason = []
        self.log = []
        self.commits = 0
        self.erase = _env_set('MOTION_ERASE')
        self.late = os.environ.get('MOTION_ORDER') == 'late'

    # --- the rule's vocabulary ---------------------------------------------

    def drop_member(self, p):
        names = _without(self.names, p)
        if len(names) != len(self.names):
            self.names = names
            self.version += 1

    def action(self, reason):
        for i in range(len(self.reasons)):
            if self.reasons[i] == reason:
                return self.actions[i]
        return RETAIN

    # --- membership ---------------------------------------------------------

    @jit.unroll_safe
    def is_member(self, p):
        for n in self.names:
            if n == p:
                return True
        return False

    @jit.dont_look_inside
    def poll(self):
        """Deliver the leave notices due at this point.  In a deployment the
        runtime does this when a device drops out; here a schedule armed by
        the program stands in for it.  A delivery bumps the version, which
        invalidates every trace specialised on the membership."""
        self.polls += 1
        for i in range(len(self.armed_at)):
            if self.armed_at[i] == self.polls:
                self.notices_p.append(self.armed_p[i])
                self.notices_reason.append(self.armed_reason[i])
                self.version += 1

    def _derive_retention(self):
        live = False
        for r in self.reasons:
            probe = _ReadSet()
            rule_leave.leave(_PolicyOnly(self), [probe], '?', r)
            if probe.read:
                live = True
        self.retains = live and not self.erase

    # --- app level ----------------------------------------------------------

    def declare(self, reason, action):
        self.reasons.append(reason)
        self.actions.append(action)
        self._derive_retention()

    @unwrap_spec(reason='text', action='text')
    def descr_declare(self, space, reason, action):
        if action != RETAIN and action != REVOKE:
            raise oefmt(space.w_ValueError, "action must be retain or revoke")
        self.declare(reason, action)

    def arm(self, p, reason, at):
        self.armed_at.append(at)
        self.armed_p.append(p)
        self.armed_reason.append(reason)

    @unwrap_spec(p='text', reason='text', at=int)
    def descr_arm(self, space, p, reason, at):
        self.arm(p, reason, at)

    def aggregate(self):
        return W_Aggregate(self)

    def descr_aggregate(self, space):
        return self.aggregate()

    @unwrap_spec(p='text')
    def descr_is_member(self, space, p):
        return space.newbool(self.is_member(p))

    def commit(self, agg, w, lr):
        """w - lr * the aggregate's value, after the membership check."""
        if self.late:
            value = agg.value()
            agg.sync()
        else:
            agg.sync()
            value = agg.value()
        agg.parts = None
        agg.total = None
        agg.committed = True
        self.commits += 1
        if value is None:
            return w
        step = nn.Tensor(runtime.scalar_of(jit.promote(lr),
                                           ops.tensor_dtype(w.t)))
        return w.sub(value.mul(step))

    @unwrap_spec(lr=float)
    def descr_commit(self, space, w_agg, w_w, lr):
        agg = space.interp_w(W_Aggregate, w_agg)
        if agg.committed:
            raise oefmt(space.w_ValueError, "aggregate already committed")
        w = space.interp_w(W_Tensor, w_w).tensor
        return W_Tensor(self.commit(agg, w, lr))

    def descr_retains(self, space):
        return space.newbool(self.retains)

    def descr_commits(self, space):
        return space.newint(self.commits)

    def descr_log(self, space):
        return space.newlist([space.newtext(s) for s in self.log])


class _PolicyOnly(Membership):
    """The group as the rule sees it during derivation: the declared policy,
    and no effect on membership."""

    def __init__(self, group):
        self.group = group

    def drop_member(self, p):
        pass

    def action(self, reason):
        return self.group.action(reason)


class _ReadSet(Uncommitted):
    """Records whether the rule reads a leaving participant's uncommitted
    contributions."""

    def __init__(self):
        self.read = False

    def exclude(self, p):
        self.read = True


class W_Aggregate(Uncommitted):
    def __init__(self, group):
        # Quasi-immutable fields are only guarded on a constant object; on
        # any other the read is taken as pure and reused across the poll.
        group = jit.promote(group)
        self.group = group
        self.seen = group.version
        self.parts = None
        self.total = None
        self.excluded = False
        self.committed = False

    def exclude(self, p):
        kept = []
        removed = False
        cur = self.parts
        while cur is not None:
            if cur.p == p:
                removed = True
            else:
                kept.append(cur)
            cur = cur.next
        if not removed:
            return
        parts = None
        for i in range(len(kept) - 1, -1, -1):
            parts = Part(kept[i].p, kept[i].t, parts)
        self.parts = parts
        self.excluded = True
        self.group.log.append('revoke %s' % p)

    def sync(self):
        """A membership check point.  Steady state: one residual call and a
        guard; the comparison below folds away."""
        g = jit.promote(self.group)
        g.poll()
        if g.version != self.seen:
            self._apply_notices()

    def _apply_notices(self):
        g = self.group
        ps = g.notices_p
        reasons = g.notices_reason
        g.notices_p = []
        g.notices_reason = []
        for i in range(len(ps)):
            g.log.append('leave %s %s' % (ps[i], reasons[i]))
            rule_leave.leave(g, [self], ps[i], reasons[i])
        self.seen = g.version

    def value(self):
        """What a commit applies: the running total, or, once a contribution
        has been revoked, the surviving parts summed in contribution order."""
        if not self.excluded:
            return self.total
        return self._rebuild()

    def _rebuild(self):
        # A loop, so the tracer calls it instead of inlining it; it is only
        # reached once a contribution has been revoked, never in the steady
        # state, where value() above must stay loop-free or the aggregate
        # escapes into the call and every contribution is materialised.
        order = []
        cur = self.parts
        while cur is not None:
            order.append(cur.t)
            cur = cur.next
        acc = None
        for i in range(len(order) - 1, -1, -1):
            if acc is None:
                acc = order[i]
            else:
                acc = acc.add(order[i])
        return acc

    def contribute(self, p, t):
        g = jit.promote(self.group)
        if not g.is_member(p):
            return False
        if g.retains:
            self.parts = Part(p, t, self.parts)
        if self.total is None:
            self.total = t
        else:
            self.total = self.total.add(t)
        return True

    @unwrap_spec(p='text')
    def descr_contribute(self, space, p, w_t):
        if self.committed:
            raise oefmt(space.w_ValueError, "aggregate already committed")
        t = space.interp_w(W_Tensor, w_t).tensor
        return space.newbool(self.contribute(p, t))

    def descr_sync(self, space):
        self.sync()


def new_group(space, w_names):
    names = [space.text_w(w) for w in space.listview(w_names)]
    return W_Group(names)


W_Group.typedef = TypeDef(
    '_metatensor.Group',
    declare=interp2app(W_Group.descr_declare),
    arm=interp2app(W_Group.descr_arm),
    aggregate=interp2app(W_Group.descr_aggregate),
    is_member=interp2app(W_Group.descr_is_member),
    commit=interp2app(W_Group.descr_commit),
    retains=interp2app(W_Group.descr_retains),
    commits=interp2app(W_Group.descr_commits),
    log=interp2app(W_Group.descr_log),
)

W_Aggregate.typedef = TypeDef(
    '_metatensor.Aggregate',
    contribute=interp2app(W_Aggregate.descr_contribute),
    sync=interp2app(W_Aggregate.descr_sync),
)
