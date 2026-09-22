"""The leave transition written by hand, for comparison with the generated one.

Same histories, same checker, same notification channel; no change rule, no
derived retention and no help from the meta-tracer.  What a hand-written
runtime needs to satisfy the three histories:

  a contributor log     every contribution materialised and kept under
                        (step, participant) until its step commits, because a
                        revoke has to be able to take it out again
  deduplication         a contribution that arrives twice for one step (a
                        retry after a transient failure) is counted once
  the transition        on a leave: drop the participant, and if the reason
                        revokes, remove its logged contributions for the
                        uncommitted step; on commit: check for notices first,
                        then apply the sum of what is left in log order

`Channel` is the harness both implementations share (in MOTION it is
Group.arm/poll); it is not counted on either side.
"""


class Channel(object):
    """Delivers leave notices at armed poll counts.  Harness, not adapter."""

    def __init__(self):
        self.armed = []
        self.polls = 0

    def arm(self, p, reason, at):
        self.armed.append((at, p, reason))

    def poll(self):
        self.polls += 1
        return [(p, r) for (at, p, r) in self.armed if at == self.polls]


class ManualAggregator(object):
    def __init__(self, members, policy, channel):
        self.members = list(members)
        self.policy = dict(policy)
        self.channel = channel
        self.step = 0
        self.log = {}
        self.order = []
        self.events = []
        self.commits = 0

    def begin(self, step):
        self.step = step
        self.log = {}
        self.order = []

    def check(self):
        for p, reason in self.channel.poll():
            self.events.append('leave %s %s' % (p, reason))
            if p in self.members:
                self.members.remove(p)
            if self.policy.get(reason, 'retain') == 'revoke':
                keys = [k for k in self.order if k[1] == p]
                for k in keys:
                    del self.log[k]
                    self.order.remove(k)
                if keys:
                    self.events.append('revoke %s' % p)

    def contribute(self, p, g):
        if p not in self.members:
            return False
        key = (self.step, p)
        if key in self.log:
            return True
        self.log[key] = g.force()
        self.order.append(key)
        return True

    def commit(self, w, lr):
        self.check()
        total = None
        for k in self.order:
            total = self.log[k] if total is None else total + self.log[k]
        self.log = {}
        self.order = []
        self.commits += 1
        if total is None:
            return w
        return w - total * lr
