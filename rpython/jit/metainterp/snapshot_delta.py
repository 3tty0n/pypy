"""
Snapshot delta encoding for resume data (Idea 1D).

Motivation. Bolz & Fijalkowski 2015 already share snapshots across
adjacent side-effect-free guards (sparse snapshots, a la LuaJIT). The
remaining redundancy: two adjacent snapshots that *do* differ typically
differ in only one or two variables out of dozens. Storing the full
vector both times is waste.

Representation. A snapshot is a tuple of frame-slot -> box references.
A delta-snapshot is:

    Snapshot(id=N, base_id=M, changes=[(idx, new_ref), ...])

Resolving a delta: take base snapshot id M, apply changes. Chains are
allowed; bounded by MAX_CHAIN_LEN so deopt walk is predictable.

This module is self-contained so it can be unit-tested and
micro-benchmarked. The metainterp side would invoke ``encode`` at guard
emission and ``decode`` at deopt time.
"""

from __future__ import print_function


MAX_CHAIN_LEN = 16  # cap on how deep a delta chain can go


class Snapshot(object):
    """A snapshot is either:

        * kind=FULL : full vector of values
        * kind=DELTA: (base_id, [(index, new_value), ...])

    Stored in a ``SnapshotStore`` (by id) so a deopt walker can resolve
    bases by lookup.
    """
    KIND_FULL  = 0
    KIND_DELTA = 1

    __slots__ = ('id', 'kind', 'values', 'base_id', 'changes', 'size')

    def __init__(self, sid, kind, values=None, base_id=-1, changes=None,
                 size=0):
        self.id = sid
        self.kind = kind
        self.values = values  # list for FULL, else None
        self.base_id = base_id
        self.changes = changes or []  # for DELTA
        self.size = size

    def __repr__(self):
        if self.kind == Snapshot.KIND_FULL:
            return '<Snap#%d FULL n=%d>' % (self.id, len(self.values or []))
        return ('<Snap#%d DELTA base=%d changes=%d>' %
                (self.id, self.base_id, len(self.changes)))


class SnapshotStore(object):
    """Id-keyed pool of snapshots. Encapsulates the encode/decode logic
    and the chain-length bound.
    """

    def __init__(self, chain_bound=MAX_CHAIN_LEN):
        self._by_id = {}
        self._next_id = 1
        self.chain_bound = chain_bound
        # Accounting.
        self.bytes_full = 0
        self.bytes_delta = 0

    def _alloc_id(self):
        i = self._next_id
        self._next_id += 1
        return i

    # --- emit paths -------------------------------------------------------

    def emit_full(self, values):
        s = Snapshot(self._alloc_id(), Snapshot.KIND_FULL,
                     values=list(values), size=len(values))
        self._by_id[s.id] = s
        # Rough wire cost: 8 bytes per value.
        self.bytes_full += 8 * len(values)
        return s

    def emit_delta(self, base, values):
        """Given the previous snapshot (either FULL or DELTA) and the
        desired new full vector, encode just the changed slots. Falls
        back to FULL if chain would exceed ``chain_bound`` OR if the
        delta would be bigger than a full snapshot.
        """
        depth = self._chain_depth(base)
        if depth + 1 > self.chain_bound:
            return self.emit_full(values)
        base_full = self._resolve(base)
        assert len(base_full) == len(values), \
            'snapshot size changed between guards'
        changes = []
        for i, (old, new) in enumerate(zip(base_full, values)):
            if old is not new and old != new:
                changes.append((i, new))
        # If more than ~half the vector changed, a full snapshot is
        # cheaper to decode and no bigger to write.
        if len(changes) * 2 >= len(values):
            return self.emit_full(values)
        s = Snapshot(self._alloc_id(), Snapshot.KIND_DELTA,
                     base_id=base.id, changes=changes, size=len(values))
        self._by_id[s.id] = s
        # Rough wire cost: 12 bytes per change (index + value).
        self.bytes_delta += 12 * len(changes)
        return s

    # --- decode / resolve -------------------------------------------------

    def resolve(self, snap):
        return self._resolve(snap)

    def _resolve(self, snap):
        if snap.kind == Snapshot.KIND_FULL:
            return list(snap.values)
        chain = []
        cur = snap
        depth = 0
        while cur.kind == Snapshot.KIND_DELTA:
            chain.append(cur)
            cur = self._by_id[cur.base_id]
            depth += 1
            if depth > self.chain_bound:
                raise RuntimeError('delta chain exceeded %d' % self.chain_bound)
        out = list(cur.values)
        for delta in reversed(chain):
            for idx, val in delta.changes:
                out[idx] = val
        return out

    def _chain_depth(self, snap):
        d = 0
        cur = snap
        while cur.kind == Snapshot.KIND_DELTA:
            cur = self._by_id[cur.base_id]
            d += 1
        return d

    # --- reporting --------------------------------------------------------

    def size_report(self):
        total = self.bytes_full + self.bytes_delta
        if total == 0:
            return (0, 0, 1.0)
        saved_vs_allfull = 0
        # Approximate "what would allFull cost?" by summing each
        # snapshot's vector size * 8.
        all_full = 0
        for s in self._by_id.values():
            all_full += 8 * s.size
        return (total, all_full,
                1.0 - (total / float(all_full)) if all_full else 0.0)
