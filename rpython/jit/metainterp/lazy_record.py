"""
Lazy op recording (Idea 1A).

Generalization of GenExtension. The baseline meta-interpreter emits a
ResOperation into the trace for *every* op it runs; the optimizer later
deletes 99%+ of them (Bolz 2025-06: 11200 recorded -> 22 kept per 1-iter
Python microbench). Recording + re-scanning the to-be-deleted ops is pure
overhead.

This module buffers ops as lightweight ``VirtualOp`` records and
materializes them into ``ResOperation`` only when one of the escape
conditions fires:

    1. side-effect op (setfield, raw store, call with side effect)
    2. referenced from a guard's resume data
    3. still live at the loop tail (becomes a jump argument)
    4. referenced by the args of a non-virtual op

Theoretical position. Bolz 2011 PEPM allocation-removal operates on
*object* allocations at the language level. Here we lift exactly the same
forward/backward-pass idea to *trace-op* allocations at the meta level.
GenExtension's heapcache-invalidation-skip is a special case of escape
condition (1) in this framework.

Prototype scope. This file gives:
  * ``VirtualOp`` / ``LazyRecorder`` / ``MaterializeSink`` core classes
  * a set of escape predicates wired up for the common op families
  * ``materialize`` that rewrites all live VirtualOps into ResOperations
    in program order and returns a cleaned-up trace

It is deliberately independent of the rest of the metainterp so it can
be unit-tested in isolation and benchmarked against direct recording.
"""

from __future__ import print_function


# --- escape reasons (debugging) --------------------------------------------

ESC_SIDE_EFFECT    = 'side_effect'
ESC_GUARD_RESUME   = 'guard_resume'
ESC_LOOP_TAIL      = 'loop_tail'
ESC_NONVIRT_USE    = 'nonvirt_use'
ESC_USER_FORCE     = 'user_force'


# --- op metadata -----------------------------------------------------------

# Side-effectful op names. Any op in this set is always materialized,
# its operands are force-escaped, and all in-flight virtual sibling ops
# sharing a containing heapcache region are too. The list matches the
# coarse side-effect groups used elsewhere in the meta-interpreter
# (setfield, setarrayitem, raw store, externally-visible calls). Keep
# this list small: adding an op here costs us materializations, missing
# one breaks correctness.
SIDE_EFFECT_OPS = frozenset([
    'setfield_gc', 'setfield_raw',
    'setarrayitem_gc', 'setarrayitem_raw',
    'strsetitem', 'unicodesetitem',
    'raw_store',
    'call_may_force_n', 'call_assembler_n',
    'guard_not_forced',  # materializes everything before it
])


# Pure ops can live entirely as VirtualOp; the optimizer can fold them
# away later (or not at all, if they escape).
PURE_OP_HINT = frozenset([
    'int_add', 'int_sub', 'int_mul', 'int_and', 'int_or', 'int_xor',
    'int_lt', 'int_le', 'int_gt', 'int_ge', 'int_eq', 'int_ne',
    'int_is_true', 'int_is_zero',
    'float_add', 'float_sub', 'float_mul', 'float_neg',
    'getfield_gc_pure',  # read-only field
    'same_as', 'cast_int_to_float', 'cast_float_to_int',
])


# --- data model ------------------------------------------------------------

class VirtualOp(object):
    """Lightweight stand-in for a ResOperation. Holds just enough to
    decide materialization later: opname, arg references, result box,
    and a side-effect/pure flag. No optimizer-facing payload is filled
    until materialize() runs.

    ``args`` is a list of either ``VirtualOp`` (back-references by
    identity, like SSA edges) or opaque "const tokens" -- whatever the
    caller wants to represent an immediate constant. We only look at
    object identity for escape analysis.
    """
    __slots__ = ('id', 'opname', 'args', 'has_result',
                 'escaped', 'escape_reason',
                 'materialized_op')

    def __init__(self, vid, opname, args, has_result):
        self.id = vid
        self.opname = opname
        self.args = args
        self.has_result = has_result
        self.escaped = False
        self.escape_reason = None
        self.materialized_op = None  # filled by materialize()

    def __repr__(self):
        star = '*' if self.escaped else ''
        return '<V%d %s%s args=%d>' % (self.id, self.opname, star,
                                        len(self.args))


# --- recorder --------------------------------------------------------------

class LazyRecorder(object):
    """Buffers VirtualOps until a materialization event. API is
    intentionally narrow so the real metainterp can swap this in for its
    ``history.record`` call.
    """

    def __init__(self):
        self._ops = []
        self._next_id = 1

    def record(self, opname, args, has_result=True):
        vop = VirtualOp(self._next_id, opname, list(args), has_result)
        self._next_id += 1
        self._ops.append(vop)
        if opname in SIDE_EFFECT_OPS:
            self._force(vop, ESC_SIDE_EFFECT, cascade=True)
        return vop

    def record_guard(self, opname, args, live_vars):
        """Record a guard. ``live_vars`` are the resume-data references
        and must materialize for the guard to be able to deopt. The
        guard's own ``args`` (typically the condition variable) must
        also survive -- the guard can't evaluate without them.
        """
        gop = VirtualOp(self._next_id, opname, list(args), has_result=False)
        self._next_id += 1
        self._ops.append(gop)
        self._force(gop, ESC_GUARD_RESUME, cascade=True)
        for v in live_vars:
            if isinstance(v, VirtualOp):
                self._force(v, ESC_GUARD_RESUME, cascade=True)
        return gop

    def record_loop_tail(self, live_vars):
        """Mark the jump args at the loop tail as escaped."""
        for v in live_vars:
            if isinstance(v, VirtualOp):
                self._force(v, ESC_LOOP_TAIL, cascade=True)

    def _force(self, vop, reason, cascade):
        if vop.escaped:
            return
        vop.escaped = True
        vop.escape_reason = reason
        if cascade:
            # Any operand referenced by an escaped op must itself survive.
            for a in vop.args:
                if isinstance(a, VirtualOp):
                    self._force(a, ESC_NONVIRT_USE, cascade=True)

    # --- stats / introspection --------------------------------------------

    def count(self):
        return len(self._ops), sum(1 for v in self._ops if v.escaped)

    def reduction_ratio(self):
        n = len(self._ops)
        if n == 0:
            return 1.0
        kept = sum(1 for v in self._ops if v.escaped)
        return 1.0 - (kept / float(n))

    # --- materialization --------------------------------------------------

    def materialize(self, emit):
        """Walk the buffered ops in program order, invoking ``emit`` for
        every escaped op. Returns the sequence of emitted op handles
        (whatever ``emit`` returned).

        ``emit(opname, resolved_args) -> op_handle`` is the caller's
        bridge to the real trace writer. For tests/benches we hand in a
        list.append closure; for the real metainterp it would wrap
        ``history.record``.
        """
        handles = {}
        out = []
        for v in self._ops:
            if not v.escaped:
                continue
            real_args = []
            for a in v.args:
                if isinstance(a, VirtualOp):
                    h = handles.get(a.id)
                    if h is None:
                        # Operand escaped but not yet emitted -- means
                        # program order was broken. Should not happen
                        # because we walk self._ops in order.
                        raise RuntimeError(
                            'virtual op %s referenced before emit' % a)
                    real_args.append(h)
                else:
                    real_args.append(a)
            h = emit(v.opname, real_args)
            handles[v.id] = h
            v.materialized_op = h
            out.append(h)
        return out


# --- simple in-memory sink for tests/benches ------------------------------

class MaterializeSink(object):
    """Callable sink that stores (opname, args) tuples in a list. Used
    as ``emit`` for tests and microbenchmarks.
    """
    def __init__(self):
        self.ops = []

    def __call__(self, opname, args):
        h = ('op#%d' % len(self.ops), opname, tuple(args))
        self.ops.append(h)
        return h
