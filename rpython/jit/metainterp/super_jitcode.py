"""
Super-jitcode fused handler generator (Idea 1, Phase B).

Takes a list of hot (op1, op2) pairs observed by the dispatch_profile module
and produces fused handlers that execute both opimpls without re-running the
outer dispatch loop plumbing (bytecodes_counter, opcode_counters,
op_live/op_goto compares, bytecode[pc] re-read).

This is the meta-level analogue of Piumarta & Riccardi's selective inlining:
we pick the hottest adjacent pairs of jitcode operations and collapse them
into a single dispatch step.

The existing full-jitcode AOT specializer (genextension.py) replaces the
*entire* dispatch loop for a jitcode. Super-jitcodes are strictly lighter: a
per-pair fast path that catches the common cases when the full specializer
hasn't fired (new jitcode at runtime, failed specialization, etc.).

Table representation (translate-safe)
-------------------------------------
The table is a plain `list[int]` of length `N * N` where N is the total
opcode count. A slot holds:
    -1  if (op1, op2) is not a fused pair
    op2 if (op1, op2) IS a fused pair -- the dispatch loop then calls
        `staticdata.opcode_implementations[op2]` exactly the same way it
        would for the normal slow path, minus the outer-loop plumbing.

Storing an int (instead of a function pointer) keeps the list homogeneous
for the RPython translator, and avoids needing `@rgc.must_be_light_finalizer`
or other GC hints. The fast-path cost is unchanged: one array load + a
`>= 0` check + the same `opcode_implementations[op]` lookup the dispatcher
already performs.

Activation sequence:
  1. Run once with PYPY_DISPATCH_PROFILE=1 PYPY_DISPATCH_PROFILE_OUT=p.json
  2. On next run: build_table_from_profile(staticdata, 'p.json')
     which fills staticdata.super_op_table.
  3. The dispatch loop in run_one_step peeks the table after each op.
"""

from rpython.jit.metainterp import dispatch_profile


EMPTY_SLOT = -1


def build_fused_handler(op1_impl, op2_impl):
    """Return a Python-level fused handler for (op1, op2).

    Runs both impls back-to-back and relies on the opimpl contract that
    each impl updates `self.pc` past its own encoding. Exceptions
    (ChangeFrame, SwitchToBlackhole, ...) propagate naturally -- the
    fused handler does not catch them, so op2 is skipped if op1 already
    transferred control.

    Kept for standalone use (tests, notebooks). The translated dispatch
    loop does not go through this -- it uses the int table built below.
    """
    def fused(self, pc):
        op1_impl(self, pc)
        op2_impl(self, self.pc)
    fused.__name__ = 'fused_' + op1_impl.__name__ + '_' + op2_impl.__name__
    return fused


def empty_table(num_opcodes):
    """Allocate a new, all-empty table of the right shape."""
    return [EMPTY_SLOT] * (num_opcodes * num_opcodes)


def build_table(num_opcodes, pairs_by_opnum, opcode_implementations=None):
    """Produce a flat `op1*N + op2 -> op2 (int)` lookup table.

    `opcode_implementations` is accepted for backwards compatibility with
    the old (function-pointer) table; it's used only to drop pairs whose
    impl is `None`.
    """
    table = empty_table(num_opcodes)
    for (op1, op2) in pairs_by_opnum:
        if not (0 <= op1 < num_opcodes and 0 <= op2 < num_opcodes):
            continue
        if opcode_implementations is not None:
            if (opcode_implementations[op1] is None or
                    opcode_implementations[op2] is None):
                continue
        table[op1 * num_opcodes + op2] = op2
    return table


def _install(staticdata, num_opcodes, pairs):
    """Shared installer. Always sets super_op_N, always leaves super_op_table
    as a list (possibly all -1), so the dispatch loop's `if table:` is a
    pure length check and the hot read is unconditional.
    """
    table = empty_table(num_opcodes)
    installed = 0
    for op1, op2 in pairs:
        if not (0 <= op1 < num_opcodes and 0 <= op2 < num_opcodes):
            continue
        impls = staticdata.opcode_implementations
        if impls[op1] is None or impls[op2] is None:
            continue
        table[op1 * num_opcodes + op2] = op2
        installed += 1
    staticdata.super_op_table = table
    staticdata.super_op_N = num_opcodes
    return installed


def build_table_from_profile(staticdata, profile_path, top_k=32):
    """Load a profile JSON emitted by dispatch_profile.dump_json and install
    the resulting super-op table on `staticdata.super_op_table`.

    Returns the number of fused pairs actually installed.
    """
    hot = dispatch_profile.load_hot_pairs_from_file(profile_path)
    name_to_opnum = {}
    for op, name in enumerate(staticdata.opcode_names):
        name_to_opnum[name] = op
    pairs = []
    for (_jitcode, a_name, b_name) in hot:
        a = name_to_opnum.get(a_name, -1)
        b = name_to_opnum.get(b_name, -1)
        if a >= 0 and b >= 0:
            pairs.append((a, b))
        if len(pairs) >= top_k:
            break
    N = len(staticdata.opcode_implementations)
    return _install(staticdata, N, pairs)


# --- programmatic installation (used by tests / harnesses) -----------------

def install_pairs(staticdata, pairs):
    """Install a list of (op1_name, op2_name) pairs directly, bypassing the
    profile JSON. Handy from tests and notebooks.
    """
    name_to_opnum = {n: i for i, n in enumerate(staticdata.opcode_names)}
    opnum_pairs = []
    for a, b in pairs:
        if a in name_to_opnum and b in name_to_opnum:
            opnum_pairs.append((name_to_opnum[a], name_to_opnum[b]))
    N = len(staticdata.opcode_implementations)
    return _install(staticdata, N, opnum_pairs)


def ensure_initialized(staticdata):
    """Make sure `super_op_table` / `super_op_N` exist on the staticdata.
    Called from pyjitpl.setup_insns so the dispatch fast path can read them
    unconditionally without attribute checks.
    """
    if getattr(staticdata, 'super_op_table', None) is None:
        staticdata.super_op_table = []
    if getattr(staticdata, 'super_op_N', 0) == 0:
        staticdata.super_op_N = 0
