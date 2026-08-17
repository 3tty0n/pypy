"""Differential test: compute_liveness_native's new worklist algorithm
must produce byte-relevant-identical output to the old "rerun full list
until no label grows" algorithm it replaced.

Keeps a self-contained copy of the OLD algorithm (not reusing anything
from native_pipeline.py's fixpoint code) so a future edit to the new
algorithm can't silently drag this comparison along with it.
"""

import random

from rpython.translator.backendopt.native_fragments import (
    NReg, NIntConst, NTLabel, NLabel, NDescr, NativeInsn)
from rpython.translator.backendopt.native_pipeline import (
    compute_liveness_native, _converge_liveness_native,
    NativeSwitchDictDescr, _remove_repeated_live_native)


# ____________________________________________________________
# Self-contained copy of the algorithm being replaced (pre-worklist port
# of codewriter/liveness.py): rerun one full backward pass over the
# WHOLE insn list, rewriting every '-live-' insn every pass, until no
# label2alive set grows anymore.

def _old_follow_label(label_id, label2alive, alive):
    alive_at_point = label2alive.get(label_id)
    if alive_at_point is not None:
        for nid, reg in alive_at_point.items():
            alive[nid] = reg


def _old_mark(x, label2alive, alive):
    if isinstance(x, NReg):
        alive[x.nid] = x
    elif isinstance(x, NTLabel):
        _old_follow_label(x.label_id, label2alive, alive)
    elif isinstance(x, NDescr):
        descr = x.descr
        if isinstance(descr, NativeSwitchDictDescr):
            for _key, label in descr._native_labels:
                _old_follow_label(label, label2alive, alive)


def _old_pass(insns, label2alive):
    alive = {}
    must_continue = False
    for i in range(len(insns) - 1, -1, -1):
        insn = insns[i]
        if insn.opcode == "@label":
            label_id = insn.operands[0].label_id
            alive_at_point = label2alive.get(label_id)
            if alive_at_point is None:
                alive_at_point = {}
                label2alive[label_id] = alive_at_point
            prevlength = len(alive_at_point)
            for nid, reg in alive.items():
                alive_at_point[nid] = reg
            if prevlength != len(alive_at_point):
                must_continue = True
            continue
        if insn.opcode == "-live-":
            labels = []
            for x in insn.operands:
                if isinstance(x, NReg):
                    alive[x.nid] = x
                elif isinstance(x, NTLabel):
                    _old_follow_label(x.label_id, label2alive, alive)
                    labels.append(x)
            insns[i] = NativeInsn("-live-", alive.values() + labels)
            continue
        if insn.opcode == "---":
            alive = {}
            continue
        if insn.result is not None and insn.result.nid in alive:
            del alive[insn.result.nid]
        for x in insn.operands:
            _old_mark(x, label2alive, alive)
    return must_continue


def old_compute_liveness_native(insns):
    """Returns the number of full-list passes it took to converge."""
    label2alive = {}
    passes = 0
    while _old_pass(insns, label2alive):
        passes += 1
    passes += 1   # the final, zero-growth pass also ran
    _remove_repeated_live_native(insns)
    return passes


# ____________________________________________________________
# Synthetic-program builder: a tiny DSL so test cases read as segment
# lists instead of hand-written NativeInsn soup.
#
# ops: ('label', name) ('def', regname) ('use', regname)
#      ('live', [('reg', name) | ('label', name), ...])
#      ('goto', name) ('switch', regname, [(key, name), ...])
#      ('return', [regname]?)

def _get_reg(reg_cache, name):
    reg = reg_cache.get(name)
    if reg is None:
        reg = NReg("int", len(reg_cache))
        reg_cache[name] = reg
    return reg


def _get_label(label_cache, name):
    if name not in label_cache:
        label_cache[name] = len(label_cache)
    return label_cache[name]


def _make_insn(op, reg_cache, label_cache):
    kind = op[0]
    if kind == "label":
        return NativeInsn("@label", [NLabel(_get_label(label_cache, op[1]))])
    if kind == "def":
        reg = _get_reg(reg_cache, op[1])
        return NativeInsn("int_copy", [NIntConst(0)], reg)
    if kind == "use":
        reg = _get_reg(reg_cache, op[1])
        return NativeInsn("int_push", [reg])
    if kind == "live":
        operands = []
        for item in op[1]:
            if item[0] == "reg":
                operands.append(_get_reg(reg_cache, item[1]))
            else:
                operands.append(NTLabel(_get_label(label_cache, item[1])))
        return NativeInsn("-live-", operands)
    if kind == "goto":
        return NativeInsn("goto", [NTLabel(_get_label(label_cache, op[1]))])
    if kind == "switch":
        _, regname, targets = op
        reg = _get_reg(reg_cache, regname)
        descr = NativeSwitchDictDescr()
        descr._native_labels = [
            (key, _get_label(label_cache, name)) for key, name in targets]
        return NativeInsn("switch", [reg, NDescr(descr)])
    if kind == "return":
        operands = [_get_reg(reg_cache, op[1])] if len(op) > 1 else []
        return NativeInsn("int_return", operands)
    raise ValueError(op)


def build_program(segments):
    """segments: list of list-of-ops. Every segment gets its own leading
    '---', mirroring emit_native's placement of each block/fragment."""
    reg_cache = {}
    label_cache = {}
    insns = []
    for ops in segments:
        insns.append(NativeInsn("---", []))
        for op in ops:
            insns.append(_make_insn(op, reg_cache, label_cache))
    return insns


# ____________________________________________________________
# Structural comparison: what actually has to match for byte identity
# (see native_pipeline.py's compute_liveness_native docstring) -- (kind,
# index) content of registers and label_id of labels, not object nid,
# which is meaningless across two independently-built programs.

def _operand_key(x):
    if isinstance(x, NReg):
        return ("reg", x.kind, x.index)
    if isinstance(x, NIntConst):
        return ("iconst", x.ivalue)
    if isinstance(x, NTLabel):
        return ("tlabel", x.label_id)
    if isinstance(x, NLabel):
        return ("label", x.label_id)
    if isinstance(x, NDescr):
        descr = x.descr
        if isinstance(descr, NativeSwitchDictDescr):
            return ("switch", tuple(sorted(descr._native_labels)))
        return ("descr", id(descr))
    raise AssertionError(x)


def _insn_key(insn):
    if insn.opcode == "-live-":
        regs = frozenset(_operand_key(x) for x in insn.operands
                         if isinstance(x, NReg))
        labels = frozenset(x.label_id for x in insn.operands
                           if isinstance(x, NTLabel))
        return ("-live-", regs, labels)
    operands = tuple(_operand_key(x) for x in insn.operands)
    result = _operand_key(insn.result) if insn.result is not None else None
    return (insn.opcode, operands, result)


def assert_liveness_equal(insns_a, insns_b):
    assert len(insns_a) == len(insns_b)
    for i, (a, b) in enumerate(zip(insns_a, insns_b)):
        assert _insn_key(a) == _insn_key(b), (
            "insn %d differs: old=%r new=%r" % (i, a, b))


def _check(segments):
    old_insns = build_program(segments)
    new_insns = build_program(segments)
    old_compute_liveness_native(old_insns)
    compute_liveness_native(new_insns)
    assert_liveness_equal(old_insns, new_insns)
    return old_insns, new_insns


# ____________________________________________________________
# Hand-built shapes: diamond, loop (inter- and intra-segment), switch,
# barriers, and a plain straight-line chain.

def test_straight_chain():
    _check([
        [("label", "a"), ("def", "x")],
        [("label", "b"), ("use", "x"), ("goto", "a")],
        [("label", "c"), ("live", [("reg", "x"), ("label", "b")]),
         ("return",)],
    ])


def test_diamond():
    _check([
        [("def", "x"), ("label", "entry"),
         ("switch", "x", [(0, "a"), (1, "b")])],
        [("label", "a"), ("use", "x"), ("goto", "end")],
        [("label", "b"), ("def", "y"), ("use", "y"), ("goto", "end")],
        [("label", "end"), ("live", [("reg", "x"), ("reg", "y")]),
         ("return",)],
    ])


def test_loop_across_segments():
    """Back-edge crossing a '---' boundary -- the classic one-hop-per-old-
    pass scenario."""
    _check([
        [("label", "header"), ("live", [("reg", "acc")]), ("goto", "body")],
        [("label", "body"), ("def", "tmp"), ("use", "tmp"),
         ("goto", "header")],
        [("label", "exit"), ("use", "acc"), ("return",)],
    ])


def test_loop_inside_one_segment():
    """Back-edge with no '---' in between -- one segment must revisit
    itself via the worklist's self-reenqueue path to converge."""
    _check([
        [("label", "L1"), ("use", "a"), ("goto", "L2"),
         ("label", "L2"), ("use", "b"), ("goto", "L1")],
    ])


def test_switch_multi_target():
    _check([
        [("def", "x"), ("def", "y"), ("def", "z"), ("label", "entry"),
         ("switch", "x", [(0, "a"), (1, "b"), (2, "c")])],
        [("label", "a"), ("use", "x"), ("goto", "end")],
        [("label", "b"), ("use", "y"), ("goto", "end")],
        [("label", "c"), ("use", "z"), ("goto", "end")],
        [("label", "end"),
         ("live", [("reg", "x"), ("reg", "y"), ("reg", "z")]),
         ("return",)],
    ])


def test_repeated_barriers_no_flow_across():
    """Segments never leak liveness through '---' by textual adjacency --
    only explicit label edges may."""
    _check([
        [("def", "x"), ("use", "x")],
        [("def", "y"), ("use", "y")],
        [("def", "z"), ("use", "z")],
    ])


def test_forced_live_register_never_used_elsewhere():
    """A '-live-' can force a register alive that nothing else reads."""
    _check([
        [("label", "a"), ("def", "lonely"),
         ("live", [("reg", "lonely")]), ("goto", "b")],
        [("label", "b"), ("return",)],
    ])


def test_diamond_with_nested_loop():
    _check([
        [("def", "x"), ("label", "entry"),
         ("switch", "x", [(0, "loop"), (1, "skip")])],
        [("label", "loop"), ("def", "i"), ("use", "i"), ("use", "x"),
         ("switch", "i", [(0, "loop"), (1, "after")])],
        [("label", "skip"), ("goto", "after")],
        [("label", "after"), ("live", [("reg", "x")]), ("return",)],
    ])


# ____________________________________________________________
# Randomized differential fuzzing: many small programs with branchy/loopy
# label graphs, forward and backward edges, mixed goto/switch/-live-.

def _random_segments(rng, num_segments, num_regs):
    names = ["r%d" % i for i in range(num_regs)]
    labels = ["L%d" % i for i in range(num_segments)]
    segments = []
    for i in range(num_segments):
        ops = [("label", labels[i])]
        for _ in range(rng.randint(0, 3)):
            name = rng.choice(names)
            if rng.random() < 0.5:
                ops.append(("def", name))
            else:
                ops.append(("use", name))
        if rng.random() < 0.3:
            live_operands = []
            for name in rng.sample(names, rng.randint(0, len(names))):
                live_operands.append(("reg", name))
            ops.append(("live", live_operands))
        kind = rng.random()
        if kind < 0.4:
            # goto: may jump forward or backward -- both matter.
            ops.append(("goto", rng.choice(labels)))
        elif kind < 0.7:
            targets = [(k, rng.choice(labels)) for k in range(rng.randint(1, 3))]
            ops.append(("switch", rng.choice(names), targets))
        else:
            ops.append(("return",))
        segments.append(ops)
    return segments


def test_random_differential():
    for seed in range(60):
        rng = random.Random(seed)
        num_segments = rng.randint(2, 12)
        num_regs = rng.randint(1, 5)
        segments = _random_segments(rng, num_segments, num_regs)
        _check(segments)


# ____________________________________________________________
# Scaling sanity check: a chain of ~200 labels, each depending on the
# previous one via a backward (lower-index) reference -- the shape that
# forces the old algorithm to need one full-list pass per hop.

def _make_chain_segments(k):
    # 'live' before any 'def' of the same register: backward-scanning a
    # 'def' kills what a later-in-list (earlier-processed) 'live' just
    # forced alive, so the def must not sit between the label and the
    # live-point or label2alive[L0] would stay empty.
    segments = [[("label", "L0"), ("live", [("reg", "base")])]]
    for i in range(1, k):
        segments.append([
            ("label", "L%d" % i),
            ("goto", "L%d" % (i - 1)),
        ])
    segments.append([
        ("label", "Lend"),
        ("live", [("label", "L%d" % (k - 1))]),
        ("return",),
    ])
    return segments


def test_old_algorithm_passes_scale_with_chain_length():
    """Not a timing assert: counts old full-list passes directly, showing
    they grow with chain depth (the O(B) problem this task fixes)."""
    passes_by_k = {}
    for k in (10, 40, 80):
        segments = _make_chain_segments(k)
        insns = build_program(segments)
        passes = old_compute_liveness_native(insns)
        passes_by_k[k] = passes
    # each extra hop needs (at most) one extra old-style pass
    assert passes_by_k[40] > passes_by_k[10]
    assert passes_by_k[80] > passes_by_k[40]
    # roughly linear in chain length, not O(1)/O(log)
    assert passes_by_k[80] >= passes_by_k[10] + (80 - 10) // 4


def test_new_algorithm_visit_count_grows_linearly_not_quadratically():
    """New algorithm's total segment-visit count (the worklist's real
    inner-loop cost driver) should scale ~linearly with chain length --
    each segment here is O(1)-sized, so linear visits means linear total
    work, unlike the old algorithm's O(passes * N) = O(K^2) blowup."""
    visits_by_k = {}
    for k in (25, 100, 400):
        segments = _make_chain_segments(k)
        insns = build_program(segments)
        label2alive = {}
        visits = _converge_liveness_native(insns, label2alive)
        visits_by_k[k] = visits

    ratio_1 = float(visits_by_k[100]) / visits_by_k[25]
    ratio_2 = float(visits_by_k[400]) / visits_by_k[100]
    # chain length grows 4x each step; a linear-cost algorithm keeps the
    # visit-count ratio near 4x, a quadratic one would land near 16x.
    assert ratio_1 < 8.0, ratio_1
    assert ratio_2 < 8.0, ratio_2


def test_chain_still_byte_identical_to_old():
    """The scaling shape itself must still produce identical output."""
    _check(_make_chain_segments(30))
