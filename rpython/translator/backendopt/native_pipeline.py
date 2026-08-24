"""Runtime-shaped ports of ProgramEmitter.emit / compute_liveness / Assembler,
operating on the native_fragments.py IR instead of SSARepr/flowspace objects.

Mirrors each original 1:1 on purpose: test_native_pipeline.py's equivalence
gate requires byte-identical output, so keep behavior identical here, not
just "cleaner".
"""

import os

from rpython.jit.codewriter.assembler import (
    Assembler, AssemblerError, USE_C_FORM, int_fits_short)
from rpython.jit.codewriter.flatten import KINDS
from rpython.jit.codewriter.jitcode import JitCode, SwitchDictDescr
from rpython.jit.metainterp.history import AbstractDescr
from rpython.rlib.debug import debug_print, have_debug_prints
from rpython.rlib.rarithmetic import intmask, r_uint
from rpython.rtyper.lltypesystem import llmemory, lltype

from rpython.translator.backendopt.native_fragments import (
    NReg, NIntConst, NRefConst, NFloatConst, NHole, NLabel,
    NTLabel, NDescr, NListOfKind, NIndirectCallTargets, NSwitchDictOperand,
    NativeInsn, native_fragment_for)
from rpython.translator.backendopt.partialeval_template import (
    flatten_resolved_targets, sort_ints, sort_strings, uses_compact_entries)


# ____________________________________________________________
# Global label ids: block boundaries and in-fragment labels must never
# collide once flattened into one program-wide, int-keyed id space.
#
# Block boundaries get negative ids; in-fragment labels get non-negative
# ids built from (placement pc, local label_id).

_LOCAL_LABEL_SPACE = 1 << 20  # headroom: pc * this must fit a 63-bit int

def _block_label_id(pc):
    # intmask: a guest interpreter may carry its pc as an unsigned value,
    # and these ids are signed by construction.
    return -(intmask(pc) + 1)

def _fragment_label_id(pc, local_label_id):
    assert 0 <= local_label_id < _LOCAL_LABEL_SPACE
    return intmask(pc) * _LOCAL_LABEL_SPACE + local_label_id


class NativeSwitchDictDescr(SwitchDictDescr):
    """Must stay a SwitchDictDescr subclass: pyjitpl.py/blackhole.py
    assert isinstance(switchdict, SwitchDictDescr) before reading it.

    Uses ._native_labels, not ._labels: ._labels holds real TLabel
    instances elsewhere, and a plain int label id cannot unify with an
    instance under RPython's type system.
    """

    # Class-level default: an interpreter whose residual code has no switch
    # never constructs one, and the isinstance branches that read this would
    # then look at a classdef with no attributes at all.
    _native_labels = []


class NativeSSARepr(object):
    """Minimal stand-in for SSARepr so Assembler.assemble can run on it."""
    def __init__(self, name):
        self.name = name
        self.insns = []
        self._insns_pos = None


# ____________________________________________________________
# emit_native: port of ProgramEmitter.emit/_place/_localise/_emit_moves.

# List holder: a plain module int would fold to a translation constant.
_prologue_copies = [0]
_boundary_moves = [0]


class _FoldedLoads(object):
    """Holder, not a module int: a prebuilt counter folds to its seed."""
    folded = 0
    # Candidates blocked from folding, by reason (see fold_constant_loads).
    blocked_crossing = 0
    blocked_live = 0
    blocked_argcode = 0
    # The fold costs a fifth of a generation; PYPY_PE_FOLD=0 turns it off
    # at run time so its net effect is measurable from one binary.
    enabled = True
    env_read = False


_folded_loads = _FoldedLoads()


def _fold_enabled():
    if not _folded_loads.env_read:
        _folded_loads.env_read = True
        if os.environ.get("PYPY_PE_FOLD") == "0":
            _folded_loads.enabled = False
    return _folded_loads.enabled



def _log_insn_mix(ssarepr, program):
    """What the emitted program is made of, per opname.

    A residual program is only worth its generation cost if what it adds --
    boundary copies, liveness records, bailout markers -- stays small next to
    the guest work it carries.
    """
    if not have_debug_prints():
        return
    counts = {}
    for insn in ssarepr.insns:
        name = insn.opcode
        counts[name] = counts.get(name, 0) + 1
    debug_print(
        "pe-cogen-mix blocks=%d insns=%d prologue=%d boundary=%d"
        % (len(program.blocks), len(ssarepr.insns), _prologue_copies[0],
           _boundary_moves[0]))
    debug_print("pe-cogen-mix folded-loads=%d" % (_folded_loads.folded,))
    _folded_loads.folded = 0
    # A copy whose source is a constant is not a register move at all: it is
    # a late-static value being materialised, which is what specialising for
    # this code object produced.
    from_const = 0
    targets = {}
    for insn in ssarepr.insns:
        if not insn.opcode.endswith("_copy"):
            continue
        if len(insn.operands) != 1 or insn.result is None:
            continue
        if isinstance(insn.operands[0], NReg):
            continue
        from_const += 1
        if isinstance(insn.result, NReg):
            key = "%s%d" % (insn.result.kind, insn.result.index)
            targets[key] = targets.get(key, 0) + 1
    debug_print("pe-cogen-mix const-loads=%d distinct-targets=%d" % (
        from_const, len(targets)))
    _prologue_copies[0] = 0
    _boundary_moves[0] = 0
    # No sorted(): not RPython, and a histogram needs no order.
    for name, count in counts.items():
        debug_print("pe-cogen-mix   %s %d" % (name, count))


def emit_native(native_table, program, name="emitted-residual-native",
                has_merge_points=False):
    """Concatenate every reached block's fragment into one flat native
    instruction list, as ProgramEmitter.emit does for SSARepr.

    Returns (NativeSSARepr, counts), counts being the per-kind register
    count to pass to NativeAssembler.assemble.
    """
    # Not set()/dict(genexpr)/sorted(): none are RPython-legal here.
    headers = {}
    if has_merge_points:
        for pc in program.loop_headers:
            headers[pc] = True
        headers[program.entry_pc] = True
    compact_entries = uses_compact_entries(program)
    fragments = {}
    for pc, block in program.blocks.items():
        fragments[pc] = native_fragment_for(
            native_table, block.key, pc in headers)

    num_regs = {}
    for kind in KINDS:
        widest = 0
        for fragment in fragments.values():
            widest = max(widest, fragment.num_regs.get(kind, 0))
        num_regs[kind] = widest
    scratch = {}
    for kind, count in num_regs.items():
        scratch[kind] = count
    counts = {}
    for kind in scratch:
        counts[kind] = scratch[kind] + 1

    ssarepr = NativeSSARepr(name)
    order = [program.entry_pc]
    rest = []
    for pc in program.blocks:
        if pc != program.entry_pc:
            rest.append(pc)
    sort_ints(rest)
    order.extend(rest)
    for pc in order:
        ssarepr.insns.append(NativeInsn("---", []))
        ssarepr.insns.append(NativeInsn("@label", [NLabel(_block_label_id(pc))]))
        if has_merge_points and (not compact_entries or pc in headers):
            _initialise_scratch_native(ssarepr, fragments, counts)
        _place_native(ssarepr, program, pc, fragments, scratch)

    return ssarepr, counts


def _register(kind, index):
    return NReg(kind, index)


def _initialise_scratch_native(ssarepr, fragments, counts):
    """Port of ProgramEmitter._initialise_scratch."""
    entry = {}
    for fragment in fragments.values():
        for kind, index in fragment.boundary_entry.values():
            entry[kind] = max(entry.get(kind, 0), index + 1)
    # Not sorted(counts) (not RPython-legal): KINDS is already in the
    # right order, so iterating it directly matches sorted(counts).
    for kind in KINDS:
        if kind not in counts:
            continue
        if kind == "ref":
            const = NRefConst(lltype.nullptr(llmemory.GCREF.TO))
        elif kind == "int":
            const = NIntConst(0)
        else:
            continue
        for index in range(entry.get(kind, 0), counts[kind]):
            ssarepr.insns.append(
                NativeInsn("%s_copy" % kind, [const], _register(kind, index)))


# Measured on PyPy: 148 of the 159 register writes in a 17-block program are
# constant loads, not register moves, and they land in only eight registers --
# each block re-establishes its own late-static values because every block
# boundary is a legal trace entry.  Four passes were tried and all removed
# nothing (region-local, liveness-aware, and global single-assignment copy
# elimination, plus dead-operation removal on the templates), so what looks
# like shuffling is the specialisation's own output.
#
# Every fragment is flattened on its own, so enforce_input_args pins its
# boundary values to the lowest registers of each kind and its body then moves
# them elsewhere.  On PyPy that costs about nine register copies per guest
# bytecode -- half of the real instructions in a residual program -- and the
# metainterp executes each one while tracing.
#
# Measured, none of it is redundant: the copies are neither fragment prologue
# nor boundary moves (both counted, both zero), the templates behind them
# average six operations, and a liveness-aware dead-copy pass removes none of
# them because every copy is read.  Removing them means assigning registers
# over the concatenated program instead of per fragment, not a peephole.
def _place_native(ssarepr, program, pc, fragments, scratch):
    """Port of ProgramEmitter._place."""
    fragment = fragments[pc]
    block = program.blocks[pc]
    for kind, index, bname in fragment.prologue:
        # intmask: a late-static value is a machine-word constant here, and a
        # guest interpreter may hold its pc unsigned.
        const = NIntConst(intmask(block.bindings[bname]))
        _prologue_copies[0] += 1
        ssarepr.insns.append(
            NativeInsn("%s_copy" % kind, [const], _register(kind, index)))
    targets = flatten_resolved_targets(
        block.template.resolve_targets(block.bindings), len(fragment.exits))

    for insn in fragment.insns:
        exit_index = _exit_index(insn)
        # -1, not None: RPython ints have no null representation.
        if exit_index < 0:
            ssarepr.insns.append(_localise_native(
                insn, pc, block.bindings, block.ref_bindings))
            continue
        exit = fragment.exits[exit_index]
        target = targets[exit_index]
        _emit_moves_native(ssarepr, exit.operands,
                           fragments[target].boundary_entry, scratch)
        ssarepr.insns.append(
            NativeInsn("goto", [NTLabel(_block_label_id(target))]))


def _exit_index(insn):
    # insn is pre-localisation here, so an NIntConst can only be a real
    # flowspace Constant, never a runtime-synthesized hole patch.
    if insn.opcode == "int_return" and len(insn.operands) == 1 and \
            isinstance(insn.operands[0], NIntConst):
        return insn.operands[0].ivalue
    return -1


def _localise_native(insn, pc, bindings, ref_bindings):
    """Port of ProgramEmitter._localise/_patch_hole."""
    is_marker = insn.opcode in ("jit_merge_point", "pe_bailout_point")
    operands = [_localise_operand(x, pc, bindings, ref_bindings, is_marker)
               for x in insn.operands]
    result = insn.result   # a register, never a label/hole -- untouched
    if insn.opcode == "@label":
        label_id = insn.operands[0].label_id
        return NativeInsn("@label", [NLabel(_fragment_label_id(pc, label_id))])
    return NativeInsn(insn.opcode, operands, result)


def _localise_operand(x, pc, bindings, ref_bindings, is_marker):
    if isinstance(x, NLabel):
        return NLabel(_fragment_label_id(pc, x.label_id))
    if isinstance(x, NTLabel):
        return NTLabel(_fragment_label_id(pc, x.label_id))
    if isinstance(x, NHole):
        return _patch_hole_native(x, pc, bindings, ref_bindings, is_marker)
    if isinstance(x, NListOfKind):
        return NListOfKind(x.kind, [
            _localise_operand(item, pc, bindings, ref_bindings, is_marker)
            for item in x.items])
    if isinstance(x, NSwitchDictOperand):
        fresh = NativeSwitchDictDescr()
        # Not zip(): RPython's rtyper has no typer for zip(). Both lists
        # are always the same length, so an indexed loop works instead.
        labels = []
        for i in range(len(x.keys)):
            labels.append((x.keys[i], _fragment_label_id(pc, x.label_ids[i])))
        fresh._native_labels = labels
        return NDescr(fresh)
    return x


def _patch_hole_native(hole, pc, bindings, ref_bindings, is_marker):
    """A marker's own 'pc' hole identifies this block; every other hole
    takes the block's bound value.

    Asserts int/ref-only: silently truncating a float hole into an int
    would be a real, hard-to-notice correctness bug.
    """
    if hole.kind == "ref":
        return NRefConst(ref_bindings[hole.name])
    # Not %r: RPython's rtyper only implements %s/%d/... formatting.
    assert hole.kind == "int", (
        "native_pipeline: non-int/ref hole %s -- no interpreter this IR "
        "currently serves has one" % (hole.name,))
    if is_marker and hole.name == "pc":
        return NIntConst(intmask(pc))
    return NIntConst(intmask(bindings[hole.name]))


def _emit_moves_native(ssarepr, sources, destinations, scratch, _names=None):
    """Port of ProgramEmitter._emit_moves.

    Processes boundary names in sorted order, not raw dict iteration:
    this runs translated, where RPython's dict order need not match
    CPython's, so a fixed order keeps output byte-reproducible.

    ``_names``, when given, overrides the sorted order (test-only hook).
    """
    if _names is None:
        _names = destinations.keys()
        sort_strings(_names)
    moves = []
    for bname in _names:
        destination = destinations[bname]
        if destination is None:
            continue
        kind, index = destination
        # Not sources.get(bname): dict values can be tuples, which have
        # no null representation to unify against a missing-key None.
        if bname in sources:
            source = sources[bname]
            if isinstance(source, tuple):
                if source == destination:
                    continue
                source = _register(*source)
        else:
            if kind == "ref":
                source = NRefConst(lltype.nullptr(llmemory.GCREF.TO))
            else:
                # Mirrors the original's quirk: a float-kind destination
                # with no source still gets an int-kind zero constant.
                source = NIntConst(0)
        moves.append((kind, index, source))

    _boundary_moves[0] += len(moves)
    pending = list(moves)
    emitted = []
    while pending:
        progressed = False
        for move in list(pending):
            kind, index, source = move
            # Not any(genexpr): that would close over loop vars, and
            # RPython functions cannot create closures.
            blocked = False
            for other in pending:
                # Not ``other is move``: RPython has no identity compare
                # for tuples. Equality is safe here: entries never collide.
                if other == move:
                    continue
                if (isinstance(other[2], NReg) and
                        other[2].kind == kind and other[2].index == index):
                    blocked = True
                    break
            if blocked:
                continue
            emitted.append(move)
            pending.remove(move)
            progressed = True
        if not progressed:
            kind, index, source = pending[0]
            park = _register(kind, scratch[kind])
            emitted.append((kind, scratch[kind], source))
            pending[0] = (kind, index, park)

    for kind, index, source in emitted:
        ssarepr.insns.append(
            NativeInsn("%s_copy" % kind, [source], _register(kind, index)))


# ____________________________________________________________
# compute_liveness_native: port of codewriter/liveness.py.
#
# ``alive`` is keyed by NReg.nid (identity), not by (kind, index) value:
# each fragment gets its own Register objects, so two fragments' "ref
# register 2" are distinct and must not cancel each other's liveness.
# A value-keyed dict would be more precise but would change the
# liveness-chunk dedup and hence the assembled byte offsets -- byte
# identity with the original is the deliverable here.
#
# nid is a plain int (see native_fragments.py): a dict of int keys is
# RPython-legal where a dict/set of arbitrary objects is not, and every
# NReg gets a distinct nid at construction, so keying by nid reproduces
# object-identity semantics exactly.

# Runtime A/B switch (PE_COGEN_LIVENESS=old) plus one-shot env caching.
class _LivenessAlgo(object):
    def __init__(self):
        self.checked = False
        self.use_old = False


_liveness_algo = _LivenessAlgo()


def compute_liveness_native(insns):
    from rpython.rlib.debug import debug_print, have_debug_prints
    if not _liveness_algo.checked:
        _liveness_algo.checked = True
        value = os.environ.get("PE_COGEN_LIVENESS")
        _liveness_algo.use_old = value == "old"
    if _liveness_algo.use_old:
        label2alive_old = {}   # nid -> NReg dicts: own dictdef, not
                                # unified with the bitmap dict below --
                                # RPython would otherwise try to merge
                                # the two incompatible value types.
        rounds = 0
        while _compute_liveness_native_pass(insns, label2alive_old):
            rounds += 1
    else:
        # Same whole-list-until-fixpoint control flow as the dict-keyed
        # path above -- field data showed pass count (~5 on real
        # programs) was never the bottleneck. _converge_liveness_native
        # only swaps the per-pass representation (bitmaps, see its
        # docstring), not this control flow.
        label2alive_bits = {}   # label_id -> list[r_uint] bitmap
        rounds = _converge_liveness_native(insns, label2alive_bits)
    if have_debug_prints():
        nlabels = 0
        for insn in insns:
            if insn.opcode == "@label":
                nlabels += 1
        debug_print("pe-cogen-live stats insns=%d labels=%d rounds=%d" % (
            len(insns), nlabels, rounds))
    _remove_repeated_live_native(insns)


# ____________________________________________________________
# Bitmap liveness: same backward-pass-until-fixpoint control flow as
# _compute_liveness_native_pass above, but "alive"/label2alive values are
# bitmaps over a dense per-run compact index of NID, not nid-keyed dicts.
# An @label merge becomes a handful of word ORs with a cheap word-array
# "did it change" check, instead of an O(alive-set-size) dict-store loop;
# no per-'-live-' insn list allocation happens until the single final pass.
#
# Deliberately keyed by NID, not (kind, index): (kind, index) values are
# NOT globally unique in this IR -- every distinct fragment key gets its
# own register numbering starting at 0 (see native_fragments.py's
# _Converter, "one NReg per (kind, index) per-fragment, not program-
# wide"), and emit_native/_emit_moves_native mint a *fresh* NReg (fresh
# nid, same (kind, index)) for every scratch-shuffle and prologue copy at
# every block placement. A single program-wide bitmap keyed by (kind,
# index) would therefore conflate unrelated registers that only
# coincidentally share a slot number -- exactly the hazard nid-keying
# exists to avoid (see the comment above _compute_liveness_native_pass).
# Indexing by a dense compact remap of NID instead is a pure bijective
# encoding of the same nid-keyed dict this replaces, so it carries none
# of that risk while still giving O(1) bit ops instead of dict ops.

_WORD_BITS = 64   # r_uint word width for the compact liveness bitmaps


def _scan_nid_registry_native(insns):
    """One pre-pass: assign every distinct nid appearing anywhere in
    insns (operands, NListOfKind items, results) a dense compact index
    0..K-1, and remember its NReg for materializing '-live-' rewrites.
    Compact indices size the bitmap, not raw nid (a program-wide, hence
    unbounded-per-run, counter).
    """
    nid_to_compact = {}     # nid -> compact index
    registry = []           # compact index -> representative NReg
    for insn in insns:
        if insn.result is not None:
            _register_nid_native(insn.result, nid_to_compact, registry)
        for x in insn.operands:
            if isinstance(x, NReg):
                _register_nid_native(x, nid_to_compact, registry)
            elif isinstance(x, NListOfKind):
                for item in x.items:
                    if isinstance(item, NReg):
                        _register_nid_native(item, nid_to_compact, registry)
    words = len(registry) // _WORD_BITS + 1
    return nid_to_compact, registry, words


def _register_nid_native(reg, nid_to_compact, registry):
    if reg.nid not in nid_to_compact:
        reg.compact = len(registry)
        nid_to_compact[reg.nid] = reg.compact
        registry.append(reg)


def _new_bits_native(words):
    bits = []
    for _ in range(words):
        bits.append(r_uint(0))
    return bits


def _bits_set_native(bits, idx):
    bits[idx // _WORD_BITS] |= (r_uint(1) << (idx % _WORD_BITS))


def _bits_clear_native(bits, idx):
    bits[idx // _WORD_BITS] &= ~(r_uint(1) << (idx % _WORD_BITS))


def _bits_clear_all_native(bits):
    for i in range(len(bits)):
        bits[i] = r_uint(0)


def _bits_or_into_native(dst, src):
    """OR src into dst in place; both are always the same length (both
    sized from the one program-wide 'words'). Returns True if dst grew."""
    grew = False
    for i in range(len(dst)):
        old = dst[i]
        new = old | src[i]
        if new != old:
            dst[i] = new
            grew = True
    return grew


def _follow_label_bits_native(label_id, label2alive, alive):
    alive_at_point = label2alive.get(label_id)
    if alive_at_point is not None:
        _bits_or_into_native(alive, alive_at_point)


def _mark_bits_native(x, nid_to_compact, label2alive, alive):
    if isinstance(x, NReg):
        _bits_set_native(alive, x.compact)
    elif isinstance(x, NListOfKind):
        for item in x.items:
            if isinstance(item, NReg):
                _bits_set_native(alive, item.compact)
    elif isinstance(x, NTLabel):
        _follow_label_bits_native(x.label_id, label2alive, alive)
    elif isinstance(x, NDescr):
        descr = x.descr
        if isinstance(descr, NativeSwitchDictDescr):
            for _key, label in descr._native_labels:
                _follow_label_bits_native(label, label2alive, alive)


def _materialize_bits_native(alive, registry, words):
    regs = []
    for word_idx in range(words):
        word = alive[word_idx]
        if word == r_uint(0):
            continue
        base = word_idx * _WORD_BITS
        for bit in range(_WORD_BITS):
            if (word >> bit) & r_uint(1) != r_uint(0):
                compact_idx = base + bit
                if compact_idx < len(registry):
                    regs.append(registry[compact_idx])
    return regs


def _rewrite_live_insns_bits(insns, label2alive, alive, registry,
                             nid_to_compact, words):
    """Single final backward pass, run once label2alive is converged:
    rewrites every '-live-' insn from the bitmap, in nid-exact fidelity
    (registry[compact_idx] is the one-and-only NReg for that nid)."""
    _bits_clear_all_native(alive)

    for i in range(len(insns) - 1, -1, -1):
        insn = insns[i]
        opcode = insn.opcode

        if opcode == "@label":
            label_id = insn.operands[0].label_id
            alive_at_point = label2alive.get(label_id)
            if alive_at_point is None:
                alive_at_point = _new_bits_native(words)
                label2alive[label_id] = alive_at_point
            _bits_or_into_native(alive_at_point, alive)
            continue

        if opcode == "-live-":
            labels = []
            for x in insn.operands:
                if isinstance(x, NReg):
                    _bits_set_native(alive, x.compact)
                elif isinstance(x, NTLabel):
                    _follow_label_bits_native(x.label_id, label2alive, alive)
                    labels.append(x)
            regs = _materialize_bits_native(alive, registry, words)
            insns[i] = NativeInsn("-live-", regs + labels)
            continue

        if opcode == "---":
            _bits_clear_all_native(alive)
            continue

        if insn.result is not None:
            _bits_clear_native(alive, insn.result.compact)
        for x in insn.operands:
            _mark_bits_native(x, nid_to_compact, label2alive, alive)


def _segment_bounds_native(insns):
    """Segments split at every '@label': a segment is the run of insns up
    to (not including) the next '@label', so no segment ever contains one
    -- unlike a '---'-bounded segment, which may carry several labels and
    everything between them. Returns (starts, ends, label_after,
    has_label_after): segment j spans insns[starts[j]:ends[j]], and, if
    has_label_after[j], is immediately followed by the '@label' whose id
    is label_after[j] (the label id space includes negative ids -- see
    the module comment on _block_label_id -- so a missing-label flag is
    needed alongside label_after, not a sentinel int).
    """
    n = len(insns)
    starts = []
    ends = []
    label_after = []
    has_label_after = []
    start = 0
    for i in range(n):
        if insns[i].opcode == "@label":
            starts.append(start)
            ends.append(i)
            label_after.append(insns[i].operands[0].label_id)
            has_label_after.append(True)
            start = i + 1
    starts.append(start)
    ends.append(n)
    label_after.append(0)
    has_label_after.append(False)
    return starts, ends, label_after, has_label_after


def _collect_labels_native(x, label_readers, seg_idx):
    """Structural (label2alive-independent) pass: which segment indices
    ever consult label2alive[label of 'x'], directly through a jump/
    switch/'-live-' operand. Mirrors the label-following cases of
    _mark_bits_native, minus the register-marking ones."""
    if isinstance(x, NTLabel):
        _add_reader_native(label_readers, x.label_id, seg_idx)
    elif isinstance(x, NDescr):
        descr = x.descr
        if isinstance(descr, NativeSwitchDictDescr):
            for _key, label in descr._native_labels:
                _add_reader_native(label_readers, label, seg_idx)


def _add_reader_native(label_readers, label_id, seg_idx):
    readers = label_readers.get(label_id)
    if readers is None:
        readers = {}
        label_readers[label_id] = readers
    readers[seg_idx] = True


def _process_segment_bits_native(insns, start, end, label2alive, alive,
                                 nid_to_compact, words, label_after,
                                 has_label_after):
    """One local backward scan over insns[start:end) -- worklist phase
    only, no '-live-' rewrite (that happens once, in _rewrite_live_insns_
    bits, after convergence). Leaves the segment's own result (the alive
    set at its leftmost point) in 'alive' for the caller to read.

    Seeded from label2alive[label_after] (or empty, run off the end of
    the program) instead of continuing from whatever segment happens to
    sit to the right in the list: a '@label' boundary, unlike '---',
    does not reset alive, so the whole-list pass this replaces treats
    "falling into" this segment from the label after it as a real
    contribution -- label2alive[label_after] is exactly that
    contribution, already accumulated from every source (jumps in, and
    this same physical fallthrough) that can reach the label.
    """
    _bits_clear_all_native(alive)
    if has_label_after:
        seed = label2alive.get(label_after)
        if seed is not None:
            _bits_or_into_native(alive, seed)

    for i in range(end - 1, start - 1, -1):
        insn = insns[i]
        opcode = insn.opcode

        if opcode == "-live-":
            for x in insn.operands:
                if isinstance(x, NReg):
                    _bits_set_native(alive, x.compact)
                elif isinstance(x, NTLabel):
                    _follow_label_bits_native(x.label_id, label2alive, alive)
            continue

        if opcode == "---":
            _bits_clear_all_native(alive)
            continue

        if insn.result is not None:
            _bits_clear_native(alive, insn.result.compact)
        for x in insn.operands:
            _mark_bits_native(x, nid_to_compact, label2alive, alive)


def _converge_liveness_native(insns, label2alive):
    """Segment-worklist replacement for repeated whole-list backward
    sweeps -- see the module comment above _WORD_BITS for why this stays
    keyed by nid, not (kind, index).

    label2alive only crosses a segment boundary (a '@label'), so each
    segment can be fixpointed on its own and only needs reprocessing when
    a label it reads grows -- either directly (a jump/switch/'-live-'
    operand naming it, see _collect_labels_native) or as its own seed
    (the label immediately following it, see _process_segment_bits_
    native) -- instead of rescanning the whole insn list every time one
    label somewhere gains a register.

    Returns the number of segment (re)processings, for the debug_print
    stat -- not full-list passes, since there is no such notion here.
    """
    nid_to_compact, registry, words = _scan_nid_registry_native(insns)
    n = len(insns)
    if n == 0:
        return 0
    starts, ends, label_after, has_label_after = _segment_bounds_native(insns)
    num_segs = len(starts)

    label_readers = {}
    for j in range(num_segs):
        for i in range(starts[j], ends[j]):
            for x in insns[i].operands:
                _collect_labels_native(x, label_readers, j)
        if has_label_after[j]:
            _add_reader_native(label_readers, label_after[j], j)

    alive = _new_bits_native(words)
    queued = [True] * num_segs
    queue = []
    for j in range(num_segs - 1, -1, -1):
        queue.append(j)

    qhead = 0
    rounds = 0
    while qhead < len(queue):
        j = queue[qhead]
        qhead += 1
        queued[j] = False
        rounds += 1
        _process_segment_bits_native(
            insns, starts[j], ends[j], label2alive, alive,
            nid_to_compact, words, label_after[j], has_label_after[j])

        if j > 0:
            # segment j's own result is what a '@label' insn at the end
            # of segment j - 1 would have OR'd into label2alive.
            label_before = label_after[j - 1]
            alive_at_point = label2alive.get(label_before)
            if alive_at_point is None:
                alive_at_point = _new_bits_native(words)
                label2alive[label_before] = alive_at_point
            if _bits_or_into_native(alive_at_point, alive):
                readers = label_readers.get(label_before)
                if readers is not None:
                    for seg2 in readers.keys():
                        if not queued[seg2]:
                            queued[seg2] = True
                            queue.append(seg2)

    _rewrite_live_insns_bits(
        insns, label2alive, alive, registry, nid_to_compact, words)
    return rounds


def _follow_label_native(label_id, label2alive, alive):
    """Module-level, not a nested closure: RPython functions cannot
    create closures."""
    alive_at_point = label2alive.get(label_id)
    if alive_at_point is not None:
        for nid, reg in alive_at_point.items():
            alive[nid] = reg


def _mark_native(x, label2alive, alive):
    """Plain function, not a closure -- see _follow_label_native."""
    if isinstance(x, NReg):
        alive[x.nid] = x
    elif isinstance(x, NListOfKind):
        for item in x.items:
            if isinstance(item, NReg):
                alive[item.nid] = item
    elif isinstance(x, NTLabel):
        _follow_label_native(x.label_id, label2alive, alive)
    elif isinstance(x, NDescr):
        # Local var, not isinstance(x.descr, ...) inline: RPython's
        # isinstance-narrowing doesn't track attribute expressions.
        descr = x.descr
        if isinstance(descr, NativeSwitchDictDescr):
            # label is already the plain int label id here, no .name.
            for _key, label in descr._native_labels:
                _follow_label_native(label, label2alive, alive)
        # A real SwitchDictDescr can reach here too, already resolved at
        # its own fragment's compile time -- no branch needed for it.


def _compute_liveness_native_pass(insns, label2alive):
    alive = {}     # nid -> NReg
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
                    _follow_label_native(x.label_id, label2alive, alive)
                    labels.append(x)
            insns[i] = NativeInsn("-live-", alive.values() + labels)
            continue

        if insn.opcode == "---":
            alive = {}
            continue

        if insn.result is not None and insn.result.nid in alive:
            del alive[insn.result.nid]
        for x in insn.operands:
            _mark_native(x, label2alive, alive)

    return must_continue


def _remove_repeated_live_native(insns):
    """Register order within a merged '-live-' insn doesn't affect the
    assembled bytes; only the final live set (deduped by nid) matters."""
    res = []
    i = 0
    while i < len(insns):
        insn = insns[i]
        if insn.opcode != "-live-":
            res.append(insn)
            i += 1
            continue
        lives = [insn]
        labels = []
        i += 1
        while i < len(insns):
            nxt = insns[i]
            if nxt.opcode == "-live-":
                lives.append(nxt)
                i += 1
            elif nxt.opcode == "@label":
                labels.append(nxt)
                i += 1
            else:
                break
        if len(lives) == 1:
            res.extend(labels)
            res.append(lives[0])
            continue
        liveset = {}     # nid -> NReg
        extra_tlabels = []
        for live in lives:
            for x in live.operands:
                if isinstance(x, NReg):
                    liveset[x.nid] = x
                elif isinstance(x, NTLabel):
                    extra_tlabels.append(x)
        res.extend(labels)
        res.append(NativeInsn("-live-", liveset.values() + extra_tlabels))
    # Not insns[:] = res: RPython's rlist doesn't support unbounded slice
    # assignment. pop() + extend() is the RPython-legal equivalent.
    while insns:
        insns.pop()
    insns.extend(res)


# ____________________________________________________________
# NativeAssembler: port of codewriter/assembler.py's Assembler.

def _get_liveness_info_native(operands, kind):
    # Dict-as-set, not set(): RPython has no native set type.
    lives = {}
    for x in operands:
        if isinstance(x, NReg) and x.kind == kind:
            lives[chr(x.index)] = True
    return lives


def _operand_argcode_options(x, allow_short):
    """Argcode letter(s) operand 'x' could contribute to an insn key --
    two only for an unplaced int-kind hole ('c' or 'i'); None means it
    contributes nothing (NIndirectCallTargets).

    Shared by write_insn and native_insn_key_options so the two can
    never disagree about what letter one operand shape means.
    """
    if isinstance(x, NReg):
        return [x.kind[0]]
    if isinstance(x, NIntConst):
        return ["c" if int_fits_short(x.ivalue, allow_short) else "i"]
    if isinstance(x, NHole):
        if x.kind == "ref":
            return ["r"]
        assert x.kind == "int"
        return ["c", "i"] if allow_short else ["i"]
    if isinstance(x, NRefConst):
        return ["r"]
    if isinstance(x, NFloatConst):
        return ["f"]
    # Not a multi-class isinstance tuple: RPython's annotator chokes on
    # those (AttributeError deep inside classdesc.py's is_primitive_type).
    if isinstance(x, NTLabel):
        return ["L"]
    if isinstance(x, NLabel):
        return ["L"]
    if isinstance(x, NListOfKind):
        return [x.kind[0].upper()]
    if isinstance(x, NDescr):
        return ["d"]
    if isinstance(x, NSwitchDictOperand):
        return ["d"]
    if isinstance(x, NIndirectCallTargets):
        return None
    raise NotImplementedError(x)


def native_insn_key_options(insn):
    """Every 'opname/argcodes' key 'insn' could assemble to -- more than
    one only for an unresolved int-kind hole allowing short-constant form.

    '---'/'@label' return None (not real insns); '-live-' returns a fixed
    key even though it never appears here, kept only so write_insn's
    reuse of this function stays total.
    """
    if insn.opcode in ("---", "@label"):
        return None
    if insn.opcode == "-live-":
        return ["live/"]
    allow_short = insn.opcode in USE_C_FORM
    argcode_options = [""]
    for x in insn.operands:
        letters = _operand_argcode_options(x, allow_short)
        if letters is None:
            continue
        argcode_options = [prefix + letter for prefix in argcode_options
                           for letter in letters]
    if insn.result is not None:
        argcode_options = [a + ">" + insn.result.kind[0]
                           for a in argcode_options]
    return [insn.opcode + "/" + a for a in argcode_options]


class NativeAssembler(Assembler):
    """Subclasses Assembler: everything not tied to insn shape (dedup,
    emit_reg, liveness byte-packing, fix_labels, check_result,
    make_jitcode) is inherited unchanged. Only write_insn and assemble
    are overridden.

    ``share_with``, when given, is a real Assembler whose cross-JitCode
    session state (descrs, the insns opcode table, all_liveness,
    counters) this instance adopts by reference, so every JitCode shares
    one global opcode table and liveness string. Not thread-safe: nothing
    guards concurrent mutation of the shared state.

    ``readonly``, when True (the runtime cogen path, PortalLinker.
    _emit_native): insns stays shared read-only, but all_liveness and
    counters stay private, so a late JitCode's own liveness lands in its
    own chunk instead of extending the frozen global string.
    """

    def __init__(self, share_with=None, readonly=False):
        Assembler.__init__(self)
        # Reset again in assemble(): inherited setup() doesn't know
        # about this list.
        self.native_switchdictdescrs = []
        self._share_with = share_with
        self._readonly = readonly
        if share_with is not None:
            self.descrs = share_with.descrs
            self._descr_dict = share_with._descr_dict
            self.insns = share_with.insns
            self.indirectcalltargets = share_with.indirectcalltargets
            self.list_of_addr2name = share_with.list_of_addr2name
            self._seen_raw_objects = share_with._seen_raw_objects
            if not readonly:
                self._counters = share_with._counters
                self.all_liveness = share_with.all_liveness
                self.all_liveness_positions = share_with.all_liveness_positions

    def assemble(self, ssarepr, jitcode=None, num_regs=None):
        self.setup(ssarepr.name)
        self.native_switchdictdescrs = []
        if num_regs is not None:
            self.count_regs.update(num_regs)
        ssarepr._insns_pos = []
        for insn in ssarepr.insns:
            ssarepr._insns_pos.append(len(self.code))
            self.write_insn(insn)
        self.fix_labels()
        self.check_result()
        if jitcode is None:
            jitcode = JitCode(ssarepr.name)
        # Not jitcode._ssarepr = ssarepr: leaving it unset avoids the
        # RPython annotator unifying it with a real SSARepr elsewhere.
        self.make_jitcode(jitcode)
        # ponytail: no jitcode._dump -- format_assembler assumes real
        # SSARepr tuples and would crash on native operands.
        self._count_jitcodes += 1
        return jitcode

    def write_insn(self, insn):
        if insn.opcode == "---":
            return
        if insn.opcode == "@label":
            self.label_positions[insn.operands[0].label_id] = len(self.code)
            return
        if insn.opcode == "-live-":
            key = len(self.code)
            self.startpoints[key] = True
            self.num_liveness_ops += 1
            live_i = _get_liveness_info_native(insn.operands, "int")
            live_r = _get_liveness_info_native(insn.operands, "ref")
            live_f = _get_liveness_info_native(insn.operands, "float")
            assert key not in self.liveness
            self.liveness[key] = live_i, live_r, live_f
            num = self._insn_number("live/")
            self.code.append(chr(num))
            self._encode_liveness(live_i, live_r, live_f)
            return

        startposition = len(self.code)
        self.code.append("temporary placeholder")
        argcodes = []
        allow_short = (insn.opcode in USE_C_FORM)
        for x in insn.operands:
            # Only the argcode letter comes from _operand_argcode_options,
            # so this stays in sync with the translation-time coverage pass.
            if isinstance(x, NReg):
                self.emit_reg(x)
            elif isinstance(x, NIntConst):
                self.emit_resolved_const(x.ivalue, "int",
                                         allow_short=allow_short)
            elif isinstance(x, NRefConst):
                # No explicit dedup_key: the default (dedup_key=value) is
                # correct now that constants_dict_r keys via rd_eq/rd_hash.
                self.emit_resolved_const(x.value, "ref")
            elif isinstance(x, NFloatConst):
                self.emit_resolved_const(x.value, "float")
            elif isinstance(x, NTLabel):
                self.alllabels[len(self.code)] = True
                self.tlabel_positions.append((x.label_id, len(self.code)))
                self.code.append("temp 1")
                self.code.append("temp 2")
            elif isinstance(x, NListOfKind):
                lst = x.items
                # AssemblerError, not assert: a legitimate method can hit
                # this cap at runtime; the caller must be able to decline.
                if len(lst) > 255:
                    raise AssemblerError("list too long!")
                self.code.append(chr(len(lst)))
                for item in lst:
                    if isinstance(item, NReg):
                        assert x.kind == item.kind
                        self.emit_reg(item)
                    elif isinstance(item, NIntConst):
                        assert x.kind == "int"
                        self.emit_resolved_const(item.ivalue, "int")
                    elif isinstance(item, NRefConst):
                        assert x.kind == "ref"
                        # Same as above: no explicit dedup_key needed.
                        self.emit_resolved_const(item.value, "ref")
                    elif isinstance(item, NFloatConst):
                        assert x.kind == "float"
                        self.emit_resolved_const(item.value, "float")
                    else:
                        # Not %r: an arbitrary operand has no RPython-
                        # legal string conversion.
                        raise NotImplementedError(
                            "found an operand of an unexpected type in "
                            "NListOfKind()")
            elif isinstance(x, NDescr):
                d = x.descr
                if isinstance(d, NativeSwitchDictDescr):
                    # Fresh every placement: its labels depend on this
                    # program's layout, so it can never be pre-stamped.
                    if d not in self._descr_dict:
                        self._descr_dict[d] = len(self.descrs)
                        self.descrs.append(d)
                    # Own list, resolved by fix_labels's override below.
                    self.native_switchdictdescrs.append(d)
                    num = self._descr_dict[d]
                elif self._readonly:
                    # Read the index already stamped by native_table.
                    # -1 means uncovered: decline the whole program.
                    num = d.pe_descr_index
                    if num < 0:
                        debug_print("runtime cogen: descr not covered by "
                                   "precompiled fragments")
                        raise AssemblerError(
                            "descr not covered by precompiled fragments")
                else:
                    if d not in self._descr_dict:
                        self._descr_dict[d] = len(self.descrs)
                        self.descrs.append(d)
                    num = self._descr_dict[d]
                assert 0 <= num <= 0xFFFF, "too many AbstractDescrs!"
                self.code.append(chr(num & 0xFF))
                self.code.append(chr(num >> 8))
            elif isinstance(x, NIndirectCallTargets):
                for target in x.lst:
                    self.indirectcalltargets[target] = True
            elif isinstance(x, NHole):
                # Not %r: see _patch_hole_native's note above.
                raise AssertionError(
                    "unpatched hole %s reached the assembler" % (x.name,))
            else:
                raise NotImplementedError(x)
            letters = _operand_argcode_options(x, allow_short)
            if letters is not None:
                assert len(letters) == 1
                argcodes.append(letters[0])

        if insn.result is not None:
            argcodes.append(">")
            self.emit_reg(insn.result)
            argcodes.append(insn.result.kind[0])

        opname = insn.opcode
        if ">" in argcodes:
            assert argcodes.index(">") == len(argcodes) - 2
            self.resulttypes[len(self.code)] = argcodes[-1]
        key = opname + "/" + "".join(argcodes)
        num = self._insn_number(key)
        self.code[startposition] = chr(num)
        self.startpoints[startposition] = True

    def _insn_number(self, key):
        """The opcode byte for 'key'.

        Readonly mode: a lookup miss means this program needs an
        (opname, argcodes) combo no precompiled fragment used -- raises
        to decline the whole program, rather than minting a new opcode
        number nothing can fold back into the shared table.
        """
        if self._readonly:
            num = self.insns.get(key, -1)
            if num < 0:
                debug_print("runtime cogen: no precompiled insn for", key)
                # Not %r: see _patch_hole_native's note above.
                raise AssemblerError("no precompiled insn for %s" % (key,))
            return num
        return self.insns.setdefault(key, len(self.insns))

    def fix_labels(self):
        """Override, not inherit: the base resolves switchdictdescrs via
        label.name, never populated by this path. Resolves
        native_switchdictdescrs instead.
        """
        for name, pos in self.tlabel_positions:
            assert self.code[pos] == "temp 1"
            assert self.code[pos + 1] == "temp 2"
            target = self.label_positions[name]
            assert 0 <= target <= 0xFFFF
            self.code[pos] = chr(target & 0xFF)
            self.code[pos + 1] = chr(target >> 8)
        for descr in self.native_switchdictdescrs:
            as_dict = {}
            for key, label_id in descr._native_labels:
                as_dict[key] = self.label_positions[label_id]
            descr.attach(as_dict)


def _reg_key(operand):
    """A register operand's identity as an int, -1 for anything else.

    An int, not a string or tuple: this runs once per operand per fold, and
    building a fresh string there dominated the whole pass; a tuple cannot
    union with the None a non-register would need.  kind is "int", "ref" or
    "float", so the first character distinguishes them.
    """
    if isinstance(operand, NReg):
        return operand.index * 4 + ord(operand.kind[0]) % 4
    return -1


def _constant_load(insn):
    """(constant, target key) of a ``<kind>_copy`` of a constant."""
    if not insn.opcode.endswith("_copy"):
        return None, -1
    if len(insn.operands) != 1 or insn.result is None:
        return None, -1
    source = insn.operands[0]
    if isinstance(source, NReg):
        return None, -1            # a register move, not a materialisation
    target = _reg_key(insn.result)
    if target < 0:
        return None, -1
    return source, target


def _ends_region(opcode):
    """A label joins other paths and a jump leaves; either ends a region."""
    return opcode == "@label" or opcode.startswith("goto")


def _keeps_argcode(constant, reader):
    """Would substituting ``constant`` leave the reader's argcode alone?

    The assembler builds an insn key from its operands' argcode letters, and
    the translation-time coverage pass only registered the keys the operands
    had then.  A ref constant contributes the same "r" a ref register does,
    and so does an int constant too wide for the short form -- but a small
    int in an opcode that takes the short form becomes "c", a key nothing
    registered, so that one must stay in its register.
    """
    if isinstance(constant, NIntConst):
        return not int_fits_short(constant.ivalue,
                                  reader.opcode in USE_C_FORM)
    return isinstance(constant, NRefConst)


def _registers_read_before_written(insns):
    """Registers some region reads before it writes them.

    A register outside this set never carries a value across a region
    boundary, so a definition of it dies at the end of its own region and
    liveness -- recomputed after this pass -- will say so.

    ``-live-`` operands count as reads too: they name what a guard may
    resume into, so treating them like any other operand read keeps a
    register that only a guard needs from looking dead.
    """
    crossing = {}
    written = {}
    for insn in insns:
        if _ends_region(insn.opcode):
            written = {}
            continue
        for operand in insn.operands:
            key = _reg_key(operand)
            if key >= 0 and key not in written:
                crossing[key] = True
        key = _reg_key(insn.result)
        if key >= 0:
            written[key] = True
    return crossing


def _live_targets(insns):
    """Every register any ``-live-`` insn names, anywhere in ``insns``.

    Computed once and reused: a load whose target is in this set must
    never be dropped, in the read-before-write pass and in the fold.
    """
    targets = {}
    for insn in insns:
        if insn.opcode != "-live-":
            continue
        for operand in insn.operands:
            key = _reg_key(operand)
            if key >= 0:
                targets[key] = True
    return targets


class _PendingLoad(object):
    """A constant load in the current region still looking for its
    region-end without having been proven unsafe to fold."""
    def __init__(self, constant, position):
        self.constant = constant
        self.position = position
        self.reads = []    # (insn index, operand index) pairs seen so far


def _finalize_pending(entry, insns, keep):
    """The load in 'entry' reached the end of its life unblocked: apply
    its collected substitutions and drop the load itself."""
    for later, index in entry.reads:
        insns[later].operands[index] = entry.constant
    keep[entry.position] = False


def _finalize_all_pending(pending, insns, keep):
    count = 0
    for key, entry in pending.items():
        _finalize_pending(entry, insns, keep)
        count += 1
    return count


def fold_constant_loads(insns):
    """Substitute a materialised late-static value into what reads it.

    Specialising for one code object turns every pc, oparg and instruction
    start into a constant, and the flattener puts each one in a register
    first: on PyPy that is 148 of the 528 instructions in a seventeen-block
    program, and the meta-interpreter executes every one of them per trace.

    Runs after liveness, so every ``-live-`` insn's operands already name
    the registers live at that point, including whatever a guard would
    resume into.  Both the read-before-write pass and this pass's own scan
    treat those operands as reads, so a register a guard needs can never
    look unread and its load is never dropped.  Only registers no region
    reads before writing qualify, which is what makes dropping the
    definition safe without tracking flow across labels.

    Dropping a load only removes registers no ``-live-`` names anywhere,
    so the existing ``-live-`` sets -- conservative supersets already --
    stay valid and liveness is not recomputed.

    One forward walk, not one scan per load: a region's constant loads
    are tracked in ``pending`` (target register -> the load looking for
    its readers) and resolved -- folded or cancelled -- as the walk
    passes over their readers, instead of each load re-scanning the rest
    of its region on its own.
    """
    live_targets = _live_targets(insns)
    crossing = _registers_read_before_written(insns)
    keep = [True] * len(insns)
    folded = 0
    blocked_crossing = 0
    blocked_live = 0
    blocked_argcode = 0
    pending = {}    # target register key -> _PendingLoad

    for position in range(len(insns)):
        insn = insns[position]
        opcode = insn.opcode

        if _ends_region(opcode):
            folded += _finalize_all_pending(pending, insns, keep)
            pending = {}
            continue

        if opcode == "-live-":
            # A guard may resume into a named register: its load must
            # survive, so cancel any pending fold for it (in practice
            # already excluded via live_targets above; kept here too so
            # this stays correct even if that upfront filter ever loosens).
            for operand in insn.operands:
                key = _reg_key(operand)
                if key in pending:
                    del pending[key]
            continue

        for index in range(len(insn.operands)):
            key = _reg_key(insn.operands[index])
            if key < 0 or key not in pending:
                continue
            entry = pending[key]
            if _keeps_argcode(entry.constant, insn):
                entry.reads.append((position, index))
            else:
                del pending[key]    # blocked: the load stays
                blocked_argcode += 1

        result_key = _reg_key(insn.result)
        if result_key >= 0 and result_key in pending:
            # A second write ends the first load's life: the reads seen
            # so far are final, whether or not this insn is itself
            # another constant load into the same register.
            _finalize_pending(pending[result_key], insns, keep)
            folded += 1
            del pending[result_key]

        constant, target = _constant_load(insn)
        if constant is not None:
            if target in crossing:
                blocked_crossing += 1
            elif target in live_targets:
                blocked_live += 1
            else:
                pending[target] = _PendingLoad(constant, position)

    # Reaching the end of insns with no closing region marker finalizes
    # whatever is still pending, same as hitting one would.
    folded += _finalize_all_pending(pending, insns, keep)

    _folded_loads.blocked_crossing += blocked_crossing
    _folded_loads.blocked_live += blocked_live
    _folded_loads.blocked_argcode += blocked_argcode
    if not folded:
        return insns
    _folded_loads.folded += folded
    result = []
    for position in range(len(insns)):
        if keep[position]:
            result.append(insns[position])
    return result


def emit_and_assemble_native(native_table, program, name,
                             has_merge_points=False, assembler=None,
                             optimise=True):
    """Full native pipeline: emit_native -> compute_liveness_native ->
    fold_constant_loads -> NativeAssembler.assemble, mirroring
    ProgramEmitter.emit's own order plus the post-liveness fold.

    Returns (jitcode, entry_positions, assembler).
    """
    from rpython.rlib.debug import debug_start, debug_stop
    debug_start("pe-cogen-emit")
    ssarepr, counts = emit_native(native_table, program, name, has_merge_points)
    debug_stop("pe-cogen-emit")
    debug_start("pe-cogen-live")
    compute_liveness_native(ssarepr.insns)
    debug_stop("pe-cogen-live")
    if optimise and _fold_enabled():
        # The equivalence gate passes optimise=False: it checks that this
        # pipeline lowers a program exactly as the translation-time one
        # does, which is about the lowering, not about what is folded
        # after.  Folds after liveness, so -live- operands name the real
        # read set (see fold_constant_loads).
        debug_start("pe-cogen-fold")
        ssarepr.insns = fold_constant_loads(ssarepr.insns)
        if have_debug_prints():
            debug_print(
                "pe-cogen-fold folded=%d blocked-crossing=%d "
                "blocked-live=%d blocked-argcode=%d" % (
                    _folded_loads.folded, _folded_loads.blocked_crossing,
                    _folded_loads.blocked_live, _folded_loads.blocked_argcode))
        _folded_loads.blocked_crossing = 0
        _folded_loads.blocked_live = 0
        _folded_loads.blocked_argcode = 0
        debug_stop("pe-cogen-fold")
    _log_insn_mix(ssarepr, program)
    if assembler is None:
        assembler = NativeAssembler()
    jitcode = JitCode(name, fnaddr=llmemory.NULL)
    debug_start("pe-cogen-asm")
    assembler.assemble(ssarepr, jitcode, counts)
    debug_stop("pe-cogen-asm")
    # Not dict(genexpr): this one needs a closure over ``assembler``,
    # which is not RPython-legal, so it's a loop instead.
    entry_positions = {}
    for pc in program.blocks:
        entry_positions[pc] = assembler.label_positions[_block_label_id(pc)]
    return jitcode, entry_positions, assembler
