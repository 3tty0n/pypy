"""Runtime port of ProgramEmitter.emit/compute_liveness onto NativeInsn."""

import os

from rpython.jit.codewriter.assembler import (
    Assembler, AssemblerError, USE_C_FORM, int_fits_short)
from rpython.jit.codewriter.flatten import KINDS
from rpython.jit.codewriter.jitcode import JitCode, SwitchDictDescr
from rpython.rlib.debug import debug_print, have_debug_prints
from rpython.rlib.rarithmetic import intmask, r_uint
from rpython.rtyper.lltypesystem import llmemory, lltype

from rpython.translator.backendopt.native_fragments import (
    NReg, NIntConst, NRefConst, NFloatConst, NHole, NLabel,
    NTLabel, NDescr, NListOfKind, NIndirectCallTargets, NSwitchDictOperand,
    NativeInsn, native_fragment_for)
from rpython.translator.backendopt.partialeval_template import (
    flatten_resolved_targets, sort_ints, sort_strings, uses_compact_entries)


# Block-boundary label ids are negative; in-fragment ids are pc-based >= 0.
_LOCAL_LABEL_SPACE = 1 << 20  # headroom: pc * this must fit a 63-bit int

def _block_label_id(pc):
    return -(intmask(pc) + 1)

def _fragment_label_id(pc, local_label_id):
    assert 0 <= local_label_id < _LOCAL_LABEL_SPACE
    return intmask(pc) * _LOCAL_LABEL_SPACE + local_label_id


class NativeSwitchDictDescr(SwitchDictDescr):
    """Uses ._native_labels (plain int ids), not ._labels (TLabels)."""
    _native_labels = []


class NativeSSARepr(object):
    """Minimal stand-in for SSARepr so Assembler.assemble can run on it."""
    def __init__(self, name):
        self.name = name
        self.insns = []
        self._insns_pos = None


_prologue_copies = [0]   # list holder: a module int would fold to a constant
_boundary_moves = [0]


class _FoldedLoads(object):
    folded = 0
    blocked_crossing = 0
    blocked_live = 0
    blocked_argcode = 0
    enabled = True   # PYPY_PE_FOLD=0 disables the fold, for A/B measurement
    env_read = False


_folded_loads = _FoldedLoads()


def _fold_enabled():
    if not _folded_loads.env_read:
        _folded_loads.env_read = True
        if os.environ.get("PYPY_PE_FOLD") == "0":
            _folded_loads.enabled = False
    return _folded_loads.enabled


def _log_insn_mix(ssarepr, program):
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
    for name, count in counts.items():
        debug_print("pe-cogen-mix   %s %d" % (name, count))


def emit_native(native_table, program, name="emitted-residual-native",
                has_merge_points=False):
    """Concatenates every reached block's fragment into one flat program."""
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
        ssarepr.insns.append(
            NativeInsn("@label", [NLabel(_block_label_id(pc))]))
        if has_merge_points and (not compact_entries or pc in headers):
            _initialise_scratch_native(ssarepr, fragments, counts)
        _place_native(ssarepr, program, pc, fragments, scratch)

    return ssarepr, counts


def _register(kind, index):
    return NReg(kind, index)


def _initialise_scratch_native(ssarepr, fragments, counts):
    entry = {}
    for fragment in fragments.values():
        for kind, index in fragment.boundary_entry.values():
            entry[kind] = max(entry.get(kind, 0), index + 1)
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


def _place_native(ssarepr, program, pc, fragments, scratch):
    fragment = fragments[pc]
    block = program.blocks[pc]
    for kind, index, bname in fragment.prologue:
        const = NIntConst(intmask(block.bindings[bname]))
        _prologue_copies[0] += 1
        ssarepr.insns.append(
            NativeInsn("%s_copy" % kind, [const], _register(kind, index)))
    targets = flatten_resolved_targets(
        block.template.resolve_targets(block.bindings), len(fragment.exits))

    for insn in fragment.insns:
        exit_index = _exit_index(insn)
        if exit_index < 0:   # -1: RPython ints have no null value
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
    if insn.opcode == "int_return" and len(insn.operands) == 1 and \
            isinstance(insn.operands[0], NIntConst):
        return insn.operands[0].ivalue
    return -1


def _localise_native(insn, pc, bindings, ref_bindings):
    is_marker = insn.opcode in ("jit_merge_point", "pe_bailout_point")
    operands = [_localise_operand(x, pc, bindings, ref_bindings, is_marker)
               for x in insn.operands]
    result = insn.result
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
        labels = []
        for i in range(len(x.keys)):
            labels.append((x.keys[i], _fragment_label_id(pc, x.label_ids[i])))
        fresh._native_labels = labels
        return NDescr(fresh)
    return x


def _patch_hole_native(hole, pc, bindings, ref_bindings, is_marker):
    """A marker's own 'pc' hole is this block; every other hole is bound."""
    if hole.kind == "ref":
        return NRefConst(ref_bindings[hole.name])
    assert hole.kind == "int", (
        "native_pipeline: non-int/ref hole %s -- no interpreter this IR "
        "currently serves has one" % (hole.name,))
    if is_marker and hole.name == "pc":
        return NIntConst(intmask(pc))
    return NIntConst(intmask(bindings[hole.name]))


def _emit_moves_native(ssarepr, sources, destinations, scratch, _names=None):
    """A parallel move; ``_names`` overrides the sorted order (test hook)."""
    if _names is None:
        _names = destinations.keys()
        sort_strings(_names)
    moves = []
    for bname in _names:
        destination = destinations[bname]
        if destination is None:
            continue
        kind, index = destination
        # source can be a tuple, so "not in sources" isn't a None default.
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
                source = NIntConst(0)
        moves.append((kind, index, source))

    _boundary_moves[0] += len(moves)
    pending = list(moves)
    emitted = []
    while pending:
        progressed = False
        for move in list(pending):
            kind, index, source = move
            # Not any(genexpr): RPython functions cannot create closures.
            blocked = False
            for other in pending:
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


# alive is keyed by NReg.nid: each fragment mints its own registers.

class _LivenessAlgo(object):
    """Runtime A/B switch (PE_COGEN_LIVENESS=old), one-shot env caching."""
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
        label2alive_old = {}   # nid -> NReg
        rounds = 0
        while _compute_liveness_native_pass(insns, label2alive_old):
            rounds += 1
    else:
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


# Bitmap liveness: nid-keyed; an @label merge is word ORs, not dict stores.
_WORD_BITS = 64   # r_uint word width for the compact liveness bitmaps


def _scan_nid_registry_native(insns):
    """Assigns every distinct nid a compact index 0..K-1, sizing the bitmap."""
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
    """ORs src into dst in place; returns True if dst grew."""
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
    """Final backward pass, run once label2alive has converged."""
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
    """Segments split at every '@label'; segment j spans [starts[j]:ends[j])."""
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
    """Which segments consult label2alive[label of 'x'] directly."""
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
    """One backward scan (worklist only, no rewrite); result left in 'alive'."""
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
    """Reprocesses a segment only when a label it reads grows."""
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
            # A trailing '@label' in segment j - 1 would OR this in.
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
    alive_at_point = label2alive.get(label_id)
    if alive_at_point is not None:
        for nid, reg in alive_at_point.items():
            alive[nid] = reg


def _mark_native(x, label2alive, alive):
    if isinstance(x, NReg):
        alive[x.nid] = x
    elif isinstance(x, NListOfKind):
        for item in x.items:
            if isinstance(item, NReg):
                alive[item.nid] = item
    elif isinstance(x, NTLabel):
        _follow_label_native(x.label_id, label2alive, alive)
    elif isinstance(x, NDescr):
        descr = x.descr
        if isinstance(descr, NativeSwitchDictDescr):
            for _key, label in descr._native_labels:
                _follow_label_native(label, label2alive, alive)


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
    """Order within a merged '-live-' doesn't matter; only the deduped set."""
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
    while insns:   # not insns[:] = res: rlist has no unbounded slice assign
        insns.pop()
    insns.extend(res)


def _get_liveness_info_native(operands, kind):
    lives = {}
    for x in operands:
        if isinstance(x, NReg) and x.kind == kind:
            lives[chr(x.index)] = True
    return lives


def _operand_argcode_options(x, allow_short):
    """Argcode letter(s) 'x' contributes to an insn key, or None."""
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
    """Every 'opname/argcodes' key 'insn' could assemble to."""
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
    """share_with adopts an Assembler's state; readonly keeps own liveness."""

    def __init__(self, share_with=None, readonly=False):
        Assembler.__init__(self)
        self.native_switchdictdescrs = []
        self._share_with = share_with
        self._readonly = readonly
        if share_with is not None:
            self.descrs = share_with.descrs
            self._descr_dict = share_with._descr_dict
            self.insns = share_with.insns
            self.indirectcalltargets = share_with.indirectcalltargets
            self.list_of_addr2name = share_with.list_of_addr2name
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
        self.make_jitcode(jitcode)
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
            if isinstance(x, NReg):
                self.emit_reg(x)
            elif isinstance(x, NIntConst):
                self.emit_resolved_const(x.ivalue, "int",
                                         allow_short=allow_short)
            elif isinstance(x, NRefConst):
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
                        self.emit_resolved_const(item.value, "ref")
                    elif isinstance(item, NFloatConst):
                        assert x.kind == "float"
                        self.emit_resolved_const(item.value, "float")
                    else:
                        raise NotImplementedError(
                            "found an operand of an unexpected type in "
                            "NListOfKind()")
            elif isinstance(x, NDescr):
                d = x.descr
                if isinstance(d, NativeSwitchDictDescr):
                    if d not in self._descr_dict:
                        self._descr_dict[d] = len(self.descrs)
                        self.descrs.append(d)
                    self.native_switchdictdescrs.append(d)
                    num = self._descr_dict[d]
                elif self._readonly:
                    num = d.pe_descr_index   # -1: uncovered, decline program
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
        """In readonly mode, a miss declines the whole program."""
        if self._readonly:
            num = self.insns.get(key, -1)
            if num < 0:
                debug_print("runtime cogen: no precompiled insn for", key)
                raise AssemblerError("no precompiled insn for %s" % (key,))
            return num
        return self.insns.setdefault(key, len(self.insns))

    def fix_labels(self):
        """Override: resolves native_switchdictdescrs, not label.name."""
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
    """A register's identity as an int, -1 otherwise (a fast fold key)."""
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
        return None, -1
    target = _reg_key(insn.result)
    if target < 0:
        return None, -1
    return source, target


def _ends_region(opcode):
    return opcode == "@label" or opcode.startswith("goto")


def _keeps_argcode(constant, reader):
    """Would substituting ``constant`` change the reader's argcode letter?"""
    if isinstance(constant, NIntConst):
        return not int_fits_short(constant.ivalue,
                                  reader.opcode in USE_C_FORM)
    return isinstance(constant, NRefConst)


def _registers_read_before_written(insns):
    """Registers some region reads before writing."""
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
    """Every register any ``-live-`` names; its load must never be dropped."""
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
    """A constant load still looking for its region-end, unblocked so far."""
    def __init__(self, constant, position):
        self.constant = constant
        self.position = position
        self.reads = []    # (insn index, operand index) pairs seen so far


def _finalize_pending(entry, insns, keep):
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
    """Substitutes a materialised late-static value into what reads it."""
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
                del pending[key]
                blocked_argcode += 1

        result_key = _reg_key(insn.result)
        if result_key >= 0 and result_key in pending:
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
    """emit_native -> compute_liveness_native -> fold -> assemble."""
    from rpython.rlib.debug import debug_start, debug_stop
    debug_start("pe-cogen-emit")
    ssarepr, counts = emit_native(native_table, program, name, has_merge_points)
    debug_stop("pe-cogen-emit")
    debug_start("pe-cogen-live")
    compute_liveness_native(ssarepr.insns)
    debug_stop("pe-cogen-live")
    if optimise and _fold_enabled():
        # optimise=False for the equivalence gate: it checks lowering only.
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
    entry_positions = {}
    for pc in program.blocks:
        entry_positions[pc] = assembler.label_positions[_block_label_id(pc)]
    return jitcode, entry_positions, assembler
