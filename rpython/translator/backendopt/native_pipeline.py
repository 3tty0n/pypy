"""Runtime-shaped ports of ProgramEmitter.emit / compute_liveness / Assembler,
operating on the native_fragments.py IR instead of SSARepr/flowspace objects.

Mirrors each original 1:1 on purpose: test_native_pipeline.py's equivalence
gate requires byte-identical output, so keep behavior identical here, not
just "cleaner".

RPython-legal only, throughout: a real runtime_cogen callback reaches
every function here, so each must translate, not just run untranslated.
"""

from rpython.jit.codewriter.assembler import Assembler, AssemblerError, USE_C_FORM
from rpython.jit.codewriter.flatten import KINDS
from rpython.jit.codewriter.jitcode import JitCode, SwitchDictDescr
from rpython.jit.metainterp.history import AbstractDescr
from rpython.rlib.debug import debug_print
from rpython.rtyper.lltypesystem import llmemory, lltype

from rpython.translator.backendopt.native_fragments import (
    NReg, NIntConst, NRefConst, NFloatConst, NHole, NLabel,
    NTLabel, NDescr, NListOfKind, NIndirectCallTargets, NSwitchDictOperand,
    NativeInsn, native_fragment_for)
from rpython.translator.backendopt.partialeval_template import (
    flatten_resolved_targets, sort_ints)


# ____________________________________________________________
# Global label ids: block boundaries and in-fragment labels must never
# collide once flattened into one program-wide, int-keyed id space.
#
# Block boundaries get negative ids; in-fragment labels get non-negative
# ids built from (placement pc, local label_id).

_LOCAL_LABEL_SPACE = 1 << 20  # headroom: pc * this must fit a 63-bit int

def _block_label_id(pc):
    return -(pc + 1)

def _fragment_label_id(pc, local_label_id):
    assert 0 <= local_label_id < _LOCAL_LABEL_SPACE
    return pc * _LOCAL_LABEL_SPACE + local_label_id


class NativeSwitchDictDescr(SwitchDictDescr):
    """SwitchDictDescr resolved by NativeAssembler's own fix_labels, not
    Assembler's shared one.

    Must stay a real SwitchDictDescr: pyjitpl.py/blackhole.py assert
    isinstance(switchdict, SwitchDictDescr) before reading
    .dict/.const_keys_in_order (set by .attach(), inherited unchanged).
    ._labels differs by backend though -- SSARepr fills it with TLabel
    instances (.name is a flowspace Link), this path only has plain ints
    (global label ids), which can't unify with an instance. Storing this
    path's data as ._native_labels instead keeps it out of that domain.
    """


class NativeSSARepr(object):
    """Minimal stand-in for SSARepr so Assembler.assemble can run on it."""
    def __init__(self, name):
        self.name = name
        self.insns = []
        self._insns_pos = None


# ____________________________________________________________
# emit_native: port of ProgramEmitter.emit/_place/_localise/_emit_moves.

def emit_native(native_table, program, name="emitted-residual-native",
                has_merge_points=False):
    """Concatenate every reached block's fragment into one flat native
    instruction list, as ProgramEmitter.emit does for SSARepr.

    Returns (NativeSSARepr, counts), counts being the per-kind register
    count to pass to NativeAssembler.assemble.
    """
    # Dict-as-set, plain loops: set()/dict(genexpr)/sorted() aren't
    # RPython-legal (see generating_extension.py, partialeval_template.py).
    headers = {}
    if has_merge_points:
        for pc in program.loop_headers:
            headers[pc] = True
        headers[program.entry_pc] = True
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
        if has_merge_points:
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
    # KINDS is already int/ref/float order, so iterating it directly
    # matches sorted(counts) (not RPython-legal) byte-for-byte.
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
    """Port of ProgramEmitter._place."""
    fragment = fragments[pc]
    block = program.blocks[pc]
    for kind, index, bname in fragment.prologue:
        const = NIntConst(block.bindings[bname])
        ssarepr.insns.append(
            NativeInsn("%s_copy" % kind, [const], _register(kind, index)))
    targets = flatten_resolved_targets(
        block.template.resolve_targets(block.bindings), len(fragment.exits))

    for insn in fragment.insns:
        exit_index = _exit_index(insn)
        # -1 not None: RPython ints have no null value (see
        # generating_extension.py's last_blocked). Real index is >= 0.
        if exit_index < 0:
            ssarepr.insns.append(_localise_native(insn, pc, block.bindings))
            continue
        exit = fragment.exits[exit_index]
        target = targets[exit_index]
        _emit_moves_native(ssarepr, exit.operands,
                           fragments[target].boundary_entry, scratch)
        ssarepr.insns.append(
            NativeInsn("goto", [NTLabel(_block_label_id(target))]))


def _exit_index(insn):
    # insn is one of fragment.insns -- pre-localisation, straight out of
    # fragment_to_native -- so the only way operands[0] can be an NIntConst
    # here at all is via a genuine flowspace Constant (_const_operand_for);
    # the runtime-synthesized NIntConsts (_patch_hole_native and friends)
    # only ever appear post-localisation.  No "was this a real Constant"
    # check needed, unlike the polymorphic NConst this replaced.
    if insn.opcode == "int_return" and len(insn.operands) == 1 and \
            isinstance(insn.operands[0], NIntConst):
        return insn.operands[0].ivalue
    return -1


def _localise_native(insn, pc, bindings):
    """Port of ProgramEmitter._localise/_patch_hole."""
    is_marker = insn.opcode in ("jit_merge_point", "pe_bailout_point")
    operands = [_localise_operand(x, pc, bindings, is_marker)
               for x in insn.operands]
    result = insn.result   # a register, never a label/hole -- untouched
    if insn.opcode == "@label":
        label_id = insn.operands[0].label_id
        return NativeInsn("@label", [NLabel(_fragment_label_id(pc, label_id))])
    return NativeInsn(insn.opcode, operands, result)


def _localise_operand(x, pc, bindings, is_marker):
    if isinstance(x, NLabel):
        return NLabel(_fragment_label_id(pc, x.label_id))
    if isinstance(x, NTLabel):
        return NTLabel(_fragment_label_id(pc, x.label_id))
    if isinstance(x, NHole):
        return _patch_hole_native(x, pc, bindings, is_marker)
    if isinstance(x, NListOfKind):
        return NListOfKind(x.kind, [
            _localise_operand(item, pc, bindings, is_marker)
            for item in x.items])
    if isinstance(x, NSwitchDictOperand):
        fresh = NativeSwitchDictDescr()
        # Not zip(x.keys, x.label_ids): RPython's rtyper has no zip()
        # typer. Both lists are always the same length (built together,
        # native_fragments.py), so a manual indexed loop suffices.
        labels = []
        for i in range(len(x.keys)):
            labels.append((x.keys[i], _fragment_label_id(pc, x.label_ids[i])))
        fresh._native_labels = labels
        return NDescr(fresh)
    return x   # NReg, NIntConst, NRefConst, NFloatConst, NDescr: placement-
               # invariant, copied verbatim


def _patch_hole_native(hole, pc, bindings, is_marker):
    """A marker's own 'pc' hole identifies this block; every other hole
    (including a Continue exit's own next-pc) takes the bound value.
    Every hole here is a plain int; asserted since silently truncating a
    ref/float hole would be a correctness bug, not just an RPython one.
    """
    # Not %r: RPython's rtyper only implements %s/%d/... formatting
    # (rstr.py's do_stringformat); hole.name is already a plain str.
    assert hole.kind == "int", (
        "native_pipeline: non-int hole %s -- no interpreter this IR "
        "currently serves has one" % (hole.name,))
    if is_marker and hole.name == "pc":
        return NIntConst(pc)
    return NIntConst(bindings[hole.name])


def _emit_moves_native(ssarepr, sources, destinations, scratch):
    """Port of ProgramEmitter._emit_moves."""
    moves = []
    for bname, destination in destinations.items():
        if destination is None:
            continue
        kind, index = destination
        # Membership-checked, not .get(bname): tuple values have no null
        # to unify with a missing-key None (see last_blocked in
        # generating_extension.py).
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

    pending = list(moves)
    emitted = []
    while pending:
        progressed = False
        for move in list(pending):
            kind, index, source = move
            # Loop, not any(genexpr): closing over move/kind/index isn't
            # RPython-legal.
            blocked = False
            for other in pending:
                # Not ``other is move``: RPython's rtyper has no identity
                # comparison for tuples. Value equality is a safe
                # substitute: 'pending' entries are one per boundary name,
                # and two boundary names never share one (kind, index,
                # source) triple, so equal means the same entry.
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

def compute_liveness_native(insns):
    label2alive = {}
    while _compute_liveness_native_pass(insns, label2alive):
        pass
    _remove_repeated_live_native(insns)


def _follow_label_native(label_id, label2alive, alive):
    """Module-level, not a nested closure (RPython disallows closures);
    takes alive/label2alive explicitly instead."""
    alive_at_point = label2alive.get(label_id)
    if alive_at_point is not None:
        for nid, reg in alive_at_point.items():
            alive[nid] = reg


def _mark_native(x, label2alive, alive):
    """Plain function, not a nested closure -- see _follow_label_native."""
    if isinstance(x, NReg):
        alive[x.nid] = x
    elif isinstance(x, NListOfKind):
        for item in x.items:
            if isinstance(item, NReg):
                alive[item.nid] = item
    elif isinstance(x, NTLabel):
        _follow_label_native(x.label_id, label2alive, alive)
    elif isinstance(x, NDescr):
        # Local var first: RPython's isinstance-narrowing tracks a plain
        # variable, not a re-evaluated x.descr attribute access.
        descr = x.descr
        if isinstance(descr, NativeSwitchDictDescr):
            # label is already the int label id (see NativeSwitchDictDescr).
            for _key, label in descr._native_labels:
                _follow_label_native(label, label2alive, alive)
        # A real (non-native) SwitchDictDescr can also reach here, carried
        # unchanged from an already-compiled template fragment -- its
        # targets got liveness during that earlier compile, so this pass
        # has nothing sound to add for it.


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
    # Not insns[:] = res: RPython's rlist has no unbounded full-replace
    # slice assignment (rlist.py's rtype_setslice). pop()+extend() mutate
    # 'insns' in place instead, on purpose, so the caller sees the result.
    while insns:
        insns.pop()
    insns.extend(res)


# ____________________________________________________________
# NativeAssembler: port of codewriter/assembler.py's Assembler.

def _get_liveness_info_native(operands, kind):
    # Dict-as-set: RPython has no native set type (see
    # generating_extension.py, partialeval_template.py).
    lives = {}
    for x in operands:
        if isinstance(x, NReg) and x.kind == kind:
            lives[chr(x.index)] = True
    return lives


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

    ``readonly``, when True, is the mode a genuine runtime_cogen caller
    uses (PortalLinker._emit_native, portal_linker.py): ``insns`` stays
    shared (read-only lookups against what precompile_fragments already
    built), but write_insn no longer grows it or ``descrs`` for an
    ordinary (non-switch-dict) descr. ``all_liveness``/
    ``all_liveness_positions``/``_counters`` are not shared in this mode:
    this assembler keeps its own fresh copies, so a late JitCode's
    ``-live-`` offsets land in its own liveness chunk (stored as
    ``JitCode.own_liveness_info``) instead of the frozen global string,
    which nothing may resync any more.
    """

    def __init__(self, share_with=None, readonly=False):
        Assembler.__init__(self)
        # Per-JitCode, like switchdictdescrs; reset again in assemble()
        # since inherited setup() doesn't know about this list.
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
        # jitcode._ssarepr left unset: it's debug-only (JitCode.dump,
        # skipped below since format_assembler can't read native operands).
        self.make_jitcode(jitcode)
        # ponytail: no jitcode._dump -- format_assembler expects real
        # SSARepr tuples and would crash on native operands; a full port
        # would teach it to read NativeInsn too.
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
                argcodes.append(x.kind[0])
            elif isinstance(x, NIntConst):
                is_short = self.emit_resolved_const(x.ivalue, "int",
                                                     allow_short=allow_short)
                argcodes.append("c" if is_short else "i")
            elif isinstance(x, NRefConst):
                # No dedup_key: default (=value) is right now that
                # constants_dict_r is keyed via new_ref_dict's rd_eq/
                # rd_hash (assembler.py).
                self.emit_resolved_const(x.value, "ref")
                argcodes.append("r")
            elif isinstance(x, NFloatConst):
                self.emit_resolved_const(x.value, "float")
                argcodes.append("f")
            elif isinstance(x, NTLabel):
                self.alllabels[len(self.code)] = True
                self.tlabel_positions.append((x.label_id, len(self.code)))
                self.code.append("temp 1")
                self.code.append("temp 2")
                argcodes.append("L")
            elif isinstance(x, NListOfKind):
                lst = x.items
                assert len(lst) <= 255, "list too long!"
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
                        # See the other emit_resolved_const(..., "ref")
                        # call above: no explicit dedup_key needed either.
                        self.emit_resolved_const(item.value, "ref")
                    elif isinstance(item, NFloatConst):
                        assert x.kind == "float"
                        self.emit_resolved_const(item.value, "float")
                    else:
                        # Not %r, and not embedding 'item' at all: an
                        # arbitrary operand instance has no RPython-legal
                        # string conversion.
                        raise NotImplementedError(
                            "found an operand of an unexpected type in "
                            "NListOfKind()")
                argcodes.append(x.kind[0].upper())
            elif isinstance(x, NDescr):
                d = x.descr
                if isinstance(d, NativeSwitchDictDescr):
                    # Genuinely fresh every placement (_localise_operand,
                    # this module): its jump targets depend on *this*
                    # program's own label layout, so it can never have been
                    # stamped ahead of time the way an ordinary descr is
                    # (see the ``elif self._readonly`` branch below) -- it
                    # still grows the shared list by append, exactly as
                    # before.  Not a decline case: no precompiled fragment
                    # could ever have "already covered" an object that
                    # doesn't exist until this call.
                    if d not in self._descr_dict:
                        self._descr_dict[d] = len(self.descrs)
                        self.descrs.append(d)
                    # NativeSwitchDictDescr, not SwitchDictDescr: this
                    # path's own switch descrs go to their own list,
                    # resolved by fix_labels's own override below -- see
                    # NativeSwitchDictDescr's docstring for why.
                    self.native_switchdictdescrs.append(d)
                    num = self._descr_dict[d]
                elif self._readonly:
                    # An ordinary (translation-time-prebuilt) descr: read
                    # the index ProgramEmitter.native_table (jitcode_
                    # emitter.py) already stamped on it, instead of
                    # appending to the shared descrs list -- see
                    # AbstractDescr.pe_descr_index's own note (history.py)
                    # for why a stamp, not an object-keyed dict lookup, at
                    # runtime.  -1 means this descr was never covered by any
                    # precompiled fragment: decline the whole program, same
                    # as an uncovered insn key (_insn_number below).
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
                argcodes.append("d")
            elif isinstance(x, NIndirectCallTargets):
                for target in x.lst:
                    self.indirectcalltargets[target] = True
            elif isinstance(x, NHole):
                # Not %r: see _patch_hole_native's own note above -- same
                # reason, same fix (x.name is already a plain str).
                raise AssertionError(
                    "unpatched hole %s reached the assembler" % (x.name,))
            else:
                raise NotImplementedError(x)

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
        """The opcode byte for 'key' (an "opname/argcodes" string).

        Growth-allowed mode (the default): identical to Assembler.write_insn
        -- hands out a fresh number one past the current end the first time
        a key is seen.  Readonly mode (see __init__'s own docstring): a
        genuine runtime_cogen caller's insns table is exactly what
        precompile_fragments already built at translation time (shared by
        reference, never copied -- __init__), so a lookup miss here means
        this program needs an (opname, argcodes) combination no precompiled
        fragment ever used -- decline the whole program (the exception
        unwinds through NativeAssembler.assemble/emit_and_assemble_native to
        generate_for_live_code's own ``except Exception: return None``)
        rather than mint a new opcode number nothing may fold back into
        MetaInterpStaticData.opcode_implementations/opcode_names any more.
        """
        if self._readonly:
            num = self.insns.get(key, -1)
            if num < 0:
                debug_print("runtime cogen: no precompiled insn for", key)
                # Not %r: see _patch_hole_native's own note -- same reason,
                # same fix (key is already a plain str).
                raise AssemblerError("no precompiled insn for %s" % (key,))
            return num
        return self.insns.setdefault(key, len(self.insns))

    def fix_labels(self):
        """Override: Assembler.fix_labels reads switchdictdescrs'
        label.name, never populated here (see NativeSwitchDictDescr).
        Resolves tlabel_positions the same way, plus
        native_switchdictdescrs instead of switchdictdescrs.
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


def emit_and_assemble_native(native_table, program, name,
                             has_merge_points=False, assembler=None):
    """Full native pipeline: emit_native -> compute_liveness_native ->
    NativeAssembler.assemble, in that order (place everything, then
    compute liveness once over the flat stream, then assemble).

    Returns (jitcode, entry_positions, assembler).
    """
    ssarepr, counts = emit_native(native_table, program, name, has_merge_points)
    compute_liveness_native(ssarepr.insns)
    if assembler is None:
        assembler = NativeAssembler()
    jitcode = JitCode(name, fnaddr=llmemory.NULL)
    assembler.assemble(ssarepr, jitcode, counts)
    # Loop, not dict(genexpr): unlike emit_native's genexprs (RPython
    # inlines those), this one closes over assembler, which isn't legal.
    entry_positions = {}
    for pc in program.blocks:
        entry_positions[pc] = assembler.label_positions[_block_label_id(pc)]
    return jitcode, entry_positions, assembler
