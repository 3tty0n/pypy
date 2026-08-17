"""Runtime-shaped ports of ProgramEmitter.emit / compute_liveness / Assembler,
operating on the native_fragments.py IR instead of SSARepr/flowspace objects.

Mirrors each original 1:1 on purpose: test_native_pipeline.py's equivalence
gate requires byte-identical output, so keep behavior identical here, not
just "cleaner".
"""

from rpython.jit.codewriter.assembler import Assembler, USE_C_FORM
from rpython.jit.codewriter.flatten import KINDS
from rpython.jit.codewriter.jitcode import JitCode, SwitchDictDescr
from rpython.jit.metainterp.history import AbstractDescr
from rpython.rtyper.lltypesystem import llmemory, lltype

from rpython.translator.backendopt.native_fragments import (
    NReg, NIntConst, NRefConst, NFloatConst, NHole, NLabel,
    NTLabel, NDescr, NListOfKind, NIndirectCallTargets, NSwitchDictOperand,
    NativeInsn, native_fragment_for)


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


class _NameHolder(object):
    """Duck-types as a TLabel for fix_labels's switchlabel.name read."""
    def __init__(self, label_id):
        self.name = label_id


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
    headers = set()
    if has_merge_points:
        headers = set(program.loop_headers) | set([program.entry_pc])
    fragments = dict(
        (pc, native_fragment_for(native_table, block.key, pc in headers))
        for pc, block in program.blocks.items())

    num_regs = dict((kind, max([0] + [f.num_regs.get(kind, 0)
                                       for f in fragments.values()]))
                    for kind in KINDS)
    scratch = dict((kind, count) for kind, count in num_regs.items())
    counts = dict((kind, scratch[kind] + 1) for kind in scratch)

    ssarepr = NativeSSARepr(name)
    order = [program.entry_pc] + sorted(
        pc for pc in program.blocks if pc != program.entry_pc)
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
    for kind in sorted(counts):
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
    targets = block.template.resolve_targets(block.bindings)
    if len(targets) == 1 and len(fragment.exits) > 1:
        targets = targets * len(fragment.exits)

    for insn in fragment.insns:
        exit_index = _exit_index(insn)
        if exit_index is None:
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
    return None


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
        fresh = SwitchDictDescr()
        fresh._labels = [
            (key, _NameHolder(_fragment_label_id(pc, label_id)))
            for key, label_id in zip(x.keys, x.label_ids)]
        return NDescr(fresh)
    return x   # NReg, NIntConst, NRefConst, NFloatConst, NDescr: placement-
               # invariant, copied verbatim


def _patch_hole_native(hole, pc, bindings, is_marker):
    """A marker's own 'pc' hole identifies this block; every other hole
    (including a Continue exit's own next-pc) takes the bound value.
    Every hole here is a plain int; asserted since silently truncating a
    ref/float hole would be a correctness bug, not just an RPython one.
    """
    assert hole.kind == "int", (
        "native_pipeline: non-int hole %r -- no interpreter this IR "
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
        source = sources.get(bname)
        if source is None:
            if kind == "ref":
                source = NRefConst(lltype.nullptr(llmemory.GCREF.TO))
            else:
                # Mirrors the original's quirk: a float-kind destination
                # with no source still gets an int-kind zero constant.
                source = NIntConst(0)
        elif isinstance(source, tuple):
            if source == destination:
                continue
            source = _register(*source)
        moves.append((kind, index, source))

    pending = list(moves)
    emitted = []
    while pending:
        progressed = False
        for move in list(pending):
            kind, index, source = move
            blocked = any(
                isinstance(other[2], NReg) and
                other[2].kind == kind and other[2].index == index
                for other in pending if other is not move)
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


def _compute_liveness_native_pass(insns, label2alive):
    alive = {}     # nid -> NReg
    must_continue = [False]

    def follow_label(label_id):
        alive_at_point = label2alive.get(label_id)
        if alive_at_point is not None:
            for nid, reg in alive_at_point.items():
                alive[nid] = reg

    def mark(x):
        if isinstance(x, NReg):
            alive[x.nid] = x
        elif isinstance(x, NListOfKind):
            for item in x.items:
                if isinstance(item, NReg):
                    alive[item.nid] = item
        elif isinstance(x, NTLabel):
            follow_label(x.label_id)
        elif isinstance(x, NDescr) and isinstance(x.descr, SwitchDictDescr):
            for _key, label in x.descr._labels:
                follow_label(label.name)

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
                must_continue[0] = True
            continue

        if insn.opcode == "-live-":
            labels = []
            for x in insn.operands:
                if isinstance(x, NReg):
                    alive[x.nid] = x
                elif isinstance(x, NTLabel):
                    follow_label(x.label_id)
                    labels.append(x)
            insns[i] = NativeInsn("-live-", alive.values() + labels)
            continue

        if insn.opcode == "---":
            alive = {}
            continue

        if insn.result is not None and insn.result.nid in alive:
            del alive[insn.result.nid]
        for x in insn.operands:
            mark(x)

    return must_continue[0]


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
    insns[:] = res


# ____________________________________________________________
# NativeAssembler: port of codewriter/assembler.py's Assembler.

def _ref_dedup_key(value):
    """The dedup key for one 'ref'-kind resolved value, mirroring
    ``Assembler.emit_const_value``'s own 'ref' branch exactly (``None`` for
    a null GCREF, its container object otherwise -- a GCREF pointer itself
    disables ``__hash__``, see ``emit_resolved_const``'s own 'ref' comment,
    so it cannot be the dedup key directly).  ``._obj``/``.container`` are
    untranslated-lltype-simulation-only, like the identical computation in
    ``emit_const_value`` -- see that method's docstring; kept here, not
    reused from there, only because NRefConst's ``.value`` already carries
    the post-cast GCREF, needing no separate pre-cast step to reproduce."""
    if not value:
        return None
    return value._obj.container


def _get_liveness_info_native(operands, kind):
    lives = set()
    for x in operands:
        if isinstance(x, NReg) and x.kind == kind:
            lives.add(chr(x.index))
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
    """

    def __init__(self, share_with=None):
        Assembler.__init__(self)
        self._share_with = share_with
        if share_with is not None:
            self.descrs = share_with.descrs
            self._descr_dict = share_with._descr_dict
            self.insns = share_with.insns
            self.indirectcalltargets = share_with.indirectcalltargets
            self.list_of_addr2name = share_with.list_of_addr2name
            self._seen_raw_objects = share_with._seen_raw_objects
            self._counters = share_with._counters
            self.all_liveness = share_with.all_liveness
            self.all_liveness_positions = share_with.all_liveness_positions

    def assemble(self, ssarepr, jitcode=None, num_regs=None):
        self.setup(ssarepr.name)
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
        jitcode._ssarepr = ssarepr
        self.make_jitcode(jitcode)
        # ponytail: no jitcode._dump here -- format_assembler assumes real
        # SSARepr tuples (Register/Label/... instances) and would crash on
        # native operand objects.  _dump is debug-only (JitCode.dump()); a
        # real port would teach format_assembler to also read NativeInsn.
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
            self.startpoints.add(key)
            self.num_liveness_ops += 1
            live_i = _get_liveness_info_native(insn.operands, "int")
            live_r = _get_liveness_info_native(insn.operands, "ref")
            live_f = _get_liveness_info_native(insn.operands, "float")
            assert key not in self.liveness
            self.liveness[key] = live_i, live_r, live_f
            num = self.insns.setdefault("live/", len(self.insns))
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
                self.emit_resolved_const(x.value, "ref",
                                         dedup_key=_ref_dedup_key(x.value))
                argcodes.append("r")
            elif isinstance(x, NFloatConst):
                self.emit_resolved_const(x.value, "float")
                argcodes.append("f")
            elif isinstance(x, NTLabel):
                self.alllabels.add(len(self.code))
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
                        self.emit_resolved_const(
                            item.value, "ref",
                            dedup_key=_ref_dedup_key(item.value))
                    elif isinstance(item, NFloatConst):
                        assert x.kind == "float"
                        self.emit_resolved_const(item.value, "float")
                    else:
                        raise NotImplementedError(
                            "found in NListOfKind(): %r" % (item,))
                argcodes.append(x.kind[0].upper())
            elif isinstance(x, NDescr):
                d = x.descr
                if d not in self._descr_dict:
                    self._descr_dict[d] = len(self.descrs)
                    self.descrs.append(d)
                if isinstance(d, SwitchDictDescr):
                    self.switchdictdescrs.append(d)
                num = self._descr_dict[d]
                assert 0 <= num <= 0xFFFF, "too many AbstractDescrs!"
                self.code.append(chr(num & 0xFF))
                self.code.append(chr(num >> 8))
                argcodes.append("d")
            elif isinstance(x, NIndirectCallTargets):
                self.indirectcalltargets.update(x.lst)
            elif isinstance(x, NHole):
                raise AssertionError(
                    "unpatched hole %r reached the assembler" % (x.name,))
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
        num = self.insns.setdefault(key, len(self.insns))
        self.code[startposition] = chr(num)
        self.startpoints.add(startposition)


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
    entry_positions = dict(
        (pc, assembler.label_positions[_block_label_id(pc)])
        for pc in program.blocks)
    return jitcode, entry_positions, assembler
