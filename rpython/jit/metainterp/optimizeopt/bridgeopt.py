""" Code to feed information from the optimizer via the resume code into the
optimizer of the bridge attached to a guard. """

from rpython.jit.metainterp import resumecode
from rpython.jit.metainterp.history import Const, ConstInt, CONST_NULL
from rpython.jit.metainterp.optimizeopt.intutils import IntBound, MININT, MAXINT
from rpython.jit.metainterp.resoperation import rop
from .info import getptrinfo


# Range for HBP intbound inheritance.  resumecode stores signed shorts; keep
# this section compact and skip wider bounds, which the bridge can rediscover.
_INTBOUND_SHORT_MIN = -(1 << 15)
_INTBOUND_SHORT_MAX = (1 << 15) - 1


# adds the following sections at the end of the resume code:
#
# ---- known classes
# <bitfield> size is the number of reference boxes in the liveboxes
#            1 klass known
#            0 klass unknown
#            (the class is found by actually looking at the runtime value)
#            the bits are bunched in bunches of 7
#
# ---- heap knowledge
# <length>
# (<box1> <descr> <box2>) length times, if getfield(box1, descr) == box2
#                         both boxes should be in the liveboxes
#                         (or constants)
#
# <length>
# (<box1> <index> <descr> <box2>) length times, if getarrayitem_gc(box1, index, descr) == box2
#                                 both boxes should be in the liveboxes
#                                 (or constants)
#
# ---- call_loopinvariant knowledge
# <length>
# (<const> <box2>) length times, if call_loopinvariant(const) == box2
#                  box2 should be in liveboxes
#
# ---- integer-bound knowledge (pypy/pypy#5184; gated at write time by
#      warmstate.hbp_inherit >= 1 and enable_hot_bridge_promotion)
# <length>
# (<box_idx> <lower> <upper>) length times
#                             all three values are signed-short sized.
#
# ---- nullness-only knowledge (pypy/pypy#5184 phase B; gated identically)
# <length>
# (<box_idx>) length times       ref boxes that are non-null but whose
#                                class section 1 did NOT cover (i.e.,
#                                class still unknown).  When class is
#                                known the parent's bit in section 1
#                                already implies non-null, so this section
#                                stays length 0.
# ----


def _hbp_inherit_level(optimizer):
    """Return active hbp_inherit level, or 0 outside HBP inheritance."""
    jd = optimizer.jitdriver_sd
    if jd is None:
        return 0
    warmstate = jd.warmstate
    if not warmstate.enable_hot_bridge_promotion:
        return 0
    return warmstate.hbp_inherit


def _reader_has_more(reader):
    return reader.cur_pos < len(reader.code.code)


def _guard_may_use_hbp_inherit(optimizer, guard_op):
    if guard_op is None:
        return True
    warmstate = optimizer.jitdriver_sd.warmstate
    opnum = guard_op.getopnum()
    if opnum == rop.GUARD_VALUE:
        arg = guard_op.getarg(0)
        if arg.type == "i":
            if optimizer.getintbound(arg).is_bool():
                return (warmstate.enable_hbp_bool_promotion and
                        warmstate.hbp_inherit_bool)
            return warmstate.enable_hbp_value_promotion
        if arg.type == "r":
            return warmstate.enable_hbp_ref_value_promotion
        if arg.type == "f":
            return warmstate.enable_hbp_float_value_promotion
        return False
    if opnum == rop.GUARD_CLASS:
        return warmstate.enable_hbp_class_promotion
    if opnum == rop.GUARD_TRUE or opnum == rop.GUARD_FALSE:
        return (warmstate.enable_hbp_guard_bool_promotion and
                warmstate.hbp_inherit_bool)
    return False


def _collect_intbound_entries(optimizer, liveboxes, skip_boxes):
    """Return (box_idx, lower, upper) triples for liveboxes whose IntBound
    is narrower than the universe and fits signed-short resume slots."""
    entries = []
    for idx, box in enumerate(liveboxes):
        if box is None or box.type != "i":
            continue
        if box in skip_boxes:
            continue
        if idx > _INTBOUND_SHORT_MAX:
            break  # liveboxes longer than 32767; can't encode the index
        fw = box.get_forwarded()
        if fw is None or not isinstance(fw, IntBound):
            continue
        # Universe range carries no usable info; skip.
        if fw.lower == MININT and fw.upper == MAXINT:
            continue
        if (fw.lower < _INTBOUND_SHORT_MIN or
                fw.upper > _INTBOUND_SHORT_MAX):
            continue
        entries.append((idx, fw.lower, fw.upper))
    return entries


def _collect_nullness_entries(optimizer, liveboxes, skip_boxes):
    """Return livebox indices for ref boxes the parent proved non-null but
    did NOT establish a known class (the class case is already covered by
    section 1)."""
    entries = []
    for idx, box in enumerate(liveboxes):
        if box is None or box.type != "r":
            continue
        if box in skip_boxes:
            continue
        if idx > _INTBOUND_SHORT_MAX:
            break
        info = getptrinfo(box)
        if info is None:
            continue
        if info.get_known_class(optimizer.cpu) is not None:
            continue  # section 1 already flipped the class bit for this box
        if not info.is_nonnull():
            continue
        entries.append(idx)
    return entries


# maybe should be delegated to the optimization classes?

def tag_box(box, liveboxes_from_env, memo):
    if isinstance(box, Const):
        return memo.getconst(box)
    else:
        return liveboxes_from_env[box] # has to exist

def decode_box(resumestorage, tagged, liveboxes, cpu):
    from rpython.jit.metainterp.resume import untag, TAGCONST, TAGINT, TAGBOX
    from rpython.jit.metainterp.resume import NULLREF, TAG_CONST_OFFSET, tagged_eq
    num, tag = untag(tagged)
    # NB: the TAGVIRTUAL case can't happen here, because this code runs after
    # virtuals are already forced again
    if tag == TAGCONST:
        if tagged_eq(tagged, NULLREF):
            box = CONST_NULL
        else:
            box = resumestorage.rd_consts[num - TAG_CONST_OFFSET]
    elif tag == TAGINT:
        box = ConstInt(num)
    elif tag == TAGBOX:
        box = liveboxes[num]
    else:
        raise AssertionError("unreachable")
    return box

def serialize_optimizer_knowledge(optimizer, numb_state, liveboxes,
                                  liveboxes_from_env, memo, guard_op=None):
    available_boxes = {}
    for box in liveboxes:
        if box is not None and box in liveboxes_from_env:
            available_boxes[box] = None

    # class knowledge is stored as bits, true meaning the class is known, false
    # means unknown. on deserializing we look at the bits, and read the runtime
    # class for the known classes (which has to be the same in the bridge) and
    # mark that as known. this works for guard_class too: the class is only
    # known *after* the guard
    bitfield = 0
    shifts = 0
    for box in liveboxes:
        if box is None or box.type != "r":
            continue
        info = getptrinfo(box)
        known_class = info is not None and info.get_known_class(optimizer.cpu) is not None
        bitfield <<= 1
        bitfield |= known_class
        shifts += 1
        if shifts == 6:
            numb_state.append_int(bitfield)
            bitfield = shifts = 0
    if shifts:
        numb_state.append_int(bitfield << (6 - shifts))

    # heap knowledge: we store triples of known heap fields in non-virtual
    # structs
    if optimizer.optheap:
        triples_struct, triples_array = optimizer.optheap.serialize_optheap(available_boxes)
        # can only encode descrs that have a known index into
        # metainterp_sd.all_descrs
        numb_state.append_int(len(triples_struct))
        for box1, descr, box2 in triples_struct:
            descr_index = descr.get_descr_index()
            numb_state.append_short(tag_box(box1, liveboxes_from_env, memo))
            numb_state.append_int(descr_index)
            numb_state.append_short(tag_box(box2, liveboxes_from_env, memo))
        numb_state.append_int(len(triples_array))
        for box1, index, descr, box2 in triples_array:
            descr_index = descr.get_descr_index()
            numb_state.append_short(tag_box(box1, liveboxes_from_env, memo))
            numb_state.append_int(index)
            numb_state.append_int(descr_index)
            numb_state.append_short(tag_box(box2, liveboxes_from_env, memo))
    else:
        numb_state.append_int(0)
        numb_state.append_int(0)

    if optimizer.optrewrite:
        tuples_loopinvariant = optimizer.optrewrite.serialize_optrewrite(
                available_boxes)
        numb_state.append_int(len(tuples_loopinvariant))
        for constarg0, box in tuples_loopinvariant:
            numb_state.append_short(
                    tag_box(ConstInt(constarg0), liveboxes_from_env, memo))
            numb_state.append_short(tag_box(box, liveboxes_from_env, memo))
    else:
        numb_state.append_int(0)

    if (_hbp_inherit_level(optimizer) < 1 or
            not _guard_may_use_hbp_inherit(optimizer, guard_op)):
        return

    skip_boxes = {}
    if guard_op is not None:
        for i in range(guard_op.numargs()):
            box = guard_op.getarg(i)
            if not isinstance(box, Const):
                skip_boxes[box] = None

    intbound_entries = _collect_intbound_entries(
        optimizer, liveboxes, skip_boxes)
    nullness_entries = _collect_nullness_entries(
        optimizer, liveboxes, skip_boxes)
    if not intbound_entries and not nullness_entries:
        return

    # integer-bound knowledge.  Only emitted when HBP inheritance is enabled
    # and this guard has inheritable state; the default hbp_inherit=0 stream
    # stays byte-identical to bridgeopt's baseline sections.  See
    # pypy/pypy#5184.
    numb_state.append_int(len(intbound_entries))
    for idx, lower, upper in intbound_entries:
        numb_state.append_int(idx)
        numb_state.append_int(lower)
        numb_state.append_int(upper)

    # nullness-only knowledge (pypy/pypy#5184 phase B).
    numb_state.append_int(len(nullness_entries))
    for idx in nullness_entries:
        numb_state.append_int(idx)

def deserialize_optimizer_knowledge(optimizer, resumestorage, frontend_boxes,
                                    liveboxes, use_hbp_inherit=False):
    reader = resumecode.Reader(resumestorage.rd_numb)
    assert len(frontend_boxes) == len(liveboxes)
    metainterp_sd = optimizer.metainterp_sd

    # skip resume section
    startcount = reader.next_item()
    reader.jump(startcount - 1)

    # class knowledge
    bitfield = 0
    mask = 0
    for i, box in enumerate(liveboxes):
        if box.type != "r":
            continue
        if not mask:
            bitfield = reader.next_item()
            mask = 0b100000
        class_known = bitfield & mask
        mask >>= 1
        if class_known:
            cls = optimizer.cpu.cls_of_box(frontend_boxes[i])
            optimizer.make_constant_class(box, cls)

    # heap knowledge
    length = reader.next_item()
    result_struct = []
    for i in range(length):
        tagged = reader.next_item()
        box1 = decode_box(resumestorage, tagged, liveboxes, metainterp_sd.cpu)
        descr_index = reader.next_item()
        descr = metainterp_sd.all_descrs[descr_index]
        tagged = reader.next_item()
        box2 = decode_box(resumestorage, tagged, liveboxes, metainterp_sd.cpu)
        result_struct.append((box1, descr, box2))
    length = reader.next_item()
    result_array = []
    for i in range(length):
        tagged = reader.next_item()
        box1 = decode_box(resumestorage, tagged, liveboxes, metainterp_sd.cpu)
        index = reader.next_item()
        descr_index = reader.next_item()
        descr = metainterp_sd.all_descrs[descr_index]
        tagged = reader.next_item()
        box2 = decode_box(resumestorage, tagged, liveboxes, metainterp_sd.cpu)
        result_array.append((box1, index, descr, box2))
    if optimizer.optheap:
        optimizer.optheap.deserialize_optheap(result_struct, result_array)

    # call_loopinvariant knowledge
    length = reader.next_item()
    result_loopinvariant = []
    for i in range(length):
        tagged1 = reader.next_item()
        const = decode_box(resumestorage, tagged1, liveboxes, metainterp_sd.cpu)
        assert isinstance(const, ConstInt)
        i = const.getint()
        tagged2 = reader.next_item()
        box = decode_box(resumestorage, tagged2, liveboxes, metainterp_sd.cpu)
        result_loopinvariant.append((i, box))
    if optimizer.optrewrite:
        optimizer.optrewrite.deserialize_optrewrite(result_loopinvariant)

    if (not use_hbp_inherit or _hbp_inherit_level(optimizer) < 1 or
            not _reader_has_more(reader)):
        return

    # integer-bound knowledge (pypy/pypy#5184).  Older/default resume data
    # has no section here; _reader_has_more() above keeps that path valid.
    length = reader.next_item()
    for i in range(length):
        idx = reader.next_item()
        lower = reader.next_item()
        upper = reader.next_item()
        if idx < 0 or idx >= len(liveboxes):
            continue
        if lower > upper:
            continue
        box = liveboxes[idx]
        if box.type != "i":
            continue
        optimizer.setintbound(box, IntBound(lower=lower, upper=upper))

    if not _reader_has_more(reader):
        return

    # nullness-only knowledge (pypy/pypy#5184 phase B).
    length = reader.next_item()
    for i in range(length):
        idx = reader.next_item()
        if idx < 0 or idx >= len(liveboxes):
            continue
        box = liveboxes[idx]
        if box.type != "r":
            continue
        # If section 1 already installed info (which subsumes nullness),
        # leave it alone; optimizer.make_nonnull would no-op anyway.
        if box.get_forwarded() is not None:
            continue
        optimizer.make_nonnull(box)
