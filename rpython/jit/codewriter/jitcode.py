from __future__ import print_function

import os

from rpython.jit.metainterp.history import AbstractDescr, ConstInt, new_ref_dict
from rpython.jit.metainterp.support import adr2int
from rpython.rlib.debug import debug_start, debug_stop, debug_print
from rpython.rlib.objectmodel import we_are_translated
from rpython.rlib.rarithmetic import intmask, specialize
from rpython.rlib.rarithmetic import base_int
from rpython.rtyper.lltypesystem import llmemory, lltype


def _never_matches(gcref):
    return False


def _pc_in(pcs, pc):
    pc = intmask(pc)
    for covered in pcs:
        if covered == pc:
            return True
    return False


# Holder, not a [0]-list: RPython would fold a prebuilt list's element.
class _LateJitcodeCounter(object):
    def __init__(self):
        self.base = 0
        self.next_index = 0


_late_jitcode_counter = _LateJitcodeCounter()
_late_jitcodes_by_index = {}


class _CogenCounters(object):
    def __init__(self):
        self.generated = 0
        self.declined = 0
        self.deferred = 0


_cogen_counters = _CogenCounters()


COGEN_RETRY_MAX_DELAY = 256


def set_late_jitcode_base(count):
    _late_jitcode_counter.base = count
    _late_jitcode_counter.next_index = count


def get_late_jitcode(index):
    # Dead code path; keeps the annotator's dict value type as JitCode.
    from rpython.rlib.nonconst import NonConstant
    if NonConstant(False):
        _late_jitcodes_by_index[-1] = JitCode(
            "late-jitcode-type-hint", fnaddr=llmemory.NULL)
    return _late_jitcodes_by_index[index]


def register_late_jitcode(jitcode, own_liveness_info):
    if jitcode.own_liveness_info is None:
        jitcode.own_liveness_info = own_liveness_info
    jitcode.index = _late_jitcode_counter.next_index
    assert jitcode.index not in _late_jitcodes_by_index, (
        "register_late_jitcode: _late_jitcode_counter produced an "
        "index already in use -- the monotonic counter did not advance "
        "between two registrations")
    _late_jitcode_counter.next_index += 1
    _late_jitcodes_by_index[jitcode.index] = jitcode


def _signed_pcs(pcs):
    return [intmask(pc) for pc in pcs]


class PELinkedProgram(object):
    """One runtime-linked JitCode, and the portal entry it stands for."""

    def __init__(self, jitcode, argument_sources, argument_constants):
        self.jitcode = jitcode
        self.code_size = len(jitcode.code)
        # Lists, not tuples: RPython can't unify tuples of differing length.
        self.argument_sources = list(argument_sources)
        self.argument_constants = list(argument_constants)
        self.match_pc_index = -1
        self.match_pcs = []
        self.match_ref_index = -1
        self.match_ref = lltype.nullptr(llmemory.GCREF.TO)
        # Valid trace-start subset of match_pcs: loop headers + entry pc.
        self.legit_entry_pcs = []
        self.leave_pcs = []
        # Loop-less programs are leaves: CALL_ASSEMBLER would force virtuals.
        self.has_loops = False
        self.matcher = _never_matches

    def set_matcher(self, pc_index, pcs, ref_index, legit_entry_pcs,
                     has_loops, leave_pcs=[]):
        self.has_loops = has_loops
        self.match_pc_index = pc_index
        self.match_pcs = _signed_pcs(pcs)
        self.match_ref_index = ref_index
        self.legit_entry_pcs = _signed_pcs(legit_entry_pcs)
        self.leave_pcs = _signed_pcs(leave_pcs)

    def is_leave_pc(self, pc):
        return _pc_in(self.leave_pcs, pc)

    def _covers(self, pc):
        return _pc_in(self.match_pcs, pc)

    def is_legit_entry_pc(self, pc):
        return _pc_in(self.legit_entry_pcs, pc)

    def start_position(self, boxes):
        index = self.match_pc_index
        if index < 0:
            return 0
        metadata = self.jitcode.pe_metadata
        if metadata is None:
            return 0
        return metadata.position_for_pc(boxes[index].getint())

    def build_call_boxes(self, boxes):
        """Map a caller's greens+reds onto this program's argument layout."""
        call_boxes = []
        constant_index = 0
        for source in self.argument_sources:
            if source >= 0:
                call_boxes.append(boxes[source])
            else:
                const = self.argument_constants[constant_index]
                call_boxes.append(ConstInt(const))
                constant_index += 1
        return call_boxes

    def matches(self, boxes):
        index = self.match_pc_index
        if index >= 0 and not self._covers(boxes[index].getint()):
            return False
        index = self.match_ref_index
        if index < 0:
            return True
        return self.matches_ref(boxes[index].getref_base())

    def matches_ref(self, actual):
        """Ref-only half of matches(): does this program own 'actual'?"""
        if self.match_ref_index < 0:
            return True
        expected = self.match_ref
        if not expected:
            if not self.matcher(actual):
                return False
            self.match_ref = actual
            return True
        return actual == expected


class PEJitCodeMetadata(object):
    """RPython-friendly runtime-cogen facts attached to a JitCode."""

    def __init__(self, entry_pc, block_pcs, loop_headers, backedge_sources,
                 backedge_targets, entry_pcs, entry_positions):
        # intmask: guest pcs may be unsigned; these tables are signed.
        self.entry_pc = intmask(entry_pc)
        self.block_pcs = _signed_pcs(block_pcs)
        self.loop_headers = _signed_pcs(loop_headers)
        self.backedge_sources = _signed_pcs(backedge_sources)
        self.backedge_targets = _signed_pcs(backedge_targets)
        self.entry_pcs = _signed_pcs(entry_pcs)
        self.entry_positions = list(entry_positions)
        self.linked_programs = []
        self.entry_position = self.position_for_pc(entry_pc)
        self.owns_linked_jitcode = False
        self.has_merge_points = False
        # ref -> program cache for linked_program_for; built lazily.
        self._program_cache = None
        self._miss_counts = None
        # ref -> next miss count at which a soft decline is retried.
        self._retry_at = None
        # Misses needed before runtime_cogen runs for a ref; 0 = first miss.
        self.cogen_threshold = 0
        self.threshold_env_var = None
        self._threshold_env_read = False
        # f(gcref) -> program or None; called once per ref.
        self.runtime_cogen = None
        self.match_ref_index = -1
        self.match_pc_index = -1
        # soft_decline: True means retry later instead of caching None.
        self.soft_decline = False

    def attach_linked_jitcode(self, jitcode, argument_sources,
                              argument_constants):
        """Add an unguarded program -- for a portal with one code object."""
        program = PELinkedProgram(jitcode, argument_sources,
                                  argument_constants)
        self.linked_programs.append(program)
        if jitcode.pe_metadata is self:
            self.owns_linked_jitcode = True
        jitcode.pe_is_linked = True
        # Portal's own program wins over the linked JitCode's self-attach.
        if argument_sources or jitcode.pe_program is None:
            jitcode.pe_program = program
        return program

    def has_linked_programs(self):
        return len(self.linked_programs) > 0

    def linked_program_for(self, boxes):
        """The program linked for the code object the portal is entering."""
        programs = self.linked_programs
        ref_index = -1
        if programs and programs[0].match_ref_index >= 0:
            ref_index = programs[0].match_ref_index
        elif not programs and self.runtime_cogen is not None:
            ref_index = self.match_ref_index
        if ref_index >= 0:
            ref = boxes[ref_index].getref_base()
            if ref:
                cache = self._program_cache
                if cache is None:
                    cache = self._program_cache = new_ref_dict()
                if ref in cache:
                    program = cache[ref]
                else:
                    program = self._resolve_ref(ref)
                    if program is None and self.runtime_cogen is not None:
                        if not self._miss_count_reached(ref):
                            # below threshold: don't cache a permanent None
                            return None
                        program = self._cogen_ref(ref)
                        if program is None and self.soft_decline:
                            # soft decline: leave uncached, retry later
                            return None
                    cache[ref] = program
                if program is None:
                    return None
                index = program.match_pc_index
                if index >= 0 and not program._covers(boxes[index].getint()):
                    return None
                return program
        for program in programs:
            if program.matches(boxes):
                return program
        return None

    def installed_program_for_ref(self, ref):
        """Already-generated program for 'ref', or None; never generates."""
        if not ref:
            return None
        cache = self._program_cache
        if cache is not None and ref in cache:
            return cache[ref]
        for program in self.linked_programs:
            if program.match_ref and program.match_ref == ref:
                return program
        return None

    def _resolve_ref(self, ref):
        """Which program owns 'ref', ignoring the pc guard."""
        for program in self.linked_programs:
            if program.matches_ref(ref):
                return program
        return None

    def _miss_count_reached(self, ref):
        """Has 'ref' missed cogen_threshold times (threshold=0: first miss)?"""
        if not self._threshold_env_read:
            self._threshold_env_read = True
            env_var = self.threshold_env_var
            if env_var is not None:
                value = os.environ.get(env_var)
                if value:
                    try:
                        self.cogen_threshold = int(value)
                    except ValueError:
                        pass
        counts = self._miss_counts
        if counts is None:
            counts = self._miss_counts = new_ref_dict()
        if ref in counts:
            count = counts[ref] + 1
        else:
            count = 1
        counts[ref] = count
        retry_at = self._retry_at
        if retry_at is not None and ref in retry_at:
            return count >= retry_at[ref]
        return count >= self.cogen_threshold

    def _defer_ref(self, ref):
        """Back off retries after a temporary runtime-cogen decline."""
        count = self._miss_counts[ref]
        delay = count
        if delay < self.cogen_threshold:
            delay = self.cogen_threshold
        if delay < 1:
            delay = 1
        if delay > COGEN_RETRY_MAX_DELAY:
            delay = COGEN_RETRY_MAX_DELAY
        retry_at = self._retry_at
        if retry_at is None:
            retry_at = self._retry_at = new_ref_dict()
        retry_at[ref] = count + delay

    def _cogen_ref(self, ref):
        """Invoked once per ref; linked_program_for caches the result."""
        debug_start("pe-cogen")
        try:
            program = self.runtime_cogen(ref)
            if program is None or program.match_ref != ref:
                program = None
            if program is None and self.soft_decline:
                self._defer_ref(ref)
                _cogen_counters.deferred += 1
                debug_print("pe-cogen ref=%d deferred "
                            "totals-generated=%d totals-declined=%d "
                            "totals-deferred=%d" % (
                    lltype.cast_ptr_to_int(ref), _cogen_counters.generated,
                    _cogen_counters.declined, _cogen_counters.deferred))
            else:
                if program is None:
                    generated = 0
                    _cogen_counters.declined += 1
                else:
                    generated = 1
                    _cogen_counters.generated += 1
                debug_print("pe-cogen ref=%d generated=%d "
                            "totals-generated=%d totals-declined=%d" % (
                    lltype.cast_ptr_to_int(ref), generated,
                    _cogen_counters.generated, _cogen_counters.declined))
        finally:
            debug_stop("pe-cogen")
        return program

    def is_linked_jitcode(self, jitcode):
        return jitcode.pe_is_linked

    def is_loop_header(self, pc):
        return intmask(pc) in self.loop_headers

    def position_for_pc(self, pc):
        # Linear scan: runs once per trace start, rare enough to skip a dict.
        pc = intmask(pc)
        for index in range(len(self.entry_pcs)):
            if self.entry_pcs[index] == pc:
                return self.entry_positions[index]
        return 0


class JitCode(AbstractDescr):
    _empty_i = []
    _empty_r = []
    _empty_f = []

    def __init__(self, name, fnaddr=None, calldescr=None, called_from=None):
        self.name = name
        self.fnaddr = fnaddr
        self.calldescr = calldescr
        self.jitdriver_sd = None # None for non-portals
        self.pe_metadata = None
        self.pe_is_linked = False # set True by attach_linked_jitcode
        self.pe_program = None    # its PELinkedProgram, when linked
        self._called_from = called_from   # debugging
        self._ssarepr     = None          # debugging
        # None: global liveness_info; set only for a runtime JitCode's own.
        self.own_liveness_info = None

    def setup(self, code='', constants_i=[], constants_r=[], constants_f=[],
              num_regs_i=255, num_regs_r=255, num_regs_f=255,
              startpoints=None, alllabels=None,
              resulttypes=None):
        self.code = code
        for x in constants_i:
            assert not isinstance(x, base_int), (
                "found constant %r of type %r, must not appear in "
                "JitCode.constants_i" % (x, type(x)))
        # if the following lists are empty, use a single shared empty list
        self.constants_i = constants_i or self._empty_i
        self.constants_r = constants_r or self._empty_r
        self.constants_f = constants_f or self._empty_f
        # encode the three num_regs into a single char each
        assert num_regs_i < 256 and num_regs_r < 256 and num_regs_f < 256
        self.c_num_regs_i = chr(num_regs_i)
        self.c_num_regs_r = chr(num_regs_r)
        self.c_num_regs_f = chr(num_regs_f)
        self._startpoints = startpoints   # debugging
        self._alllabels = alllabels       # debugging
        self._resulttypes = resulttypes   # debugging

    def get_fnaddr_as_int(self):
        return adr2int(self.fnaddr)

    def num_regs_i(self):
        return ord(self.c_num_regs_i)

    def num_regs_r(self):
        return ord(self.c_num_regs_r)

    def num_regs_f(self):
        return ord(self.c_num_regs_f)

    def num_regs_and_consts_i(self):
        return ord(self.c_num_regs_i) + len(self.constants_i)

    def num_regs_and_consts_r(self):
        return ord(self.c_num_regs_r) + len(self.constants_r)

    def num_regs_and_consts_f(self):
        return ord(self.c_num_regs_f) + len(self.constants_f)


    def _live_vars(self, pc, all_liveness, op_live):
        from rpython.jit.codewriter.liveness import LivenessIterator
        # for testing only
        if ord(self.code[pc]) != op_live:
            self._missing_liveness(pc)
        offset = self.get_live_vars_info(pc, op_live)
        lst_i = []
        lst_r = []
        lst_f = []
        enumerate_vars(offset, all_liveness,
                lambda index: lst_i.append("%%i%d" % (index, )),
                lambda index: lst_r.append("%%r%d" % (index, )),
                lambda index: lst_f.append("%%f%d" % (index, )),
                None)
        return ' '.join(lst_i + lst_r + lst_f)

    def get_live_vars_info(self, pc, op_live):
        from rpython.jit.codewriter.liveness import decode_offset, OFFSET_SIZE
        # either this, or the previous instruction must be -live-
        if not we_are_translated():
            assert pc in self._startpoints
        if ord(self.code[pc]) != op_live:
            pc -= OFFSET_SIZE + 1
            if not we_are_translated():
                assert pc in self._startpoints
            if ord(self.code[pc]) != op_live:
                self._missing_liveness(pc)
        return decode_offset(self.code, pc + 1)

    def _missing_liveness(self, pc):
        msg = "missing liveness[%d] in %s" % (pc, self.name)
        if we_are_translated():
            print(msg)
            raise AssertionError
        raise MissingLiveness("%s\n%s" % (msg, self.dump()))

    def follow_jump(self, position):
        """Assuming that 'position' points just after a bytecode
        instruction that ends with a label, follow that label."""
        code = self.code
        position -= 2
        assert position >= 0
        if not we_are_translated():
            assert position in self._alllabels
        labelvalue = ord(code[position]) | (ord(code[position+1])<<8)
        assert labelvalue < len(code)
        return labelvalue

    def pe_loop_header_position(self):
        """Where a jump means a runtime-linked loop's back edge, or -1."""
        metadata = self.pe_metadata
        if (metadata is None or not metadata.owns_linked_jitcode
                or metadata.has_merge_points):
            return -1
        return metadata.entry_position

    def dump(self):
        if self._ssarepr is None:
            return '<no dump available for %r>' % (self.name,)
        else:
            from rpython.jit.codewriter.format import format_assembler
            return format_assembler(self._ssarepr)

    def __repr__(self):
        return '<JitCode %r>' % self.name

    def _clone_if_mutable(self):
        raise NotImplementedError

class MissingLiveness(Exception):
    pass


class SwitchDictDescr(AbstractDescr):
    "Get a 'dict' attribute mapping integer values to bytecode positions."

    def attach(self, as_dict):
        self.dict = as_dict
        # map()/sorted() aren't RPython-legal; loops + O(n^2) insertion sort.
        keys = []
        for key in as_dict:
            keys.append(key)
        index = 1
        while index < len(keys):
            key = keys[index]
            gap = index - 1
            while gap >= 0 and keys[gap] > key:
                keys[gap + 1] = keys[gap]
                gap -= 1
            keys[gap + 1] = key
            index += 1
        const_keys = []
        for key in keys:
            const_keys.append(ConstInt(key))
        self.const_keys_in_order = const_keys

    def __repr__(self):
        dict = getattr(self, 'dict', '?')
        return '<SwitchDictDescr %s>' % (dict,)

    def _clone_if_mutable(self):
        raise NotImplementedError


@specialize.arg(5)
def enumerate_vars(offset, all_liveness, callback_i, callback_r, callback_f, spec):
    from rpython.jit.codewriter.liveness import LivenessIterator
    length_i = ord(all_liveness[offset])
    length_r = ord(all_liveness[offset + 1])
    length_f = ord(all_liveness[offset + 2])
    offset += 3
    if length_i:
        it = LivenessIterator(offset, length_i, all_liveness)
        for index in it:
            callback_i(index)
        offset = it.offset
    if length_r:
        it = LivenessIterator(offset, length_r, all_liveness)
        for index in it:
            callback_r(index)
        offset = it.offset
    if length_f:
        it = LivenessIterator(offset, length_f, all_liveness)
        for index in it:
            callback_f(index)


def dump_jitcode(jitcode, metainterp_sd):
    """PYPYLOG=jit-jitcode-dump: one line per instruction of 'jitcode'."""
    from rpython.rlib.debug import have_debug_prints
    debug_start("jit-jitcode-dump")
    if have_debug_prints():
        _dump_jitcode(jitcode, metainterp_sd)
    debug_stop("jit-jitcode-dump")


def _operand(jitcode, kind, index):
    if kind == 'i':
        if index < jitcode.num_regs_i():
            return "%%i%d" % index
        return "$%d" % jitcode.constants_i[index - jitcode.num_regs_i()]
    if kind == 'r':
        if index < jitcode.num_regs_r():
            return "%%r%d" % index
        ref = jitcode.constants_r[index - jitcode.num_regs_r()]
        return "$ref(0x%x)" % lltype.cast_ptr_to_int(ref)
    if index < jitcode.num_regs_f():
        return "%%f%d" % index
    return "$f%d" % (index - jitcode.num_regs_f())


def _pad5(n):
    s = str(n)
    while len(s) < 5:
        s = " " + s
    return s


def _dump_jitcode(jitcode, metainterp_sd):
    from rpython.jit.metainterp.blackhole import signedord
    code = jitcode.code
    names = metainterp_sd.opcode_names
    descrs = metainterp_sd.opcode_descrs
    debug_print("jitcode %s: %d bytes, regs i=%d r=%d f=%d, "
                "consts i=%d r=%d f=%d" % (
        jitcode.name, len(code), jitcode.num_regs_i(),
        jitcode.num_regs_r(), jitcode.num_regs_f(),
        len(jitcode.constants_i), len(jitcode.constants_r),
        len(jitcode.constants_f)))
    pos = 0
    while pos < len(code):
        start = pos
        opcode = ord(code[pos])
        pos += 1
        if opcode == metainterp_sd.op_live:
            offset = ord(code[pos]) | (ord(code[pos + 1]) << 8)
            pos += 2
            debug_print("%s: -live- @%d" % (_pad5(start), offset))
            continue
        key = names[opcode]
        slash = key.find('/')
        assert slash >= 0
        name = key[:slash]
        argcodes = key[slash + 1:]
        parts = []
        i = 0
        while i < len(argcodes):
            c = argcodes[i]
            if c == '>':
                parts.append("-> " + _operand(jitcode, argcodes[i + 1],
                                              ord(code[pos])))
                pos += 1
                i += 2
                continue
            if c == 'i' or c == 'r' or c == 'f':
                parts.append(_operand(jitcode, c, ord(code[pos])))
                pos += 1
            elif c == 'c':
                parts.append(str(signedord(code[pos])))
                pos += 1
            elif c == 'L':
                parts.append("L%d" % (ord(code[pos]) |
                                      (ord(code[pos + 1]) << 8)))
                pos += 2
            elif c == 'I' or c == 'R' or c == 'F':
                length = ord(code[pos])
                pos += 1
                items = []
                for k in range(length):
                    items.append(_operand(jitcode, c.lower(),
                                          ord(code[pos])))
                    pos += 1
                parts.append("[" + ", ".join(items) + "]")
            elif c == 'd':
                index = ord(code[pos]) | (ord(code[pos + 1]) << 8)
                pos += 2
                descr = descrs[index]
                if isinstance(descr, JitCode):
                    parts.append("<JitCode %s>" % descr.name)
                else:
                    parts.append(descr.repr_of_descr())
            else:
                parts.append("?" + c)
            i += 1
        debug_print("%s: %s %s" % (_pad5(start), name, " ".join(parts)))
