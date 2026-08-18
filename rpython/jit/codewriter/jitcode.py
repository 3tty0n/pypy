from __future__ import print_function

import os

from rpython.jit.metainterp.history import AbstractDescr, ConstInt, new_ref_dict
from rpython.jit.metainterp.support import adr2int
from rpython.rlib.debug import debug_start, debug_stop, debug_print
from rpython.rlib.objectmodel import we_are_translated, specialize
from rpython.rlib.rarithmetic import base_int
from rpython.rtyper.lltypesystem import llmemory, lltype


def _never_matches(gcref):
    return False


# Holder class, not a single-element list: RPython folds a prebuilt
# list's element into a literal at each read site; a holder avoids it.
class _LateJitcodeCounter(object):
    def __init__(self):
        self.base = 0
        self.next_index = 0


_late_jitcode_counter = _LateJitcodeCounter()
_late_jitcodes_by_index = {}


# Running totals for PEJitCodeMetadata._cogen_ref (below), read out via
# debug_print on every call.  A holder class instance, not a module-level
# [0]-list: see _LateJitcodeCounter's own docstring above for why a list
# seeded before annotation gets its reads constant-folded.
class _CogenCounters(object):
    def __init__(self):
        self.generated = 0
        self.declined = 0


_cogen_counters = _CogenCounters()


def set_late_jitcode_base(count):
    _late_jitcode_counter.base = count
    _late_jitcode_counter.next_index = count


def get_late_jitcode(index):
    # NonConstant(False) keeps this branch live for annotation (not pruned),
    # so _late_jitcodes_by_index gets a known JitCode value type; never runs.
    from rpython.rlib.nonconst import NonConstant
    if NonConstant(False):
        # fnaddr=llmemory.NULL, not None: JitCode.fnaddr's type is unified
        # across call sites; None here conflicts (SomeAddress vs None).
        _late_jitcodes_by_index[-1] = JitCode(
            "late-jitcode-type-hint", fnaddr=llmemory.NULL)
    return _late_jitcodes_by_index[index]


def register_late_jitcode(jitcode, own_liveness_info):
    """Single-threaded: no locking, matching the rest of this setup pipeline."""
    if jitcode.own_liveness_info is None:
        jitcode.own_liveness_info = own_liveness_info
    jitcode.index = _late_jitcode_counter.next_index
    assert jitcode.index not in _late_jitcodes_by_index, (
        "register_late_jitcode: _late_jitcode_counter produced an "
        "index already in use -- the monotonic counter did not advance "
        "between two registrations")
    _late_jitcode_counter.next_index += 1
    _late_jitcodes_by_index[jitcode.index] = jitcode


class PELinkedProgram(object):
    """One offline-linked JitCode, and the portal entry it stands for.

    A portal runs many code objects, so a program says which one it was linked
    for.  An interpreter with a single bytecode string (TLA) leaves both guard
    indices at -1 and always matches.
    """

    def __init__(self, jitcode, argument_sources, argument_constants):
        self.jitcode = jitcode
        # Size of the assembled residual jitcode, in bytes -- the axis
        # pe_call_threshold (rlib/jit.py) compares against to decide
        # whether an already-compiled call to this program is worth an
        # assembler call (large body) or better off inlined (tiny body).
        self.code_size = len(jitcode.code)
        # Lists keep one homogeneous RPython annotation whether linked
        # lowering is enabled or not; fixed-size tuples of () and (2, 1)
        # cannot be unified during native translation.
        self.argument_sources = list(argument_sources)
        self.argument_constants = list(argument_constants)
        self.guard_pc_index = -1
        # Every block boundary this program may be entered at -- one entry
        # per block the emitted code has a position for, not only its loop
        # headers.  One generated program covers each of them, so a method
        # the JIT traced from several points needs one program, not one per
        # point: the later ones would be the same code with its front cut
        # off.  Only these carry the register setup a trace start needs.
        self.guard_pcs = []
        self.guard_ref_index = -1
        self.guard_ref = lltype.nullptr(llmemory.GCREF.TO)
        # Subset of guard_pcs where a trace may legitimately start: this
        # program's loop headers plus its primary entry pc, exactly what the
        # offline CFG declares as trace starts.  A pc in guard_pcs but not
        # here is a mid-block pc some OTHER trace duplicated into this
        # program's tail -- looping there again would duplicate a residual
        # loop this program already provides (see pe_tick_suppressed in
        # warmstate.py).
        self.legit_entry_pcs = []
        # A program is built from a bytecode image, so the code object it
        # belongs to does not exist yet at translation time.  The matcher
        # recognises it the first time the portal hands one over, and the
        # pointer is remembered from then on.
        self.guard_match = _never_matches

    def set_guard(self, pc_index, pcs, ref_index, legit_entry_pcs):
        self.guard_pc_index = pc_index
        self.guard_pcs = list(pcs)
        self.guard_ref_index = ref_index
        self.legit_entry_pcs = list(legit_entry_pcs)

    def _covers(self, pc):
        for covered in self.guard_pcs:
            if covered == pc:
                return True
        return False

    def is_legit_entry_pc(self, pc):
        for covered in self.legit_entry_pcs:
            if covered == pc:
                return True
        return False

    def start_position(self, boxes):
        """Where tracing begins, for whichever entry point matched.

        Only the guarded instructions are valid starts: each is a merge point
        and each carries the register setup, so entering there is as sound as
        entering the program's own first instruction.
        """
        index = self.guard_pc_index
        if index < 0:
            return 0
        metadata = self.jitcode.pe_metadata
        if metadata is None:
            return 0
        return metadata.position_for_pc(boxes[index].getint())

    def build_call_boxes(self, boxes):
        """Map a caller's greens+reds onto this program's argument layout.

        Mirrors what initialize_state_from_start builds for a trace root:
        each source index >= 0 takes the box at that position from the
        caller, each negative one is a folded-in constant taken from
        argument_constants in order.  Both call sites -- trace roots and
        inlined calls -- route through here so they cannot diverge.
        """
        call_boxes = []
        constant_index = 0
        for source in self.argument_sources:
            if source >= 0:
                call_boxes.append(boxes[source])
            else:
                call_boxes.append(ConstInt(self.argument_constants[constant_index]))
                constant_index += 1
        return call_boxes

    def matches(self, boxes):
        """Is the portal entering the code object this program was linked for?"""
        index = self.guard_pc_index
        if index >= 0 and not self._covers(boxes[index].getint()):
            return False
        index = self.guard_ref_index
        if index < 0:
            return True
        return self.matches_ref(boxes[index].getref_base())

    def matches_ref(self, actual):
        """Ref-only half of matches(): does this program own 'actual'?

        Split out so PEJitCodeMetadata.linked_program_for can resolve the
        ref independently of the per-call pc guard, and cache the result.
        """
        if self.guard_ref_index < 0:
            return True
        expected = self.guard_ref
        if not expected:
            if not self.guard_match(actual):
                return False
            self.guard_ref = actual
            return True
        return actual == expected


class PEJitCodeMetadata(object):
    """RPython-friendly offline PE facts attached to a JitCode."""

    def __init__(self, entry_pc, block_pcs, loop_headers, backedge_sources,
                 backedge_targets, entry_pcs, entry_positions):
        self.entry_pc = entry_pc
        # All of these must be RPython lists rather than tuples.  A portal may
        # carry several linked programs, whose CFGs differ in size, and tuples
        # of different lengths cannot be unified; the searched ones would also
        # need a constant index as tuples.
        self.block_pcs = list(block_pcs)
        self.loop_headers = list(loop_headers)
        self.backedge_sources = list(backedge_sources)
        self.backedge_targets = list(backedge_targets)
        self.entry_pcs = list(entry_pcs)
        self.entry_positions = list(entry_positions)
        # One portal serves every code object the interpreter runs, so it can
        # carry several linked programs; the first whose guard matches wins.
        self.linked_programs = []
        # Read on every goto while tracing linked code, so it is resolved once
        # here rather than searched for each time.
        self.entry_position = self.position_for_pc(entry_pc)
        self.owns_linked_jitcode = False
        # True when the linked JitCode carries real jit_merge_points, which
        # close loops themselves -- including loops nested inside the program.
        self.has_merge_points = False
        # ref -> program (or None) cache for linked_program_for, keyed by the
        # runtime method GCREF.  Built lazily: most metadata never links more
        # than a couple of programs, where the linear walk is cheap enough
        # that a cache would just be overhead.
        self._program_cache = None
        # ref -> miss count, consulted only while below cogen_threshold (see
        # linked_program_for/_miss_count_reached below); same lazy-new_ref_dict
        # idiom as _program_cache, kept separate since it is written on every
        # miss, not only on a ref's first one.
        self._miss_counts = None
        # Misses required for a ref before runtime_cogen is invoked for it;
        # 0 (default) keeps today's behaviour of generating on the first
        # miss.  Set at runtime only (e.g. from an env var), never at
        # translation time, so this plain int field is safe to mutate.
        self.cogen_threshold = 0
        # Env var name to read cogen_threshold from, lazily on the first miss.
        # Read there, not at process start: startup code annotates before
        # apply_jit builds this metadata, so an early read folds to a no-op.
        self.threshold_env_var = None
        self._threshold_env_read = False
        # runtime_cogen: f(gcref) -> program or None; called once per ref.
        self.runtime_cogen = None
        # Green-box index carrying the ref, used only before any program is
        # installed; a runtime_cogen setter must also set this.
        self.guard_ref_index = -1

    def attach_linked_jitcode(self, jitcode, argument_sources,
                              argument_constants):
        """Add an unguarded program -- for a portal with one code object."""
        program = PELinkedProgram(jitcode, argument_sources,
                                  argument_constants)
        self.linked_programs.append(program)
        if jitcode.pe_metadata is self:
            # This metadata belongs to the linked JitCode itself, which is what
            # the goto check below asks about.
            self.owns_linked_jitcode = True
        # A residual jitcode is only ever attached to one portal, so a flag
        # on the jitcode itself is equivalent to (and cheaper than) asking
        # this metadata's is_linked_jitcode() to search for it.
        jitcode.pe_is_linked = True
        return program

    def has_linked_programs(self):
        return len(self.linked_programs) > 0

    def linked_program_for(self, boxes):
        """The program linked for the code object the portal is entering.

        All programs on one portal share the same guard_ref_index (derived
        from the portal's green layout), so the runtime method ref resolves
        to at most one program; cache that resolution keyed by ref instead
        of re-walking every program -- each doing a full guard_match byte
        compare -- on every call.  The pc guard still varies per call, so it
        is checked fresh after a cache hit.  Portals with no ref to guard on
        (guard_ref_index < 0, e.g. TLA's single-program case) fall back to
        the plain linear walk.
        """
        programs = self.linked_programs
        ref_index = -1
        if programs and programs[0].guard_ref_index >= 0:
            ref_index = programs[0].guard_ref_index
        elif not programs and self.runtime_cogen is not None:
            ref_index = self.guard_ref_index
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
                            # Below threshold: neither generate nor cache --
                            # a cached None here is a permanent decline.
                            return None
                        program = self._cogen_ref(ref)
                    cache[ref] = program
                if program is None:
                    return None
                index = program.guard_pc_index
                if index >= 0 and not program._covers(boxes[index].getint()):
                    return None
                return program
        for program in programs:
            if program.matches(boxes):
                return program
        return None

    def _resolve_ref(self, ref):
        """Which program owns 'ref', ignoring the pc guard.

        Kept independent of the pc check: a program that owns this ref but
        whose pc guard fails on this particular call must still be cached
        under the ref, so a later call with a covered pc finds it instead of
        being cached as a permanent non-match.
        """
        for program in self.linked_programs:
            if program.matches_ref(ref):
                return program
        return None

    def _miss_count_reached(self, ref):
        """Has 'ref' missed at least cogen_threshold times (this one
        included)?  threshold=0 makes this always True on the first miss,
        i.e. today's "generate immediately" behaviour.
        """
        if not self._threshold_env_read:
            self._threshold_env_read = True
            # Local binding: attribute re-reads don't flow-narrow away None.
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
        return count >= self.cogen_threshold

    def _cogen_ref(self, ref):
        """Invoked once per ref; linked_program_for caches the result.

        The sole choke point every runtime generation routes through
        (both PySOM's production callback and every in-process test one) --
        wrapped in a PYPYLOG section (PYPYLOG=pe-cogen:...) so generation
        cost shows up as wall time for free, with running totals alongside.
        """
        debug_start("pe-cogen")
        try:
            program = self.runtime_cogen(ref)
            if program is None or program.guard_ref != ref:
                program = None
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
        return pc in self.loop_headers

    def position_for_pc(self, pc):
        # Linear scan, even though every block boundary is now a guard pc and
        # this list can run to dozens of entries: it only runs once per trace
        # start, which is rare, so a dict would trade a real cost (building
        # and keeping it) for a saving that never shows up in profiles.
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
        self._called_from = called_from   # debugging
        self._ssarepr     = None          # debugging
        # None: offsets relative to global liveness_info. Set (runtime
        # JitCodes only): own private liveness chunk; offsets relative to it.
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
        """Where a jump means an offline-linked loop's back edge, or -1.

        Resolved once per frame rather than on every goto.  A JitCode with
        real merge points closes its loops through those instead, so it wants
        no position check at all.
        """
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
