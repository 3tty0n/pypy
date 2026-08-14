from __future__ import print_function

from rpython.jit.metainterp.history import AbstractDescr, ConstInt
from rpython.jit.metainterp.support import adr2int
from rpython.rlib.objectmodel import we_are_translated, specialize
from rpython.rlib.rarithmetic import base_int
from rpython.rtyper.lltypesystem import llmemory, lltype


def _never_matches(gcref):
    return False


class PELinkedProgram(object):
    """One offline-linked JitCode, and the portal entry it stands for.

    A portal runs many code objects, so a program says which one it was linked
    for.  An interpreter with a single bytecode string (TLA) leaves both guard
    indices at -1 and always matches.
    """

    def __init__(self, jitcode, argument_sources, argument_constants):
        self.jitcode = jitcode
        # Lists keep one homogeneous RPython annotation whether linked
        # lowering is enabled or not; fixed-size tuples of () and (2, 1)
        # cannot be unified during native translation.
        self.argument_sources = list(argument_sources)
        self.argument_constants = list(argument_constants)
        self.guard_pc_index = -1
        self.guard_pc = -1
        self.guard_ref_index = -1
        self.guard_ref = lltype.nullptr(llmemory.GCREF.TO)
        # A program is built from a bytecode image, so the code object it
        # belongs to does not exist yet at translation time.  The matcher
        # recognises it the first time the portal hands one over, and the
        # pointer is remembered from then on.
        self.guard_match = _never_matches

    def set_guard(self, pc_index, pc, ref_index):
        self.guard_pc_index = pc_index
        self.guard_pc = pc
        self.guard_ref_index = ref_index

    def matches(self, boxes):
        """Is the portal entering the code object this program was linked for?"""
        index = self.guard_pc_index
        if index >= 0 and boxes[index].getint() != self.guard_pc:
            return False
        index = self.guard_ref_index
        if index >= 0:
            actual = boxes[index].getref_base()
            expected = self.guard_ref
            if not expected:
                if not self.guard_match(actual):
                    return False
                self.guard_ref = actual
            elif actual != expected:
                return False
        return True


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
        return program

    def has_linked_programs(self):
        return len(self.linked_programs) > 0

    def linked_program_for(self, boxes):
        """The program linked for the code object the portal is entering."""
        for program in self.linked_programs:
            if program.matches(boxes):
                return program
        return None

    def is_linked_jitcode(self, jitcode):
        for program in self.linked_programs:
            if program.jitcode is jitcode:
                return True
        return False

    def is_loop_header(self, pc):
        return pc in self.loop_headers

    def position_for_pc(self, pc):
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
        self._called_from = called_from   # debugging
        self._ssarepr     = None          # debugging

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
        self.const_keys_in_order = map(ConstInt, sorted(as_dict.keys()))

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
