import math

from rpython.jit.metainterp.history import AbstractDescr, getkind, new_ref_dict
from rpython.jit.metainterp.support import adr2int, int2adr
from rpython.jit.codewriter.flatten import Register, Label, TLabel, KINDS
from rpython.jit.codewriter.flatten import ListOfKind, IndirectCallTargets
from rpython.jit.codewriter.format import format_assembler
from rpython.jit.codewriter.jitcode import SwitchDictDescr, JitCode
from rpython.jit.codewriter import longlong
from rpython.rlib.objectmodel import ComputedIntSymbolic, specialize
from rpython.rlib.rarithmetic import r_int
from rpython.flowspace.model import Constant
from rpython.rtyper.lltypesystem import lltype, llmemory, rffi
from rpython.rtyper import rclass


class AssemblerError(Exception):
    pass


def int_fits_short(value, allow_short):
    """Would emit_resolved_const's 'int' branch pick the short 1-byte
    encoding for 'value'? Pulled out so native_pipeline.py's insn-key
    computation can answer the same question for a not-yet-placed
    operand, without an assembler instance or emitting anything.
    """
    if not allow_short:
        return False
    try:
        return -128 <= value <= 127
    except TypeError:    # "Symbolics cannot be compared!"
        return False


def _fixed_size_copy(src, empty):
    """A fresh copy of 'src', built by index-fill, not .append() -- see
    Assembler.make_jitcode's own note for why. 'empty' is returned as-is
    when 'src' is empty, matching JitCode.setup's own
    ``constants_i or self._empty_i``.
    """
    n = len(src)
    if n == 0:
        return empty
    dst = [src[0]] * n
    for i in range(1, n):
        dst[i] = src[i]
    return dst
# call_location, not arglistitemtype(0): item type may still annotate as
# Impossible when first reached, collapsing int/ref/float into one graph.
_fixed_size_copy._annspecialcase_ = 'specialize:call_location'


def _sorted_chars(live):
    """Sorted list of the (already-unique) chars in live.

    sorted(live) isn't RPython-legal here (see indirectcalltargets note
    below). ponytail: O(n^2) insertion sort, fine under 256 entries.
    """
    chars = []
    for char in live:
        chars.append(char)
    index = 1
    while index < len(chars):
        key = chars[index]
        gap = index - 1
        while gap >= 0 and chars[gap] > key:
            chars[gap + 1] = chars[gap]
            gap -= 1
        chars[gap + 1] = key
        index += 1
    return chars


class _NoDedupKeyGiven(object):
    """Distinct from None: None is itself a legit dedup key (null GCREF)."""
_NO_DEDUP_KEY = _NoDedupKeyGiven()


class _CounterState(object):
    # Holder for int counters: unlike list/dict, ints don't alias by
    # reference, so sharing this object is what lets share_with() work.
    def __init__(self):
        self.count_jitcodes = 0
        self.all_liveness_length = 0
        self.num_liveness_ops = 0


class Assembler(object):

    def __init__(self):
        self.insns = {}
        self.descrs = []
        # Dict-as-set (value always True): RPython has no native set type.
        # NativeAssembler shares this field and pyjitpl.py's
        # register_late_jitcode reads it too.
        self.indirectcalltargets = {}    # set of JitCodes
        self.list_of_addr2name = []
        self._descr_dict = {}
        self._counters = _CounterState()
        self._seen_raw_objects = {}    # dict-as-set (no RPython set type)
        self.all_liveness = []
        self.all_liveness_positions = {}

    # Properties over self._counters, not plain attrs: lets other methods
    # read/write them unchanged while sharing state via self._counters.

    def _get_count_jitcodes(self):
        return self._counters.count_jitcodes

    def _set_count_jitcodes(self, value):
        self._counters.count_jitcodes = value

    _count_jitcodes = property(_get_count_jitcodes, _set_count_jitcodes)

    def _get_all_liveness_length(self):
        return self._counters.all_liveness_length

    def _set_all_liveness_length(self, value):
        self._counters.all_liveness_length = value

    all_liveness_length = property(_get_all_liveness_length,
                                   _set_all_liveness_length)

    def _get_num_liveness_ops(self):
        return self._counters.num_liveness_ops

    def _set_num_liveness_ops(self, value):
        self._counters.num_liveness_ops = value

    num_liveness_ops = property(_get_num_liveness_ops, _set_num_liveness_ops)

    def assemble(self, ssarepr, jitcode=None, num_regs=None):
        """Take the 'ssarepr' representation of the code and assemble
        it inside the 'jitcode'.  If jitcode is None, make a new one.
        """
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
        if self._count_jitcodes < 20:    # stop if we have a lot of them
            jitcode._dump = format_assembler(ssarepr)
        self._count_jitcodes += 1
        return jitcode

    def setup(self, name):
        self.code = []
        # Split by kind: one dict keyed by mixed int/ref/float isn't
        # RPython-legal (no uniform key type across kinds).
        self.constants_dict_i = {}    # int value -> index
        # GCREF value -> index. Plain dict can't key on a GCREF (no
        # __hash__); new_ref_dict (history.py) is the RPython way. Its
        # rd_hash asserts on a null ref, so null is tracked separately.
        self.constants_dict_r = new_ref_dict()
        self._null_ref_const_index = -1
        self.constants_dict_f = {}    # (float/longlong value, negzero) -> index
        self.constants_i = []
        self.constants_r = []
        self.constants_f = []
        self.label_positions = {}
        self.tlabel_positions = []
        self.switchdictdescrs = []
        # Not dict.fromkeys(KINDS, 0): RPython can't annotate a mutable
        # module-level list as a constant call argument this way.
        count_regs = {}
        for kind in KINDS:
            count_regs[kind] = 0
        self.count_regs = count_regs
        self.liveness = {}
        # Dict-as-set: see indirectcalltargets note in __init__ above.
        self.startpoints = {}
        self.alllabels = {}
        self.resulttypes = {}
        self.ssareprname = name

    def emit_reg(self, reg):
        assert reg.index < self.count_regs[reg.kind]
        self.code.append(chr(reg.index))

    def count_reg(self, reg):
        if reg.index >= self.count_regs[reg.kind]:
            self.count_regs[reg.kind] = reg.index + 1

    def emit_const(self, const, kind, allow_short=False):
        return self.emit_const_value(const.value, const.concretetype, kind,
                                     allow_short=allow_short)

    def emit_const_value(self, value, concretetype, kind, allow_short=False):
        """Constant-free core of emit_const, keyed off (value, concretetype)
        instead of a flowspace Constant, producing the kind-appropriate
        resolved form emit_resolved_const dedups/emits.

        dedup_key stays the pre-cast value (except the 'ref' branch, which
        reassigns it post-cast): a cast like adr2int builds a fresh
        AddressAsInt wrapper each call, so keying dedup on the cast result
        would miss repeats of "the same" raw-pointer constant, while the
        pre-cast value is the same shared object across every placement of
        one template fragment (see jitcode_emitter.py's _localise) -- see
        test_repeated_helper_call_constants_dedup in test_native_pipeline.py.
        """
        dedup_key = value
        if kind == 'int':
            TYPE = concretetype
            if isinstance(TYPE, lltype.Ptr):
                assert TYPE.TO._gckind == 'raw'
                self.see_raw_object(value)
                value = llmemory.cast_ptr_to_adr(value)
                TYPE = llmemory.Address
            if TYPE == llmemory.Address:
                value = adr2int(value)
            if TYPE is lltype.SingleFloat:
                value = longlong.singlefloat2int(value)
            if not isinstance(value, (llmemory.AddressAsInt,
                                      ComputedIntSymbolic)):
                value = lltype.cast_primitive(lltype.Signed, value)
                if type(value) is r_int:
                    value = int(value)
        elif kind == 'ref':
            value = lltype.cast_opaque_ptr(llmemory.GCREF, value)
            # Not value._obj.container: constants_dict_r is keyed via
            # new_ref_dict's rd_eq/rd_hash on the ref itself now.
            dedup_key = value
        elif kind == 'float':
            if concretetype == lltype.Float:
                value = longlong.getfloatstorage(value)
            else:
                assert longlong.is_longlong(concretetype)
                value = rffi.cast(lltype.SignedLongLong, value)
        else:
            raise AssemblerError('unimplemented %r/%r in %r' %
                                 (value, concretetype, self.ssareprname))
        return self.emit_resolved_const(value, kind, allow_short=allow_short,
                                        dedup_key=dedup_key)

    @specialize.arg(2)
    def emit_resolved_const(self, value, kind, allow_short=False,
                            dedup_key=_NO_DEDUP_KEY):
        """Dedup/emit core given an already-resolved (cast) value; see
        emit_const_value's docstring for why dedup_key is pre-cast for
        'int'/'float' (post-cast for 'ref').  Defaults to value itself for
        NativeAssembler.write_insn (native_pipeline.py), whose native IR
        resolves constants once at translation time and reuses the same
        already-resolved operand object across every placement -- see
        native_fragments.py's NIntConst/NRefConst/NFloatConst docstrings.
        """
        if dedup_key is _NO_DEDUP_KEY:
            dedup_key = value
        if kind == 'int':
            if int_fits_short(value, allow_short):
                # emit the constant as a small integer
                self.code.append(chr(value & 0xFF))
                return True
            try:
                val = self.constants_dict_i[dedup_key]
            except KeyError:
                self.constants_i.append(value)
                val = self.count_regs['int'] + len(self.constants_i) - 1
                if not 0 <= val < 256:
                    raise AssemblerError("too many constants")
                self.constants_dict_i[dedup_key] = val
            except TypeError:
                # Unhashable dedup_key (e.g. a Symbolic): fall back to id().
                # id() isn't RPython-legal for a Symbolic; unexercised gap.
                idkey = id(dedup_key)
                try:
                    val = self.constants_dict_i[idkey]
                except KeyError:
                    self.constants_i.append(value)
                    val = self.count_regs['int'] + len(self.constants_i) - 1
                    if not 0 <= val < 256:
                        raise AssemblerError("too many constants")
                    self.constants_dict_i[idkey] = val
        elif kind == 'ref':
            if not value:
                # rd_hash asserts on a null ref; dedup it via this index
                # instead of ever making it a dict key.
                if self._null_ref_const_index < 0:
                    self.constants_r.append(value)
                    self._null_ref_const_index = (
                        self.count_regs['ref'] + len(self.constants_r) - 1)
                    if not 0 <= self._null_ref_const_index < 256:
                        raise AssemblerError("too many constants")
                val = self._null_ref_const_index
            else:
                try:
                    val = self.constants_dict_r[dedup_key]
                except KeyError:
                    self.constants_r.append(value)
                    val = self.count_regs['ref'] + len(self.constants_r) - 1
                    if not 0 <= val < 256:
                        raise AssemblerError("too many constants")
                    self.constants_dict_r[dedup_key] = val
        elif kind == 'float':
            # +0.0/-0.0 compare and hash equal but must not dedup together;
            # folded into dedup_key (not value), which is what gets looked
            # up.
            negzero = isinstance(dedup_key, float) and dedup_key == 0.0 and \
                math.copysign(1.0, dedup_key) < 0.0
            key = (dedup_key, negzero)
            try:
                val = self.constants_dict_f[key]
            except KeyError:
                self.constants_f.append(value)
                val = self.count_regs['float'] + len(self.constants_f) - 1
                if not 0 <= val < 256:
                    raise AssemblerError("too many constants")
                self.constants_dict_f[key] = val
        else:
            raise AssemblerError('unimplemented resolved kind %r in %r' %
                                 (kind, self.ssareprname))
        # emit the constant normally, as one byte that is an index in the
        # list of constants
        self.code.append(chr(val))
        return False

    def write_insn(self, insn):
        if insn[0] == '---':
            return
        if isinstance(insn[0], Label):
            self.label_positions[insn[0].name] = len(self.code)
            return
        if insn[0] == '-live-':
            key = len(self.code)
            self.startpoints[key] = True
            self.num_liveness_ops += 1
            live_i = self.get_liveness_info(insn[1:], 'int')
            live_r = self.get_liveness_info(insn[1:], 'ref')
            live_f = self.get_liveness_info(insn[1:], 'float')
            assert key not in self.liveness
            self.liveness[key] = live_i, live_r, live_f
            num = self.insns.setdefault('live/', len(self.insns))
            self.code.append(chr(num))
            self._encode_liveness(live_i, live_r, live_f)
            return
        startposition = len(self.code)
        self.code.append("temporary placeholder")
        #
        argcodes = []
        allow_short = (insn[0] in USE_C_FORM)
        for x in insn[1:]:
            if isinstance(x, Register):
                self.emit_reg(x)
                argcodes.append(x.kind[0])
            elif isinstance(x, Constant):
                kind = getkind(x.concretetype)
                is_short = self.emit_const(x, kind, allow_short=allow_short)
                if is_short:
                    argcodes.append('c')
                else:
                    argcodes.append(kind[0])
            elif isinstance(x, TLabel):
                self.alllabels[len(self.code)] = True
                self.tlabel_positions.append((x.name, len(self.code)))
                self.code.append("temp 1")
                self.code.append("temp 2")
                argcodes.append('L')
            elif isinstance(x, ListOfKind):
                itemkind = x.kind
                lst = list(x)
                assert len(lst) <= 255, "list too long!"
                self.code.append(chr(len(lst)))
                for item in lst:
                    if isinstance(item, Register):
                        assert itemkind == item.kind
                        self.emit_reg(item)
                    elif isinstance(item, Constant):
                        assert itemkind == getkind(item.concretetype)
                        self.emit_const(item, itemkind)
                    else:
                        raise NotImplementedError("found in ListOfKind(): %r"
                                                  % (item,))
                argcodes.append(itemkind[0].upper())
            elif isinstance(x, AbstractDescr):
                if x not in self._descr_dict:
                    self._descr_dict[x] = len(self.descrs)
                    self.descrs.append(x)
                if isinstance(x, SwitchDictDescr):
                    self.switchdictdescrs.append(x)
                num = self._descr_dict[x]
                assert 0 <= num <= 0xFFFF, "too many AbstractDescrs!"
                self.code.append(chr(num & 0xFF))
                self.code.append(chr(num >> 8))
                argcodes.append('d')
            elif isinstance(x, IndirectCallTargets):
                for target in x.lst:
                    self.indirectcalltargets[target] = True
            elif x == '->':
                assert '>' not in argcodes
                argcodes.append('>')
            else:
                raise NotImplementedError(x)
        #
        opname = insn[0]
        if '>' in argcodes:
            assert argcodes.index('>') == len(argcodes) - 2
            self.resulttypes[len(self.code)] = argcodes[-1]
        key = opname + '/' + ''.join(argcodes)
        num = self.insns.setdefault(key, len(self.insns))
        self.code[startposition] = chr(num)
        self.startpoints[startposition] = True

    def get_liveness_info(self, args, kind):
        """Return a set whose characters are register numbers.
        """
        lives = set()    # set of characters
        for reg in args:
            if isinstance(reg, Register) and reg.kind == kind:
                lives.add(chr(reg.index))
        return lives

    def _encode_liveness(self, live_i, live_r, live_f):
        from rpython.jit.codewriter.liveness import encode_offset, encode_liveness
        # Not frozenset() (same RPython-legality reason as set(), see
        # indirectcalltargets note in __init__): a sorted, deduplicated
        # char list joined into a string is an equally good dedup key.
        sorted_i = _sorted_chars(live_i)
        sorted_r = _sorted_chars(live_r)
        sorted_f = _sorted_chars(live_f)
        key = ("".join(sorted_i), "".join(sorted_r), "".join(sorted_f))
        try:
            pos = self.all_liveness_positions[key]
        except KeyError:
            pos = self.all_liveness_positions[key] = self.all_liveness_length
            self.all_liveness.append(
                chr(len(sorted_i)) + chr(len(sorted_r)) + chr(len(sorted_f)))
            self.all_liveness_length += 3
            # A list, not a bare tuple literal: RPython's rtyper only
            # supports iterating a tuple of static length 1; sorted_i/r/f
            # are all list-of-str, so a list literal unifies fine.
            for live in [sorted_i, sorted_r, sorted_f]:
                liveness = encode_liveness(live)
                if liveness:
                    self.all_liveness.append(liveness)
                    self.all_liveness_length += len(liveness)
        encode_offset(pos, self.code)

    def fix_labels(self):
        for name, pos in self.tlabel_positions:
            assert self.code[pos  ] == "temp 1"
            assert self.code[pos+1] == "temp 2"
            target = self.label_positions[name]
            assert 0 <= target <= 0xFFFF
            self.code[pos  ] = chr(target & 0xFF)
            self.code[pos+1] = chr(target >> 8)
        for descr in self.switchdictdescrs:
            as_dict = {}
            for key, switchlabel in descr._labels:
                target = self.label_positions[switchlabel.name]
                as_dict[key] = target
            descr.attach(as_dict)

    def check_result(self):
        # Register/constant cap from the single-byte encoding; a real limit
        # a large method can hit. AssemblerError, not assert, so a caller
        # can catch and decline instead of aborting.
        if self.count_regs['int'] + len(self.constants_i) > 256:
            raise AssemblerError("too many int registers/constants")
        if self.count_regs['ref'] + len(self.constants_r) > 256:
            raise AssemblerError("too many ref registers/constants")
        if self.count_regs['float'] + len(self.constants_f) > 256:
            raise AssemblerError("too many float registers/constants")

    def make_jitcode(self, jitcode):
        """Hand the assembled code and constant pools off to 'jitcode'.

        constants_i/r/f are rebuilt as fresh, index-filled copies, never
        the live self.constants_i/r/f lists built by .append(): RPython's
        "ever resized" list flag is unified across every JitCode sharing
        this attribute, so one append()-grown constants list would make
        blackhole.py's make_sure_not_resized reject every JitCode, not
        just this one. Mirrors BlackholeInterpreter.setposition's own
        ``[default_i] * n`` idiom on the sibling 'registers' argument.
        """
        jitcode.setup(''.join(self.code),
                      _fixed_size_copy(self.constants_i, JitCode._empty_i),
                      _fixed_size_copy(self.constants_r, JitCode._empty_r),
                      _fixed_size_copy(self.constants_f, JitCode._empty_f),
                      self.count_regs['int'],
                      self.count_regs['ref'],
                      self.count_regs['float'],
                      startpoints=self.startpoints,
                      alllabels=self.alllabels,
                      resulttypes=self.resulttypes)

    def see_raw_object(self, value):
        if value._obj not in self._seen_raw_objects:
            self._seen_raw_objects[value._obj] = True
            if not value:    # filter out NULL pointers
                return
            TYPE = lltype.typeOf(value).TO
            if isinstance(TYPE, lltype.FuncType):
                name = value._obj._name
            elif TYPE == rclass.OBJECT_VTABLE:
                if not value.name:    # this is really the "dummy" class
                    return            #   pointer from some dict
                name = ''.join(value.name.chars)
            else:
                return
            addr = llmemory.cast_ptr_to_adr(value)
            self.list_of_addr2name.append((addr, name))

    def finished(self, callinfocollection):
        # Helper called at the end of assembling.  Registers the extra
        # functions shown in _callinfo_for_oopspec.
        for func in callinfocollection.all_function_addresses_as_int():
            func = int2adr(func)
            self.see_raw_object(func.ptr)


# A set of instructions that use the 'c' encoding for small constants.
# Allowing it anywhere causes the number of instruction variants to
# expode, growing past 256.  So we list here only the most common
# instructions where the 'c' variant might be useful.
#
# Not a real set(): RPython's annotator has no representation for a set,
# even a module-level prebuilt one, once an 'in' test on it gets annotated
# (here, and in NativeAssembler.write_insn). dict.fromkeys still runs as
# plain Python at module import time; only the resulting type differs.
USE_C_FORM = dict.fromkeys([
    'copystrcontent',
    'getarrayitem_gc_pure_i',
    'getarrayitem_gc_pure_r',
    'getarrayitem_gc_i',
    'getarrayitem_gc_r',
    'goto_if_not_int_eq',
    'goto_if_not_int_ge',
    'goto_if_not_int_gt',
    'goto_if_not_int_le',
    'goto_if_not_int_lt',
    'goto_if_not_int_ne',
    'int_add',
    'int_and',
    'int_copy',
    'int_eq',
    'int_ge',
    'int_gt',
    'int_le',
    'int_lt',
    'int_ne',
    'int_return',
    'int_sub',
    'jit_merge_point',
    'pe_bailout_point',
    'new_array',
    'new_array_clear',
    'newstr',
    'setarrayitem_gc_i',
    'setarrayitem_gc_r',
    'setfield_gc_i',
    'strgetitem',
    'strsetitem',

    'foobar', 'baz',    # for tests
], True)
