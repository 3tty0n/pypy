"""fragment_to_native converts a translation-time TemplateFragment into a
NativeFragment/NativeInsn IR that is RPython-typed and runtime-legal.
"""

from rpython.jit.codewriter.flatten import (
    Register, Label, TLabel, ListOfKind, IndirectCallTargets, KINDS)
from rpython.jit.codewriter.jitcode import SwitchDictDescr
from rpython.jit.metainterp.history import AbstractDescr, getkind
from rpython.translator.backendopt.jitcode_emitter import HoleConstant


class NOperand(object):
    """Base of the tagged operand hierarchy; never dispatch on flowspace/
    codewriter translation-time classes at the runtime-read path."""


class _Counter(object):
    def __init__(self):
        self.value = 0

# Shared counter: translation-time and runtime nid's must never collide,
# since compute_liveness_native keys "alive" by nid, not object identity.
_nreg_id_counter = _Counter()


def _next_nreg_id():
    nid = _nreg_id_counter.value
    _nreg_id_counter.value = nid + 1
    return nid


class NReg(NOperand):
    """Duck-types Register (kind, index) so Assembler.emit_reg/count_reg
    work unmodified.  nid: RPython can't key a set/dict by object identity
    at the runtime-read path, so compute_liveness_native uses nid instead.
    """
    def __init__(self, kind, index):
        self.kind = kind
        self.index = index
        self.nid = _next_nreg_id()

    def __repr__(self):
        return "%%%s%d" % (self.kind[0], self.index)


class NIntConst(NOperand):
    """int, or a Symbolic (AddressAsInt/ComputedIntSymbolic; RPython-legal
    as SomeInteger()).  Resolved once; reuse verbatim to keep dedup working.
    """
    def __init__(self, ivalue):
        self.ivalue = ivalue


class NRefConst(NOperand):
    def __init__(self, value):
        self.value = value


class NFloatConst(NOperand):
    """A float boundary with no source falls back to an int-kind
    NIntConst(0), not NFloatConst(0) -- mirrors a legacy quirk.

    Nothing constructs one, so the class-level default is what gives the
    classdef its attribute: an interpreter whose residual code has float
    boundaries makes the assembler's isinstance branch reachable, and
    reading .value off an attribute-less classdef blocks annotation.
    """
    value = 0.0

    def __init__(self, value):
        self.value = value


def _const_operand_for(x, const_cache=None):
    """const_cache must be program-wide, not per-fragment: a resolved
    AddressAsInt isn't hashable, so id()-based dedup needs the *same*
    cached object, not merely an equal one (mirrors emit_resolved_const).
    """
    from rpython.rtyper.lltypesystem import lltype, llmemory, rffi
    from rpython.jit.codewriter import longlong
    from rpython.jit.metainterp.support import adr2int
    from rpython.rlib.objectmodel import ComputedIntSymbolic
    from rpython.rlib.rarithmetic import r_int

    cache_key = None
    if const_cache is not None:
        # lltype pointer objects raise TypeError on hash(); fall back to
        # id() -- sound since the rtyper interns one object per pointer.
        try:
            hash(x.value)
            cache_key = (x.value, x.concretetype)
        except TypeError:
            cache_key = (id(x.value), x.concretetype)
        try:
            return const_cache[cache_key]
        except KeyError:
            pass

    kind = getkind(x.concretetype)
    if kind == "ref":
        result = NRefConst(lltype.cast_opaque_ptr(llmemory.GCREF, x.value))
    elif kind == "float":
        if x.concretetype == lltype.Float:
            value = longlong.getfloatstorage(x.value)
        else:
            assert longlong.is_longlong(x.concretetype)
            value = rffi.cast(lltype.SignedLongLong, x.value)
        result = NFloatConst(value)
    elif kind == "int":
        value = x.value
        TYPE = x.concretetype
        if isinstance(TYPE, lltype.Ptr):
            assert TYPE.TO._gckind == 'raw'
            value = llmemory.cast_ptr_to_adr(value)
            TYPE = llmemory.Address
        if TYPE == llmemory.Address:
            value = adr2int(value)
        if TYPE is lltype.SingleFloat:
            value = longlong.singlefloat2int(value)
        if not isinstance(value, (llmemory.AddressAsInt, ComputedIntSymbolic)):
            value = lltype.cast_primitive(lltype.Signed, value)
            if type(value) is r_int:
                value = int(value)
        # else: Symbolic stays as-is -- legal wherever Signed int is.
        result = NIntConst(value)
    else:
        raise NotImplementedError(
            "native_fragments: unhandled constant kind %r for %r" % (kind, x))

    if cache_key is not None:
        const_cache[cache_key] = result
    return result


class NHole(NOperand):
    """Unresolved late-static value; emit_native replaces it with an
    NIntConst before liveness/assembly ever run."""
    def __init__(self, name, concretetype):
        self.name = name
        self.concretetype = concretetype
        self.kind = getkind(concretetype)

    def __repr__(self):
        return "hole(%s)" % (self.name,)


class NLabel(NOperand):
    """label_id is fragment-local before placement, program-wide unique
    after (see native_pipeline._block_label_id/_fragment_label_id)."""
    def __init__(self, label_id):
        self.label_id = label_id


class NTLabel(NOperand):
    def __init__(self, label_id):
        self.label_id = label_id


class NDescr(NOperand):
    def __init__(self, descr):
        self.descr = descr


class NSwitchDictOperand(NOperand):
    """Not a direct SwitchDictDescr: a shared fragment's descr must not be
    mutated by two different placements -- emit_native rebuilds it fresh.

    Class-level defaults: an interpreter whose residual code has no switch
    never constructs one, and the isinstance branch that reads these would
    then look at a classdef with no attributes at all.
    """
    keys = []
    label_ids = []

    def __init__(self, keys, label_ids):
        self.keys = keys
        self.label_ids = label_ids


class NListOfKind(NOperand):
    def __init__(self, kind, items):
        self.kind = kind
        self.items = items


class NIndirectCallTargets(NOperand):
    def __init__(self, lst):
        self.lst = lst


class NativeInsn(object):
    """opcode '---'/'-live-'/'@label' are magic, pipeline-special-cased
    (barrier/liveness-point/label-def).  'result' is its own field, not
    embedded in operands.
    """
    def __init__(self, opcode, operands, result=None):
        self.opcode = opcode
        self.operands = operands
        self.result = result

    def __repr__(self):
        return "NativeInsn(%r, %r, result=%r)" % (
            self.opcode, self.operands, self.result)


class NativeFragmentExit(object):
    def __init__(self, index, operands, terminator):
        self.index = index
        # name -> (kind, index) tuple, or None; copied verbatim, since
        # FragmentExit.operands never holds a raw Register/Constant here.
        self.operands = operands
        self.terminator = terminator


class NativeFragment(object):
    def __init__(self, insns, exits, num_regs, boundary_entry, prologue,
                 num_labels, merge_point):
        self.insns = insns
        self.exits = exits
        self.num_regs = num_regs
        self.boundary_entry = boundary_entry
        self.prologue = prologue
        self.num_labels = num_labels
        self.merge_point = merge_point


class _Converter(object):
    """Cache one NReg per (kind, index) per-fragment, not program-wide:
    liveness keys "alive" by identity, so the caching scope must be exact
    -- too narrow never cancels a register, too wide cancels it too eagerly.
    """

    def __init__(self, const_cache=None):
        self._label_ids = {}
        self._registers = {}
        # const_cache is shared program-wide; _registers/_label_ids are
        # per-fragment -- constants dedup across fragments, registers don't.
        self._const_cache = const_cache

    def label_id(self, name):
        label_id = self._label_ids.get(id(name))
        if label_id is None:
            label_id = len(self._label_ids)
            self._label_ids[id(name)] = label_id
        return label_id

    def num_labels(self):
        return len(self._label_ids)

    def convert_insn(self, insn):
        if insn[0] == "---":
            return NativeInsn("---", [])
        if isinstance(insn[0], Label):
            return NativeInsn("@label", [NLabel(self.label_id(insn[0].name))])
        if insn[0] == "-live-":
            operands = [self.convert_operand(x) for x in insn[1:]]
            return NativeInsn("-live-", operands)

        args = list(insn[1:])
        result = None
        if len(args) >= 2 and args[-2] == "->":
            reg = args[-1]
            assert isinstance(reg, Register)
            result = self.register(reg.kind, reg.index)
            args = args[:-2]
        operands = [self.convert_operand(x) for x in args]
        return NativeInsn(insn[0], operands, result)

    def register(self, kind, index):
        key = (kind, index)
        reg = self._registers.get(key)
        if reg is None:
            reg = NReg(kind, index)
            self._registers[key] = reg
        return reg

    def convert_operand(self, x):
        if isinstance(x, Register):
            return self.register(x.kind, x.index)
        if isinstance(x, HoleConstant):
            return NHole(x.hole_name, x.concretetype)
        if isinstance(x, TLabel):
            return NTLabel(self.label_id(x.name))
        if isinstance(x, ListOfKind):
            return NListOfKind(x.kind,
                               [self.convert_operand(item) for item in x.content])
        if isinstance(x, SwitchDictDescr):
            keys = [key for key, _label in x._labels]
            label_ids = [self.label_id(label.name) for _key, label in x._labels]
            return NSwitchDictOperand(keys, label_ids)
        if isinstance(x, AbstractDescr):
            return NDescr(x)
        if isinstance(x, IndirectCallTargets):
            return NIndirectCallTargets(x.lst)
        # Constant checked last: HoleConstant is itself a Constant
        # subclass, so it must be (and already is, above) checked first.
        from rpython.flowspace.model import Constant
        if isinstance(x, Constant):
            return _const_operand_for(x, self._const_cache)
        raise NotImplementedError(
            "native_fragments: unhandled operand %r" % (x,))


def fragment_to_native(fragment, merge_point=False, const_cache=None):
    converter = _Converter(const_cache)
    insns = [converter.convert_insn(insn) for insn in fragment.insns]
    exits = [NativeFragmentExit(e.index, dict(e.operands), e.terminator)
             for e in fragment.exits]
    return NativeFragment(
        insns, exits, dict(fragment.num_regs), dict(fragment.boundary_entry),
        list(fragment.prologue), converter.num_labels(), merge_point)


def build_native_table(fragments):
    """table[key] = (no_merge, merge); either slot is None when that
    merge-point variant was never compiled (see native_fragment_for)."""
    table = {}
    const_cache = {}
    for (key, merge_point), fragment in fragments.items():
        no_merge, merge = table.get(key, (None, None))
        native = fragment_to_native(fragment, merge_point, const_cache)
        if merge_point:
            merge = native
        else:
            no_merge = native
        table[key] = (no_merge, merge)
    return table


# ____________________________________________________________
# Runtime boundary: the data classes above (+ native_fragment_for below)
# are read at runtime, so must stay RPython-legal; _Converter and the
# conversion functions above run only at translation time and are exempt.

def native_fragment_for(native_table, key, merge_point):
    no_merge, merge = native_table[key]
    fragment = merge if merge_point else no_merge
    assert fragment is not None, (
        "no native fragment compiled for opcode %r (merge_point=%r)" %
        (key, merge_point))
    return fragment
