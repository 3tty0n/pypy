"""fragment_to_native converts a translation-time TemplateFragment into a
NativeFragment/NativeInsn IR that is RPython-typed and runtime-legal.

TemplateFragment's insns are SSARepr tuples over translation-time-only
classes (flowspace Constant, codewriter Register/Label/TLabel,
SwitchDictDescr); this converts each one, once, into a flat list of
NativeInsn holding operands drawn from a small, closed set of tagged
classes (NReg/NIntConst/NRefConst/NFloatConst/NHole/NLabel/NTLabel/
NDescr/NListOfKind/NSwitchDictOperand).  native_pipeline.py's
emit_native/compute_liveness_native/NativeAssembler read only this
representation, never the original SSARepr/flowspace objects.

A fragment's own Label/TLabel operands name flow-graph Block objects
(identity-keyed, translation-time-only); only their identity *within one
fragment* matters, so each distinct name gets a small sequential int
(label_id), assigned per-fragment and discarded once conversion finishes.
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

# One counter, shared by every NReg ever built: the ones fragment_to_native
# bakes into prebuilt NativeFragments at translation time, and the ones
# native_pipeline.py's runtime _register() synthesizes on the fly for
# parallel-move/scratch writes.  Sharing it (instead of two independent
# counters) is what guarantees a translation-time id and a runtime-conjured
# id can never collide: the counter's value at the end of translation
# becomes its frozen starting value inside the translated binary, and every
# runtime call continues incrementing from there.  compute_liveness_native
# depends on that -- it keys "alive" by this id instead of by object
# identity, so two different NReg objects that happened to share an id
# would wrongly compare equal (see native_pipeline.py's liveness note).
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
        self.kind = kind          # 'int', 'ref' or 'float'
        self.index = index
        self.nid = _next_nreg_id()

    def __repr__(self):
        return "%%%s%d" % (self.kind[0], self.index)


class NIntConst(NOperand):
    """An 'int'-kind constant operand: a plain RPython ``int`` -- or,
    for a raw external-symbol/function pointer (e.g. a residual call's own
    callee address) or a ``ComputedIntSymbolic``, the ``Symbolic``
    (``llmemory.AddressAsInt`` / ``ComputedIntSymbolic``) the final backend
    link resolves.  A Symbolic is RPython-legal wherever a plain Signed int
    is, by design (its ``annotation()`` is ``SomeInteger()``), which is
    exactly why the rest of the JIT already stores such values in
    otherwise-int slots, e.g. ``jitcode.constants_i`` itself.

    Resolved once -- either at translation time by _const_operand_for from
    a flowspace Constant, or synthesized at runtime (holes, boundary
    fallbacks) with no source Constant at all.  Reused verbatim across
    every placement of one shared fragment rather than re-cast per
    placement: required to match the legacy Assembler.emit_resolved_const
    dedup behavior (see test_repeated_helper_call_constants_dedup).
    """
    def __init__(self, ivalue):
        self.ivalue = ivalue


class NRefConst(NOperand):
    """A 'ref'-kind constant operand: a plain ``llmemory.GCREF``, resolved
    once via the same ``lltype.cast_opaque_ptr`` ``Assembler.emit_const``
    applies at write_insn time -- either at translation time (from a
    genuine flowspace Constant, via ``_const_operand_for``) or at runtime,
    with no upstream Constant, for the null "no boundary source" fallback
    (``_emit_moves_native``/``_initialise_scratch_native``) -- both mean the
    same thing on the wire, a null GCREF."""
    def __init__(self, value):
        self.value = value


class NFloatConst(NOperand):
    """A 'float'-kind constant operand: the already-computed float-storage
    value (matching ``Assembler.emit_const_value``'s own 'float' branch --
    ``lltype.Float`` storage, or a ``SignedLongLong``), resolved once at
    translation time by ``_const_operand_for``.  Never runtime-synthesized:
    a 'float'-kind boundary with no source gets an *int*-kind zero
    ``NIntConst`` instead, reproducing a quirk of the original code
    (``Assembler.emit_const_value`` buckets a constant by its own
    concretetype, not by the surrounding opcode's kind) -- see
    native_pipeline.py's fallback-construction comment."""
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
    mutated by two different placements -- emit_native rebuilds it fresh."""
    def __init__(self, keys, label_ids):
        self.keys = keys
        self.label_ids = label_ids


class NListOfKind(NOperand):
    def __init__(self, kind, items):
        self.kind = kind
        self.items = items   # list of NReg/NIntConst/NRefConst/NFloatConst/NHole


class NIndirectCallTargets(NOperand):
    def __init__(self, lst):
        self.lst = lst        # list of JitCodes


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
    """One template, ready to be placed in a program by ``emit_native``."""

    def __init__(self, insns, exits, num_regs, boundary_entry, prologue,
                 num_labels, merge_point):
        self.insns = insns                      # list of NativeInsn
        self.exits = exits                      # list of NativeFragmentExit
        self.num_regs = num_regs                # dict kind -> int
        self.boundary_entry = boundary_entry     # dict name -> (kind, index)
        self.prologue = prologue                 # list of (kind, index, name)
        self.num_labels = num_labels             # local label id space size
        self.merge_point = merge_point


class _Converter(object):
    """Cache one NReg per (kind, index) per-fragment, not program-wide:
    liveness keys "alive" by object identity (see native_pipeline.py's
    liveness note), so the caching scope must be exact -- too narrow
    never cancels a register, too wide cancels it too eagerly."""

    def __init__(self, const_cache=None):
        self._label_ids = {}
        self._registers = {}
        # Shared *across* fragments (passed in from build_native_table, one
        # dict for the whole program), unlike _registers/_label_ids above --
        # see convert_operand's own note on why constants need the opposite
        # scope from registers.
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
        # Resolve to a monomorphic NIntConst/NRefConst/NFloatConst here, at
        # translation time -- see _const_operand_for.  Constant checked
        # last: HoleConstant is itself a Constant subclass, checked above.
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
# Runtime boundary: everything above this line (_Counter/NReg/NIntConst/
# NAddrIntConst/NRefConst/NFloatConst/NHole/NLabel/NTLabel/NDescr/
# NSwitchDictOperand/NListOfKind/NIndirectCallTargets/NativeInsn/
# NativeFragmentExit/NativeFragment as plain data classes, plus
# native_fragment_for below) is read/constructed at *runtime* -- inside a
# translated binary -- and must stay RPython-legal.  Everything above that
# builds/converts a NativeFragment (_Converter, convert_insn/
# convert_operand, _const_operand_for, fragment_to_native, build_native_table)
# runs only once, at *translation* time, and is exempt: it may freely touch
# flowspace.model.Constant, lltype casts, dynamic isinstance dispatch on
# translation-time-only classes, and anything else ordinary Python code can
# do, since none of it survives past producing the prebuilt NativeFragment
# tables native_pipeline.py reads.

def native_fragment_for(native_table, key, merge_point):
    no_merge, merge = native_table[key]
    fragment = merge if merge_point else no_merge
    assert fragment is not None, (
        "no native fragment compiled for opcode %r (merge_point=%r)" %
        (key, merge_point))
    return fragment
