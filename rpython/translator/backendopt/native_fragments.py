"""fragment_to_native converts a translation-time TemplateFragment into a
NativeFragment/NativeInsn IR that is RPython-typed and runtime-legal.

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


class NReg(NOperand):
    """Duck-types Register (kind, index) so Assembler.emit_reg/count_reg
    work unmodified."""
    def __init__(self, kind, index):
        self.kind = kind          # 'int', 'ref' or 'float'
        self.index = index

    def __repr__(self):
        return "%%%s%d" % (self.kind[0], self.index)


class NConst(NOperand):
    """A constant operand.

    ``constant``, when set, is the original translation-time-built
    flowspace Constant.  It is None for values only ever synthesized
    during placement -- a hole's bound int, or the zero/null fallback
    used to fill an unset boundary register -- in which case ``ivalue``
    carries the plain int directly (meaningless for kind == 'ref').
    """
    def __init__(self, kind, constant=None, ivalue=0):
        self.kind = kind
        self.constant = constant
        self.ivalue = ivalue


class NHole(NOperand):
    """Unresolved late-static value; emit_native replaces it with an
    NConst before liveness/assembly ever run."""
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
        self.items = items   # list of NReg/NConst/NHole


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

    def __init__(self):
        self._label_ids = {}
        self._registers = {}

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
            return NConst(getkind(x.concretetype), constant=x)
        raise NotImplementedError(
            "native_fragments: unhandled operand %r" % (x,))


def fragment_to_native(fragment, merge_point=False):
    converter = _Converter()
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
    for (key, merge_point), fragment in fragments.items():
        no_merge, merge = table.get(key, (None, None))
        native = fragment_to_native(fragment, merge_point)
        if merge_point:
            merge = native
        else:
            no_merge = native
        table[key] = (no_merge, merge)
    return table


def native_fragment_for(native_table, key, merge_point):
    no_merge, merge = native_table[key]
    fragment = merge if merge_point else no_merge
    assert fragment is not None, (
        "no native fragment compiled for opcode %r (merge_point=%r)" %
        (key, merge_point))
    return fragment
