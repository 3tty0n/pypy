"""Assemble each residual template once, and emit a program by
concatenating and patching the resulting fragments -- avoids re-running
the codewriter over a whole-program graph for every generated program.

A fragment is the instruction list the flattener produces for one
template: its boundary values arrive in the lowest registers of each
kind (``GraphFlattener.enforce_input_args``) and leave in the exit
records, and its late-static values are ``HoleConstant`` placeholders
patched in per placement.
"""

from rpython.flowspace.model import Constant
from rpython.rtyper.lltypesystem import lltype

#: A value no real late-static operand is likely to take, so an unpatched
#: placeholder shows up as an obviously wrong number.
HOLE_SENTINEL = 0x5E7717E1


def _boundary_value_origins(graph):
    """Like partialeval_template's _variable_origins, but also sees
    through a same-block setfield/getfield round trip and through
    cast_pointer/same_as.
    """
    identity_ops = ("cast_pointer", "same_as")
    origins = dict((var, var) for var in graph.startblock.inputargs)
    changed = True
    while changed:
        changed = False
        for block in graph.iterblocks():
            stored = {}
            for op in block.operations:
                if op.opname == "setfield" and isinstance(op.args[1], Constant):
                    stored[(op.args[0], op.args[1].value)] = op.args[2]
                elif (op.opname == "getfield"
                      and isinstance(op.args[1], Constant)):
                    key = (op.args[0], op.args[1].value)
                    if key in stored and op.result not in origins:
                        source = stored[key]
                        origins[op.result] = origins.get(source, source)
                        changed = True
                elif (op.opname in identity_ops and len(op.args) == 1
                      and op.result not in origins):
                    source = op.args[0]
                    origins[op.result] = origins.get(source, source)
                    changed = True
            for link in block.exits:
                for value, target_var in zip(link.args, link.target.inputargs):
                    origin = origins.get(value)
                    if origin is not None and target_var not in origins:
                        origins[target_var] = origin
                        changed = True
    return origins


class HoleConstant(Constant):
    """A Constant standing in for a late-static value until a program
    supplies it; behaves as an ordinary Constant through transformation,
    register allocation and flattening.
    """

    def __init__(self, name, concretetype):
        # A GC/raw pointer needs a well-typed placeholder, or jtransform
        # sees a Constant whose .value doesn't match its concretetype.
        if isinstance(concretetype, lltype.Ptr):
            value = lltype.nullptr(concretetype.TO)
        else:
            value = HOLE_SENTINEL
        Constant.__init__(self, value, concretetype)
        self.hole_name = name

    def __repr__(self):
        return "hole(%s)" % (self.hole_name,)


class FragmentExit(object):
    """One way out of a fragment; ``operands`` holds, per boundary name,
    the ssarepr operand carrying that value where the fragment leaves off.
    """

    def __init__(self, index, operands, terminator):
        self.index = index
        self.operands = operands
        self.terminator = terminator


class TemplateFragment(object):
    """One template, assembled once and ready to be placed in a program."""

    def __init__(self, insns, exits, num_regs, boundary_entry, prologue=()):
        self.insns = insns
        self.exits = exits
        self.num_regs = num_regs
        self.boundary_entry = boundary_entry
        self.prologue = tuple(prologue)


class FragmentCompiler(object):
    """Runs the codewriter once per template, producing relocatable
    fragments. Each template is reshaped so every exit becomes a plain
    ``return <exit index>`` -- a shape the flattener handles, and a mark
    the concatenator can turn into a jump.
    """

    def __init__(self, codewriter, portal_jd, static_name, hole_names,
                 boundary_names, jit_merge_point_args=(), null_names=()):
        self.codewriter = codewriter
        self.portal_jd = portal_jd
        self.static_name = static_name
        self.hole_names = tuple(hole_names)
        self.boundary_names = tuple(boundary_names)
        self.jit_merge_point_args = tuple(jit_merge_point_args)
        # Names declared invariant for the whole portal, since they can't
        # be inferred from one template.
        self.null_names = tuple(null_names)

    def compile(self, template, bindings={}, merge_point=False):
        from rpython.flowspace.model import Block, Link, copygraph
        from rpython.jit.codewriter.flatten import flatten_graph, KINDS
        from rpython.jit.codewriter.jtransform import transform_graph
        from rpython.jit.codewriter.regalloc import perform_register_allocation
        from rpython.rtyper.lltypesystem import lltype
        from rpython.translator.backendopt.partialeval import (
            _find_split_transitions, replace_uses)
        from rpython.translator.backendopt.partialeval_template import (
            Finish, LinkedResidualLowerer)

        graph = copygraph(template.residual_graph, shallowvars=True)
        # Snapshot before _reorder_arguments overwrites graph.signature.
        original_order = list(graph.signature[0])
        named = dict(zip(graph.signature[0], graph.startblock.inputargs))
        self._thread_boundary_values(graph, named)
        replace_uses(graph, self._placeholders(template, named, bindings))
        helper = LinkedResidualLowerer(
            self.boundary_names, (), (), self.jit_merge_point_args)
        helper.portal_jd = self.portal_jd
        if merge_point and self.jit_merge_point_args:
            # A real merge point: without it the metainterp must spot the
            # back edge by position, whose -live- cannot be placed (the
            # assembler hoists a label above a -live- that follows it).
            helper._remove_runtime_loop_markers(graph)
            helper._insert_merge_point(graph)
        elif self.jit_merge_point_args:
            # Every other block boundary gets a pe_bailout_point instead:
            # same greens+reds, but a no-op while tracing (see
            # opimpl_pe_bailout_point) -- lets blackhole bail out of this
            # jitcode one block at a time.
            helper._insert_bailout_point(graph)
        # After the merge point, so its greens are the bound values; a
        # merge point on an unbound pc would name every instruction alike.
        # Reds must not be constants (the jitdriver rejects that), so a
        # late-static red gets assigned into its register instead.
        reds = self._bound_reds(named, bindings) if merge_point else ()
        replace_uses(graph, self._placeholders(
            template, named, bindings, skip=reds))
        # After both: _insert_merge_point rebuilds the name-to-argument
        # map from the signature, and the portal fills entry registers in
        # its own order, not the step function's.
        ordered = self._reorder_arguments(graph, named)

        transitions = _find_split_transitions(graph)
        terminators = self._align(template, len(transitions))
        helper.state_names = self._state_names(terminators)
        helper.state_count = len(helper.state_names)
        finishing = [isinstance(t, Finish) for t in terminators]
        if any(finishing) and not all(finishing):
            raise ValueError("template %r mixes Finish and Continue exits"
                             % (template.key,))

        if all(finishing):
            # After _remove_result_tuple nothing else uses the result, and
            # a value no exit consumes gets no register.
            exit_block = Block([self._like(transitions[0].fields["item1"])])
        else:
            exit_block = Block([self._signed()])
        exit_block.operations = ()
        exit_block.exits = ()

        tails = []
        for index, transition in enumerate(transitions):
            values = self._boundary_values(
                helper, graph, transition, original_order)
            helper._remove_result_tuple(transition)
            if all(finishing):
                tails.append(None)
                transition.block.recloseblock(
                    Link([transition.fields["item1"]], exit_block))
                continue
            args, tail = self._tail_block(named, values, index, exit_block)
            transition.block.recloseblock(Link(args, tail))
            tails.append(tail)
        graph.returnblock = exit_block

        callcontrol = self.codewriter.callcontrol
        transform_graph(graph, self.codewriter.cpu, callcontrol, self.portal_jd)
        regallocs = dict((kind, perform_register_allocation(graph, kind))
                         for kind in KINDS)
        ssarepr = flatten_graph(graph, regallocs, cpu=callcontrol.cpu)

        entry = self._entry_registers(ordered)
        exits = []
        for index, tail in enumerate(tails):
            exits.append(FragmentExit(
                index, self._tail_registers(regallocs, tail, entry),
                terminators[index]))
        num_regs = dict(
            (kind, max(regallocs[kind]._coloring.values()) + 1
             if regallocs[kind]._coloring else 0)
            for kind in KINDS)
        return TemplateFragment(ssarepr.insns, exits, num_regs, entry,
                                self._prologue(ordered, reds))

    def _thread_boundary_values(self, graph, named):
        """Without this, an unused-here boundary name has no live copy in
        a later block, and the register allocator can reuse its color."""
        terminal = (graph.returnblock, graph.exceptblock)
        extra = {graph.startblock: named}
        for name in self.boundary_names:
            var = named.get(name)
            if var is None:
                continue
            for block in graph.iterblocks():
                if block is graph.startblock or block in terminal:
                    continue
                fresh = self._like(var)
                block.inputargs = list(block.inputargs) + [fresh]
                extra.setdefault(block, {})[name] = fresh
            for block in graph.iterblocks():
                source = extra.get(block, {}).get(name)
                if source is None:
                    continue
                for link in block.exits:
                    if link.target in terminal:
                        continue
                    link.args = list(link.args) + [source]

    def _state_names(self, terminators):
        for terminator in terminators:
            state = getattr(terminator, "state", ())
            if state:
                return tuple(name for name, _expression in state)
        return ()

    def _align(self, template, count):
        terminators = template.terminators
        if len(terminators) == 1 and count > 1:
            # RTyping can keep equivalent syntactic exits that the
            # symbolic template already canonicalised into one terminator.
            terminators = terminators * count
        if len(terminators) != count:
            raise ValueError(
                "template %r has %d exits but %d terminators"
                % (template.key, count, len(terminators)))
        return terminators

    def _like(self, var):
        from rpython.flowspace.model import Variable
        fresh = Variable()
        fresh.concretetype = var.concretetype
        return fresh

    # ------------------------------------------------------------------

    def _signed(self):
        from rpython.flowspace.model import Variable
        from rpython.rtyper.lltypesystem import lltype
        var = Variable()
        var.concretetype = lltype.Signed
        return var

    def _bound_reds(self, named, bindings):
        if self.portal_jd is None:
            return ()
        jitdriver = self.portal_jd.jitdriver
        reds = self.jit_merge_point_args[len(jitdriver.greens):]
        return [name for name in reds if name in bindings and name in named]

    def _prologue(self, ordered, reds):
        """Only fixes the register here; ProgramEmitter._place fills the
        value in from block.bindings."""
        from rpython.jit.metainterp.history import getkind
        counts = {}
        places = {}
        for name, var in ordered:
            kind = getkind(var.concretetype)
            if kind == "void":
                continue
            places[name] = (kind, counts.get(kind, 0))
            counts[kind] = places[name][1] + 1
        return [(places[name][0], places[name][1], name)
                for name in reds if name in places]

    def _placeholders(self, template, named, bindings, skip=()):
        """Fixes the static key; every other late-static name gets a
        hole, not bindings' value -- one fragment is shared by every
        instance of the opcode."""
        replacements = {}
        if self.static_name in named:
            var = named[self.static_name]
            replacements[var] = Constant(template.key, var.concretetype)
        late_static = set(bindings) | set(self.hole_names)
        for name in late_static:
            if name == self.static_name or name in skip:
                continue
            if name in named:
                var = named[name]
                replacements[var] = HoleConstant(name, var.concretetype)
        return replacements

    def _boundary_values(self, helper, graph, transition, original_order):
        """Which value carries each boundary name where this exit leaves
        off.

        An exit tuple can omit a boundary name whose value is unchanged
        (see _thread_boundary_values), with no marker saying so, so a
        naive positional zip against boundary_names would misassign every
        later name once an earlier one is omitted. Instead match each
        tuple item's origin against a boundary name's start value; an
        unmatched item is fresh, consumed in order against unmatched names.
        """
        carried = helper._available_carried_values(graph, transition.block)
        dynamic = helper._runtime_values(transition)
        named = helper._named_start_arguments(graph)

        # Origin-matching by identity misreads a shift-register boundary
        # (e.g. SOM's s0->s1->s2 stack cache): an old-s0-now-s1 item's
        # origin is s0's, wrongly read as "s0, unchanged". When nothing
        # is elided, zip dynamic straight against original_order instead.
        ordered = [name for name in original_order if name in named
                  and name in self.boundary_names]
        if len(dynamic) == len(ordered):
            return dict(zip(ordered, dynamic))

        origins = _boundary_value_origins(graph)
        fresh = []
        unchanged_names = set()
        for item in dynamic:
            item_origin = origins.get(item, item)
            matched = None
            for name in self.boundary_names:
                if name not in named or name in unchanged_names:
                    continue
                start_var = named[name]
                if origins.get(start_var, start_var) is item_origin:
                    matched = name
                    break
            if matched is not None:
                unchanged_names.add(matched)
            else:
                fresh.append(item)

        fresh = iter(fresh)
        values = {}
        for name in self.boundary_names:
            if name not in named:
                continue
            if name in unchanged_names or name in self.null_names:
                values[name] = carried.get(name)
                continue
            try:
                values[name] = next(fresh)
            except StopIteration:
                values[name] = carried.get(name)
        return values

    def _tail_block(self, named, values, index, exit_block):
        from rpython.flowspace.model import Block, Link
        from rpython.rtyper.lltypesystem import lltype

        args = []
        inputargs = []
        for name in self.boundary_names:
            var = named[name]
            value = values.get(name)
            args.append(var if value is None else value)
            inputargs.append(self._like(var))
        tail = Block(inputargs)
        tail.operations = ()
        tail.closeblock(Link([Constant(index, lltype.Signed)], exit_block))
        return args, tail

    def _tail_registers(self, regallocs, tail, entry):
        from rpython.jit.metainterp.history import getkind
        if tail is None:
            return {}
        result = {}
        for name, var in zip(self.boundary_names, tail.inputargs):
            kind = getkind(var.concretetype)
            try:
                result[name] = (kind, regallocs[kind].getcolor(var))
            except KeyError:
                # Untouched all the way through: still in its entry register.
                result[name] = entry.get(name)
        return result

    def _reorder_arguments(self, graph, named):
        """Boundary arguments first, in the order the portal supplies
        them; everything else is already a constant, so its position no
        longer matters. Returns the (name, variable) pairs in new order.
        """
        from rpython.flowspace.argument import Signature

        ordered = [(name, named[name]) for name in self.boundary_names
                   if name in named]
        ordered += [(name, named[name]) for name in self.jit_merge_point_args
                    if name in named and name not in self.boundary_names]
        taken = set(name for name, _var in ordered)
        ordered.extend((name, var) for name, var in
                       zip(graph.signature[0], graph.startblock.inputargs)
                       if name not in taken)
        graph.startblock.inputargs = [var for _name, var in ordered]
        # A fresh Signature, not an in-place edit: copygraph(shallowvars=
        # True) aliases the same Signature across every compile() call for
        # one template, so mutating it would corrupt the other fragments.
        graph.signature = Signature(
            [name for name, _var in ordered],
            graph.signature.varargname, graph.signature.kwargname)
        return ordered

    def _entry_registers(self, ordered):
        """Where each boundary value arrives -- enforce_input_args puts
        start arguments in the lowest registers of each kind, in order."""
        from rpython.jit.metainterp.history import getkind
        counts = {}
        result = {}
        for name, var in ordered:
            kind = getkind(var.concretetype)
            if kind == "void":
                continue
            index = counts.get(kind, 0)
            counts[kind] = index + 1
            if name in self.boundary_names:
                result[name] = (kind, index)
        return result


class ProgramEmitter(object):
    """Emits a generated program by concatenating fragments. No register
    renaming is needed between them: boundary values live in the
    registers enforce_input_args pinned them to. Only the exits are
    rewritten, from ``return <index>`` into moves plus a jump.
    """

    def __init__(self, codewriter, portal_jd, static_name, split_names,
                 hole_names, boundary_names, jit_merge_point_args=(),
                 null_names=()):
        self.compiler = FragmentCompiler(
            codewriter, portal_jd, static_name, hole_names, boundary_names,
            jit_merge_point_args, null_names)
        self.codewriter = codewriter
        self.split_names = tuple(split_names)
        self.boundary_names = tuple(boundary_names)
        self._fragments = {}

    def fragment_for(self, block, merge_point=False):
        # late-static values don't enter the key: the fragment carries
        # them as holes, patched in per block by _place/_localise.
        key = (block.key, merge_point)
        if key not in self._fragments:
            self._fragments[key] = self.compiler.compile(
                block.template, block.bindings, merge_point)
        return self._fragments[key]

    def precompile_fragments(self, templates, state_names=()):
        """Compile every opcode's fragment(s) before any program exists.

        split_names (typically "pc") must be holed here too: baking a
        dummy pc in as a real constant would still assemble and even run
        correctly for the one block matching it, but every other
        placement of the shared fragment would carry it wrong.
        """
        bindings = dict.fromkeys(self.compiler.hole_names, 0)
        bindings.update(dict.fromkeys(self.split_names, 0))
        bindings.update(dict.fromkeys(state_names, 0))
        variants = (False, True) if self.compiler.jit_merge_point_args \
            else (False,)
        for key, template in templates.items():
            for merge_point in variants:
                cache_key = (key, merge_point)
                if cache_key not in self._fragments:
                    self._fragments[cache_key] = self.compiler.compile(
                        template, bindings, merge_point)

    def native_table(self):
        """{opcode_key: (native_no_merge, native_merge)}.

        Does not stamp descr indices: pe_linked_setup runs before
        codewriter.make_jitcodes(), so assembler.descrs is still empty
        here -- see stamp_descr_indices instead.
        """
        from rpython.translator.backendopt.native_fragments import (
            build_native_table)
        return build_native_table(self._fragments)

    def emit(self, program, name="emitted-residual"):
        from rpython.jit.codewriter.assembler import JitCode
        from rpython.jit.codewriter.flatten import Label, SSARepr, TLabel
        from rpython.jit.codewriter.liveness import compute_liveness
        from rpython.rtyper.lltypesystem import llmemory
        from rpython.translator.backendopt.partialeval_template import (
            uses_compact_entries)

        if (self.compiler.portal_jd is not None
                and not self.compiler.jit_merge_point_args):
            raise ValueError(
                "emitting a program needs jit_merge_point_args")
        headers = set()
        if self.compiler.jit_merge_point_args:
            headers = set(program.loop_headers) | set([program.entry_pc])
        compact_entries = uses_compact_entries(program)
        fragments = dict((pc, self.fragment_for(block, pc in headers))
                         for pc, block in program.blocks.items())
        num_regs = self._widest(fragments.values())
        scratch = dict((kind, count) for kind, count in num_regs.items())

        # One more register per kind than any fragment used, as scratch
        # for breaking cycles in the parallel moves between fragments.
        counts = dict((kind, scratch[kind] + 1) for kind in scratch)

        ssarepr = SSARepr(name)
        # The entry block goes first, not behind a prologue goto: that
        # goto would target the entry position, which the metainterp
        # takes for a back edge, closing an empty loop at trace start.
        order = [program.entry_pc] + sorted(
            pc for pc in program.blocks if pc != program.entry_pc)
        for index, pc in enumerate(order):
            # Every entry is a jump, never a fall-through, so liveness
            # must not carry anything across the boundary.
            ssarepr.insns.append(("---",))
            ssarepr.insns.append((Label(("block", pc)),))
            if (self.compiler.jit_merge_point_args
                    and (not compact_entries or pc in headers)):
                self._initialise_scratch(ssarepr, fragments, counts)
            self._place(ssarepr, program, pc, fragments, scratch)

        compute_liveness(ssarepr)
        self.last_ssarepr = ssarepr
        self.last_program = program
        jitcode = JitCode(name, fnaddr=llmemory.NULL)
        self.codewriter.assembler.assemble(ssarepr, jitcode, counts)
        entry_positions = dict(
            (pc, self.codewriter.assembler.label_positions[("block", pc)])
            for pc in program.blocks)
        return jitcode, entry_positions

    # ------------------------------------------------------------------

    def _initialise_scratch(self, ssarepr, fragments, counts):
        """Give every register above the calling convention a defined
        value.

        Only needed where a merge point snapshots mid-program:
        compute_liveness over-approximates around a loop, so a register a
        fragment writes before reading can still come out live at the
        loop's entry, and here block boundaries assign only the calling
        convention, so an over-approximated live set can name a register
        no path has written. These values are dead by construction, so
        only that they hold something matters, not what.
        """
        from rpython.flowspace.model import Constant
        from rpython.rtyper.lltypesystem import llmemory, lltype

        entry = {}
        for fragment in fragments.values():
            for kind, index in fragment.boundary_entry.values():
                entry[kind] = max(entry.get(kind, 0), index + 1)
        for kind in sorted(counts):
            if kind == "ref":
                zero = Constant(lltype.nullptr(llmemory.GCREF.TO),
                                llmemory.GCREF)
            elif kind == "int":
                zero = Constant(0, lltype.Signed)
            else:
                continue
            for index in range(entry.get(kind, 0), counts[kind]):
                ssarepr.insns.append(
                    ("%s_copy" % kind, zero, "->", self._register(kind, index)))

    def _widest(self, fragments):
        from rpython.jit.codewriter.flatten import KINDS
        return dict(
            (kind, max([0] + [f.num_regs.get(kind, 0) for f in fragments]))
            for kind in KINDS)

    def _register(self, kind, index):
        from rpython.jit.codewriter.flatten import Register
        return Register(kind, index)

    def _place(self, ssarepr, program, pc, fragments, scratch):
        """Copy a fragment in, rewriting its exits to jump to successors."""
        from rpython.jit.codewriter.flatten import Label, TLabel

        from rpython.flowspace.model import Constant
        from rpython.rtyper.lltypesystem import lltype

        fragment = fragments[pc]
        block = program.blocks[pc]
        for kind, index, name in fragment.prologue:
            ssarepr.insns.append(
                ("%s_copy" % kind,
                 Constant(block.bindings[name], lltype.Signed),
                 "->", self._register(kind, index)))
        from rpython.translator.backendopt.partialeval_template import (
            flatten_resolved_targets)
        targets = flatten_resolved_targets(
            block.template.resolve_targets(block.bindings), len(fragment.exits))

        for insn in fragment.insns:
            exit_index = self._exit_index(insn)
            if exit_index is None:
                ssarepr.insns.append(self._localise(insn, pc, block.bindings))
                continue
            exit = fragment.exits[exit_index]
            target = targets[exit_index]
            # Into the successor's own entry registers: two fragments
            # allocate independently.
            self._emit_moves(ssarepr, exit.operands,
                             fragments[target].boundary_entry, scratch)
            # No -live- before the goto: it generates no guard, so an
            # extra one would only widen the fragments' own live sets.
            ssarepr.insns.append(("goto", TLabel(("block", target))))

    def _exit_index(self, insn):
        from rpython.flowspace.model import Constant
        if len(insn) == 2 and insn[0] == "int_return" and \
                isinstance(insn[1], Constant):
            return insn[1].value
        return None

    def _localise(self, insn, pc, bindings):
        """Copy one fragment insn in for this placement: relabel, and
        patch every HoleConstant, including one hidden inside a
        jit_merge_point/pe_bailout_point green/red ListOfKind.

        A marker's own "pc" green means this block's leading pc, not
        bindings["pc"] (a Continue exit's *next* pc).
        """
        from rpython.flowspace.model import Constant
        from rpython.jit.codewriter.flatten import Label, TLabel, ListOfKind
        from rpython.jit.codewriter.jitcode import SwitchDictDescr
        is_marker = insn and insn[0] in ("jit_merge_point", "pe_bailout_point")
        out = []
        for item in insn:
            if isinstance(item, Label):
                out.append(Label(("in", pc, item.name)))
            elif isinstance(item, TLabel):
                out.append(TLabel(("in", pc, item.name)))
            elif isinstance(item, HoleConstant):
                out.append(Constant(
                    bindings[item.hole_name], item.concretetype))
            elif isinstance(item, ListOfKind):
                out.append(ListOfKind(item.kind, [
                    self._patch_hole(x, pc, bindings, is_marker)
                    if isinstance(x, HoleConstant) else x
                    for x in item.content]))
            elif isinstance(item, SwitchDictDescr):
                # Shared with every other placement of this fragment, so
                # it must be copied rather than renamed in place.
                fresh = SwitchDictDescr()
                fresh._labels = [(key, TLabel(("in", pc, label.name)))
                                 for key, label in item._labels]
                out.append(fresh)
            else:
                out.append(item)
        return tuple(out)

    def _patch_hole(self, hole, pc, bindings, is_marker):
        from rpython.flowspace.model import Constant
        if is_marker and hole.hole_name == "pc":
            return Constant(pc, hole.concretetype)
        return Constant(bindings[hole.hole_name], hole.concretetype)

    def _emit_moves(self, ssarepr, sources, destinations, scratch):
        """Place each boundary value in the register its successor reads,
        as a parallel move: a destination still somebody's source goes
        through scratch, so a rotation can't lose a value."""
        from rpython.flowspace.model import Constant
        from rpython.rtyper.lltypesystem import llmemory, lltype

        moves = []
        for name, destination in destinations.items():
            if destination is None:
                continue
            kind, index = destination
            source = sources.get(name)
            if source is None:
                if kind == "ref":
                    source = Constant(lltype.nullptr(llmemory.GCREF.TO),
                                      llmemory.GCREF)
                else:
                    source = Constant(0, lltype.Signed)
            elif isinstance(source, tuple):
                if source == destination:
                    continue
                source = self._register(*source)
            moves.append((kind, index, source))

        pending = list(moves)
        emitted = []
        while pending:
            progressed = False
            for move in list(pending):
                kind, index, source = move
                blocked = any(
                    isinstance(other[2], type(self._register(kind, 0))) and
                    other[2].kind == kind and other[2].index == index
                    for other in pending if other is not move)
                if blocked:
                    continue
                emitted.append(move)
                pending.remove(move)
                progressed = True
            if not progressed:
                # A cycle: break it by parking one source in scratch.
                kind, index, source = pending[0]
                park = self._register(kind, scratch[kind])
                emitted.append((kind, scratch[kind], source))
                pending[0] = (kind, index, park)

        for kind, index, source in emitted:
            ssarepr.insns.append(
                ("%s_copy" % kind, source, "->", self._register(kind, index)))


def stamp_descr_indices(codewriter, native_table):
    """Stamp pe_descr_index on every descr, once assembler.descrs is
    final. Call only after codewriter.make_jitcodes().

    Two passes: stamp what's already in assembler.descrs, then walk
    native_table's own fragments for any NDescr the first pass missed
    (calldescrof isn't guaranteed to cache) and append + stamp those too.
    NativeSwitchDictDescr operands are skipped: never prebuilt, so never
    stamped -- write_insn still grows the shared list for those.
    """
    from rpython.translator.backendopt.native_fragments import NDescr
    from rpython.translator.backendopt.native_pipeline import (
        NativeSwitchDictDescr)

    descrs = codewriter.assembler.descrs
    for i in range(len(descrs)):
        descrs[i].pe_descr_index = i

    found_only_in_fragments = 0
    for pair in native_table.values():
        for fragment in pair:
            if fragment is None:
                continue
            for insn in fragment.insns:
                for operand in insn.operands:
                    if not isinstance(operand, NDescr):
                        continue
                    d = operand.descr
                    if isinstance(d, NativeSwitchDictDescr):
                        continue
                    if d.pe_descr_index < 0:
                        d.pe_descr_index = len(descrs)
                        descrs.append(d)
                        found_only_in_fragments += 1
    if found_only_in_fragments:
        # Plain print: this runs at translation-build time, not inside
        # the translated binary, so debug_print doesn't apply.
        print "[offline-pe] stamped %d descr(s) only a fragment " \
              "referenced (no ordinary jitcode did)" % found_only_in_fragments


# Insns emit_native can produce outside any fragment's own insns, so not
# discoverable by walking native_table. Must track _place_native/
# _emit_moves_native/_initialise_scratch_native if those change.
_EMIT_NATIVE_CONNECTIVE_TISSUE = [
    "goto/L",
    "int_copy/i>i", "int_copy/c>i",
    "ref_copy/r>r",
    "float_copy/f>f", "float_copy/i>f",
]


def register_native_insn_coverage(codewriter, native_table):
    """Register every (opname, argcodes) a runtime_cogen program's
    readonly NativeAssembler could need, in codewriter.assembler.insns.

    Must run after make_jitcodes(), before MetaInterpStaticData.
    finish_setup: BlackholeInterpBuilder/opcode_implementations both take
    a one-time snapshot of this table the moment finish_setup runs.
    """
    from rpython.jit.codewriter.assembler import USE_C_FORM
    from rpython.translator.backendopt.native_pipeline import (
        native_insn_key_options)

    insns = codewriter.assembler.insns
    for pair in native_table.values():
        for fragment in pair:
            if fragment is None:
                continue
            for insn in fragment.insns:
                keys = native_insn_key_options(insn)
                if keys is None:
                    continue
                for key in keys:
                    insns.setdefault(key, len(insns))
            for kind, _index, _name in fragment.prologue:
                copy_name = "%s_copy" % kind
                allow_short = copy_name in USE_C_FORM
                for letter in (["c", "i"] if allow_short else ["i"]):
                    insns.setdefault(
                        "%s/%s>%s" % (copy_name, letter, kind[0]), len(insns))
    for key in _EMIT_NATIVE_CONNECTIVE_TISSUE:
        insns.setdefault(key, len(insns))
