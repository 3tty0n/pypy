"""Assemble each residual template once, and emit a program by concatenation.

The whole-program lowerer in ``partialeval_template`` builds one FunctionGraph
for a generated program and hands it to the codewriter.  That works, but it
puts the codewriter *after* generation, which is why generation cannot happen
anywhere but during translation: ``transform_graph`` resolves each call into a
JitCode to enter or a residual call to record, and that needs the whole-program
view a translated binary no longer has.

Nothing the codewriter does actually depends on the program, though.  Only
three things do:

  * the values filling the templates' holes,
  * which templates get strung together,
  * where the branches go.

So the codewriter can run once per *template*, offline, and emitting a program
becomes concatenate-and-patch.  This module is that arrangement.

A fragment is the instruction list the flattener produces for one template,
with two properties that make it relocatable:

  * its boundary values arrive in the lowest registers of each kind, which
    ``GraphFlattener.enforce_input_args`` already guarantees, and leave in
    registers the exit records;
  * its late-static values are ``HoleConstant`` placeholders, replaced when a
    concrete instruction supplies them.
"""

from rpython.flowspace.model import Constant

#: A value no real late-static operand is likely to take, so a placeholder
#: that escapes patching shows up as an obviously wrong number rather than as
#: a plausible one.
HOLE_SENTINEL = 0x5E7717E1


def _boundary_value_origins(graph):
    """Like partialeval_template's _variable_origins, but also sees through
    a same-block setfield/getfield round trip (RPython's graph simplifier
    can merge two tail blocks into an unpack-then-repack of one tuple) and
    through cast_pointer/same_as, which don't change what a value denotes.
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
                elif op.opname == "getfield" and isinstance(op.args[1], Constant):
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
    """A Constant standing in for a late-static value until supplied.

    It behaves as an ordinary Constant through transformation, register
    allocation and flattening, so the hole survives into the instruction list
    at exactly the positions the value is used.
    """

    def __init__(self, name, concretetype):
        Constant.__init__(self, HOLE_SENTINEL, concretetype)
        self.hole_name = name

    def __repr__(self):
        return "hole(%s)" % (self.hole_name,)


class FragmentExit(object):
    """One way out of a fragment.

    ``operands`` holds, per boundary name, the ssarepr operand carrying that
    value where the fragment leaves off -- a register, or a constant.  The
    concatenator turns those into moves into the calling convention before the
    jump to the successor.
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
        # Register each boundary value arrives in, by name.
        self.boundary_entry = boundary_entry
        # (kind, index, value) triples to assign before the fragment runs.
        self.prologue = tuple(prologue)


class FragmentCompiler(object):
    """Run the codewriter once per template, producing relocatable fragments.

    The template is reshaped first so that each of its exits becomes a plain
    ``return <exit index>``: that is a shape the flattener handles, and it
    leaves a mark the concatenator can find and turn into a jump.  Where the
    boundary values sit at that point is read out of the register allocation
    rather than guessed.
    """

    def __init__(self, codewriter, portal_jd, static_name, hole_names,
                 boundary_names, jit_merge_point_args=(), null_names=()):
        self.codewriter = codewriter
        self.portal_jd = portal_jd
        self.static_name = static_name
        self.hole_names = tuple(hole_names)
        self.boundary_names = tuple(boundary_names)
        self.jit_merge_point_args = tuple(jit_merge_point_args)
        # Names declared invariant for the whole portal (never freshly
        # returned by any opcode) -- can't be inferred from one template.
        self.null_names = tuple(null_names)

    def compile(self, template, bindings={}, merge_point=False):
        from rpython.flowspace.model import Block, Link, Variable, copygraph
        from rpython.jit.codewriter.flatten import flatten_graph, KINDS
        from rpython.jit.codewriter.jtransform import transform_graph
        from rpython.jit.codewriter.regalloc import perform_register_allocation
        from rpython.rtyper.lltypesystem import lltype
        from rpython.translator.backendopt.partialeval import (
            _find_split_transitions, replace_uses)
        from rpython.translator.backendopt.partialeval_template import (
            Finish, LinkedResidualLowerer)

        graph = copygraph(template.residual_graph, shallowvars=True)
        # Snapshot the argument order before _reorder_arguments (below)
        # overwrites graph.signature with the portal's boundary-name
        # order: _boundary_values needs the original order to zip
        # against the exit tuple, which follows it, not the portal's.
        original_order = list(graph.signature[0])
        named = dict(zip(graph.signature[0], graph.startblock.inputargs))
        self._thread_boundary_values(graph, named)
        replace_uses(graph, self._placeholders(template, named, bindings))
        helper = LinkedResidualLowerer(
            self.boundary_names, (), (), self.jit_merge_point_args)
        helper.portal_jd = self.portal_jd
        if merge_point and self.jit_merge_point_args:
            # A real merge point, exactly as the whole-graph lowerer plants
            # one.  Without it the metainterp has to recognise the back edge by
            # position, and the -live- record it then needs cannot be placed:
            # the assembler hoists a label above a -live- that follows it.
            helper._remove_runtime_loop_markers(graph)
            helper._insert_merge_point(graph)
        elif self.jit_merge_point_args:
            # Every other block boundary gets a cheap pe_bailout_point
            # instead.  It carries the same greens+reds a merge point here
            # would, but is a no-op while tracing -- see
            # opimpl_pe_bailout_point.  What it buys is a place for the
            # *blackhole* interpreter to bail out of this residual jitcode
            # one block at a time, instead of running all the way to the
            # next real jit_merge_point (a whole method away).
            helper._insert_bailout_point(graph)
        # Substitution comes after the merge point so that its greens are the
        # bound values: a merge point on an unbound pc names every instruction
        # of the program alike, and the metainterp identifies loops by it.
        # Its *reds* may not be constants, though -- the jitdriver rejects that
        # -- so a late-static red is assigned into its register instead.
        reds = self._bound_reds(named, bindings) if merge_point else ()
        replace_uses(graph, self._placeholders(
            template, named, bindings, skip=reds))
        # And the reorder comes after both, because ``_insert_merge_point``
        # rebuilds the name-to-argument map from the signature: the portal
        # fills the entry registers in *its* order, not the step function's.
        ordered = self._reorder_arguments(graph, named)

        transitions = _find_split_transitions(graph)
        terminators = self._align(template, len(transitions))
        # The step function returns its late-static state ahead of the dynamic
        # values.  Skipping the wrong number of them shifts every boundary
        # value by one, which shows up as a copy between mismatched kinds.
        helper.state_names = self._state_names(terminators)
        helper.state_count = len(helper.state_names)
        finishing = [isinstance(t, Finish) for t in terminators]
        if any(finishing) and not all(finishing):
            # One graph has one return block, so its type has to be either the
            # exit index or the program's result -- it cannot be both.
            raise ValueError("template %r mixes Finish and Continue exits"
                             % (template.key,))

        if all(finishing):
            # Return the result itself: after ``_remove_result_tuple`` nothing
            # else uses it, and a value no exit consumes gets no register.
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
            # A tail block per exit, taking every boundary value: nothing else
            # uses them once the result tuple is gone, and the allocator only
            # places values that something consumes.  Where they land is then
            # read off the tail's own arguments.
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
        """Without this, an unused-here boundary name has no live copy in a
        later block, and falling back to the start-block variable lets the
        register allocator silently reuse its color for something else.
        """
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
        """The late-static state the template carries, read off its exits."""
        for terminator in terminators:
            state = getattr(terminator, "state", ())
            if state:
                return tuple(name for name, _expression in state)
        return ()

    def _align(self, template, count):
        """One terminator per split transition."""
        terminators = template.terminators
        if len(terminators) == 1 and count > 1:
            # RTyping can keep equivalent syntactic exits that the symbolic
            # template already canonicalised into one terminator.
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
        """Which merge-point reds this opcode knows late-statically, by
        name -- only bindings' keys matter, not its values."""
        if self.portal_jd is None:
            return ()
        jitdriver = self.portal_jd.jitdriver
        reds = self.jit_merge_point_args[len(jitdriver.greens):]
        return [name for name in reds if name in bindings and name in named]

    def _prologue(self, ordered, reds):
        """Where to put each late-static red before the merge point sees it
        -- only the register is fixed here; ProgramEmitter._place fills
        the value in from block.bindings."""
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
        """Fix the static key; every other late-static name gets a hole,
        not bindings' value -- one fragment is shared by every instance
        of the opcode, so no one instance's value belongs baked in."""
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
        """Which value carries each boundary name where this exit leaves off.

        The exit tuple can omit a boundary name whose value is unchanged
        (see _thread_boundary_values), with no marker saying so -- so a
        naive positional zip against boundary_names silently misassigns
        every later name once an earlier one turns out to be omitted.
        Fix: match each tuple item's origin against a boundary name's own
        start value; an unmatched item is genuinely fresh, consumed in
        order against whichever names had no match.

        Plain _variable_origins isn't enough here: it only follows block
        links, but RPython's simplifier can turn a template's exit tuple
        into a same-block "unpack, then repack" round trip;
        _boundary_value_origins extends origin-tracking through that too.
        """
        carried = helper._available_carried_values(graph, transition.block)
        dynamic = helper._runtime_values(transition)
        named = helper._named_start_arguments(graph)

        # Origin-matching (below) places a name the exit tuple omits by
        # value identity, which can't tell "name X unchanged" from "name
        # Y now holding X's old value" -- exactly what a shift-register
        # boundary does on every write (SOM's s0/s1/s2: incoming s0
        # becomes the new s1 unmodified), so it mismatches names by one.
        #
        # Skip that guessing when nothing was elided: if every boundary
        # name shows up in this exit's own tuple (true of SOM's
        # _interp_step), a plain positional zip against the step
        # function's *original* argument order is exact.
        # original_order, not graph.signature[0]: _reorder_arguments has
        # already overwritten the latter with the portal's own
        # boundary-name order, which zips wrong.
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
            # null_names never take a fresh slot, even one going begging.
            if name in unchanged_names or name in self.null_names:
                values[name] = carried.get(name)
                continue
            try:
                values[name] = next(fresh)
            except StopIteration:
                values[name] = carried.get(name)
        return values

    def _tail_block(self, named, values, index, exit_block):
        """A block taking every boundary value, so each one gets a register."""
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
        """Boundary arguments first, in the order the portal supplies them.

        Everything else is a late-static value the placeholders already turned
        into a constant, so where it sits no longer matters.

        Returns the (name, variable) pairs in their new order.
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
        # graph.signature[0] must stay positionally matched to inputargs
        # (_named_start_arguments zips them). A fresh Signature, not an
        # in-place edit: copygraph(shallowvars=True) aliases the same
        # Signature object across every compile() call for one template,
        # so mutating it would corrupt every other fragment sharing it.
        graph.signature = Signature(
            [name for name, _var in ordered],
            graph.signature.varargname, graph.signature.kwargname)
        return ordered

    def _entry_registers(self, ordered):
        """Where each boundary value arrives.

        ``enforce_input_args`` puts the start arguments in the lowest registers
        of each kind, in order, so this needs no lookup.
        """
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
    """Emit a generated program by concatenating fragments.

    No register renaming is needed between fragments.  Boundary values live in
    the registers ``enforce_input_args`` pinned them to, and every fragment
    re-establishes them on the way out, so what a fragment does with the
    registers above those is its own business and cannot disturb its
    neighbours.  Only the exits are rewritten: the ``return <index>`` a
    fragment ends on becomes the moves into the calling convention followed by
    a jump to whichever block the generated program says comes next.
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
        """The fragment for this block, keyed on (opcode, merge_point) --
        late-static values don't enter the key, since the fragment carries
        them as holes patched in per block by _place/_localise."""
        key = (block.key, merge_point)
        if key not in self._fragments:
            self._fragments[key] = self.compiler.compile(
                block.template, block.bindings, merge_point)
        return self._fragments[key]

    def precompile_fragments(self, templates, state_names=()):
        """Compile every opcode's fragment(s) before any program exists.

        split_names (typically "pc") must be holed here too, even though
        every real block's bindings carries it anyway: baking a dummy pc
        in as a real constant would still assemble, and even run
        correctly for the one block matching the dummy, but every other
        placement of the shared fragment would silently carry it wrong.
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

        Does NOT stamp descr indices here: pe_linked_setup runs strictly
        before codewriter.make_jitcodes(), so assembler.descrs is still
        empty at this point -- see stamp_descr_indices below instead.
        """
        from rpython.translator.backendopt.native_fragments import (
            build_native_table)
        return build_native_table(self._fragments)

    def emit(self, program, name="emitted-residual"):
        from rpython.jit.codewriter.assembler import JitCode
        from rpython.jit.codewriter.flatten import Label, SSARepr, TLabel
        from rpython.jit.codewriter.liveness import compute_liveness
        from rpython.rtyper.lltypesystem import llmemory

        if (self.compiler.portal_jd is not None
                and not self.compiler.jit_merge_point_args):
            # The metainterp would then have to spot the back edge by position
            # and read a -live- immediately before it, which cannot be placed:
            # the assembler hoists a label above a -live- that follows it.
            raise ValueError(
                "emitting a program needs jit_merge_point_args")
        headers = set()
        if self.compiler.jit_merge_point_args:
            headers = set(program.loop_headers) | set([program.entry_pc])
        fragments = dict((pc, self.fragment_for(block, pc in headers))
                         for pc, block in program.blocks.items())
        num_regs = self._widest(fragments.values())
        scratch = dict((kind, count) for kind, count in num_regs.items())

        # One more register per kind than any fragment used, for the scratch
        # that breaks cycles in the parallel moves between fragments.
        counts = dict((kind, scratch[kind] + 1) for kind in scratch)

        ssarepr = SSARepr(name)
        # The entry block goes first rather than behind a prologue goto: that
        # goto would target the entry position, which is what the metainterp
        # takes for a back edge, and would close an empty loop at trace start.
        order = [program.entry_pc] + sorted(
            pc for pc in program.blocks if pc != program.entry_pc)
        for index, pc in enumerate(order):
            # Every entry into a block is a jump, never a fall-through, so the
            # liveness scan must not carry anything across the boundary: it
            # would make each block's guards capture registers its own path
            # never wrote.
            ssarepr.insns.append(("---",))
            ssarepr.insns.append((Label(("block", pc)),))
            if self.compiler.jit_merge_point_args:
                # Every block, not only loop headers: the portal now guards on
                # every block boundary (see PortalLinker.install), so any of
                # them may be a trace start, and a block entered from outside
                # has none of the registers above the calling convention set
                # yet.  Reaching a block by an ordinary jump re-runs the
                # copies, which the optimiser removes; reaching it as a trace
                # start is what needs them.  The writes are into registers a
                # fragment never reads before its own code writes them, so
                # running this unconditionally is a no-op except where a
                # snapshot mid-fragment needs the value defined.
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
        """Give every register above the calling convention a defined value.

        Only needed where a merge point snapshots mid-program:
        ``compute_liveness`` over-approximates around a loop, so a register a
        fragment writes before reading still comes out live at the loop's entry.
        In a single lowered graph that is harmless, because every block boundary
        is a link that assigns all of them.  Here the boundaries assign only the
        calling convention, so an over-approximated live set can name a register
        no path has written, and the metainterp finds None when it snapshots.

        These values are dead by construction -- nothing reads them before its
        own fragment writes them -- so what they hold does not matter, only that
        they hold something.
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
        """Copy a fragment in, rewriting its exits to jump to the successors."""
        from rpython.jit.codewriter.flatten import Label, TLabel

        from rpython.flowspace.model import Constant
        from rpython.rtyper.lltypesystem import lltype

        fragment = fragments[pc]
        block = program.blocks[pc]
        for kind, index, name in fragment.prologue:
            # A late-static value the merge point takes as a red: it has to be
            # in a register by the time the merge point reads it.
            # The fragment fixed only the register; value comes from name.
            ssarepr.insns.append(
                ("%s_copy" % kind, Constant(block.bindings[name], lltype.Signed),
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
            # Into the *successor's* entry registers: two fragments allocate
            # independently, so only the successor knows where it reads from.
            self._emit_moves(ssarepr, exit.operands,
                             fragments[target].boundary_entry, scratch)
            # No -live- before the goto: a goto cannot generate a guard, so it
            # needs no snapshot, and an extra one only widens the live sets the
            # fragments' own guards capture.
            ssarepr.insns.append(("goto", TLabel(("block", target))))

    def _exit_index(self, insn):
        from rpython.flowspace.model import Constant
        if len(insn) == 2 and insn[0] == "int_return" and \
                isinstance(insn[1], Constant):
            return insn[1].value
        return None

    def _localise(self, insn, pc, bindings):
        """Copy one fragment insn in for this placement: relabel, and
        patch every HoleConstant -- including one hidden inside a
        jit_merge_point/pe_bailout_point green/red ListOfKind, or it
        survives unpatched as the raw sentinel.

        A marker's own "pc" green means this block's leading pc, not
        bindings["pc"] (a Continue exit's *next* pc) -- same hole name,
        told apart only by whether it sits in a green/red box list.
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
                out.append(Constant(bindings[item.hole_name], item.concretetype))
            elif isinstance(item, ListOfKind):
                out.append(ListOfKind(item.kind, [
                    self._patch_hole(x, pc, bindings, is_marker)
                    if isinstance(x, HoleConstant) else x
                    for x in item.content]))
            elif isinstance(item, SwitchDictDescr):
                # A switch keeps its targets inside the descriptor rather than
                # in the instruction, and the descriptor is the fragment's --
                # shared with every other placement of it, so it has to be
                # copied rather than renamed in place.
                fresh = SwitchDictDescr()
                fresh._labels = [(key, TLabel(("in", pc, label.name)))
                                 for key, label in item._labels]
                out.append(fresh)
            else:
                out.append(item)
        return tuple(out)

    def _patch_hole(self, hole, pc, bindings, is_marker):
        """Value for one HoleConstant inside a green/red box list -- see
        _localise's docstring for the marker "pc" special case."""
        from rpython.flowspace.model import Constant
        if is_marker and hole.hole_name == "pc":
            return Constant(pc, hole.concretetype)
        return Constant(bindings[hole.hole_name], hole.concretetype)

    def _emit_moves(self, ssarepr, sources, destinations, scratch):
        """Place each boundary value in the register its successor reads.

        Done as a parallel move: a destination that is still somebody's source
        goes through a scratch register, so a rotation between two boundary
        values cannot lose one of them.
        """
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
                # A cycle: break it by parking one source in the scratch.
                kind, index, source = pending[0]
                park = self._register(kind, scratch[kind])
                emitted.append((kind, scratch[kind], source))
                pending[0] = (kind, index, park)

        for kind, index, source in emitted:
            ssarepr.insns.append(
                ("%s_copy" % kind, source, "->", self._register(kind, index)))


def stamp_descr_indices(codewriter, native_table):
    """Stamp pe_descr_index on every descr, once assembler.descrs is
    final. Call only after codewriter.make_jitcodes() -- any earlier and
    assembler.descrs is still (nearly) empty (see native_table's note).

    Two passes: stamp what's already in assembler.descrs, then walk
    native_table's own fragments for any NDescr a first pass missed
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
        # Plain print: this runs at translation-build time, never inside
        # the translated binary's own graph, so debug_print doesn't apply.
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
    """Register every (opname, argcodes) combo a readonly NativeAssembler
    could need, into ``codewriter.assembler.insns``. Must run before
    ``MetaInterpStaticData.finish_setup`` snapshots this table (same
    constraint as ``stamp_descr_indices`` above).

    Without this, generated-program-only insns get declined:
    ``pe_bailout_point``/``jit_merge_point`` markers, and hole-patched
    constant-constant arithmetic that ordinary jitcodes never produce
    (RPython's own optimizer folds those away before rtyping).
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
            # fragment.prologue entries are always an NIntConst operand
            # regardless of 'kind' (_place_native); this adapts to
            # whichever kinds a fragment's prologue actually uses.
            for kind, _index, _name in fragment.prologue:
                copy_name = "%s_copy" % kind
                allow_short = copy_name in USE_C_FORM
                for letter in (["c", "i"] if allow_short else ["i"]):
                    insns.setdefault(
                        "%s/%s>%s" % (copy_name, letter, kind[0]), len(insns))
    for key in _EMIT_NATIVE_CONNECTIVE_TISSUE:
        insns.setdefault(key, len(insns))
