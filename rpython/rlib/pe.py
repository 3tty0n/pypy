"""Runtime support for offline partial evaluation.

A file of scalars shadowing array slots whose indices are late-static.

An interpreter keeps state in arrays: a register file, an operand stack, a
frame's locals.  Meta-tracing records every access to such an array, and those
records survive optimisation whenever the array escapes -- which it does as
soon as it is handed to a variadic call.  The values are already known; only
the array is not.

When the *index* of an access is late-static -- resolved once per instruction
by the offline partial evaluator, rather than carried at runtime -- the value
can be held in an ordinary variable instead, and the array need not be touched
at all.

This module is indifferent to what the array means.  It offers a file of
``FILE_SIZE`` scalars addressed by a slot number that the caller derives from
its own addressing scheme:

    a register machine   slot = the decoded register number
    an operand stack     slot = depth - position, from a late-static depth
    a frame's locals     slot = the decoded local index

In each case the slot number is a link-time constant, so every test below is a
comparison between constants and folds away; an access landing inside the file
compiles to nothing, and one landing outside falls back to the array.

Two shapes of use are supported.  Direct addressing (registers, locals) needs
only ``read`` and ``write``.  Sliding-window addressing (a stack, a ring) also
needs ``shift_in``/``shift_out``, which move the whole file by one slot and
exchange the slot falling off the end with the array.  Building a push and a
pop out of those two is a couple of lines, and belongs to the interpreter that
knows its own stack discipline rather than here.

Around anything that receives the array itself, ``spill`` and ``refill``
exchange the file with the array; their index arguments spell out the mapping,
so this module never has to assume one.

The file is flat scalars rather than a tuple or an object on purpose.  Either
of those survives as an allocation across the step-function boundary, which is
exactly the traffic being removed.  ``FILE_SIZE`` is therefore part of the
signatures: widening the file means changing them and the interpreter's step
function together.
"""

from rpython.rtyper.extregistry import ExtRegistryEntry
from rpython.rlib.objectmodel import always_inline, specialize

FILE_SIZE = 3

#: Slot numbers at or above this live in the array rather than the file.
OUT_OF_FILE = FILE_SIZE


def _names(value):
    """Argument names, written either as a string or as a sequence."""
    if isinstance(value, str):
        return tuple(value.split())
    return tuple(value)


def value_file(array, scalars, late_static):
    """Declare an array whose late-static accesses are held in scalars.

    ``array`` names the argument holding the array, ``scalars`` the arguments
    standing in for its hot slots, and ``late_static`` the arguments whose
    values must be known at link time for the slot numbers to be constants.
    Every name in ``late_static`` has to be a split argument; that is what
    makes the tests in this module fold, and without it the file costs more
    than it saves.  Name lists may be written as a string or a list.

    Pass the result to PEDriver's ``value_file``.
    """
    return (array, _names(scalars), _names(late_static))


def check_value_file(func):
    """Raise if a declared value file cannot actually be held in scalars.

    Returns the declaration, or None when the function does not declare one.
    """
    declaration = getattr(func, "_pe_value_file_", None)
    if declaration is None:
        return None
    array, scalars, late_static = declaration
    split = getattr(func, "_pe_split_args_", ())
    for name in late_static:
        if name not in split:
            raise ValueError(
                "%r declares %r as late-static for its value file, but the "
                "split arguments are %r -- an index that is not known at link "
                "time leaves the file's tests in the residual code"
                % (func, name, split))
    if len(scalars) != FILE_SIZE:
        raise ValueError(
            "%r declares %d value-file slots, but this module implements %d"
            % (func, len(scalars), FILE_SIZE))
    argnames = func.func_code.co_varnames[:func.func_code.co_argcount]
    for name in (array,) + tuple(scalars) + tuple(late_static):
        if name not in argnames:
            raise ValueError(
                "%r declares value-file name %r, which is not one of its "
                "arguments %r" % (func, name, argnames))
    return declaration


@specialize.arg(0)
@always_inline
def read(slot, index, array, v0, v1, v2):
    """The value in ``slot``, falling back to ``array[index]``.

    ``slot`` must be a constant; ``index`` is only evaluated for a slot outside
    the file.
    """
    if slot == 0:
        return v0
    if slot == 1:
        return v1
    if slot == 2:
        return v2
    return array[index]


@specialize.arg(0)
@always_inline
def write(slot, index, value, array, v0, v1, v2):
    """Store ``value`` in ``slot``, or in ``array[index]`` if it is outside.

    Returns the new file.  ``slot`` must be a constant.
    """
    if slot == 0:
        return value, v1, v2
    if slot == 1:
        return v0, value, v2
    if slot == 2:
        return v0, v1, value
    array[index] = value
    return v0, v1, v2


@always_inline
def shift_in(value, evict_index, array, v0, v1, v2):
    """Move the file up by one, ``value`` entering at slot 0.

    The slot pushed off the end is written to ``array[evict_index]``, and only
    then; a negative index means nothing is evicted because the file is not yet
    full.
    """
    if evict_index >= 0:
        array[evict_index] = v2
    return value, v0, v1


@always_inline
def shift_out(fill_index, array, v0, v1, v2):
    """Move the file down by one, dropping slot 0.

    The slot entering at the end is read from ``array[fill_index]``; a negative
    index means there is nothing left to read and the slot becomes empty.
    """
    if fill_index >= 0:
        return v1, v2, array[fill_index]
    return v1, v2, None


@always_inline
def spill(array, i0, i1, i2, v0, v1, v2):
    """Write the file back, before handing the array to someone else.

    A negative index marks a slot that the array does not own, so it is
    skipped.
    """
    if i0 >= 0:
        array[i0] = v0
    if i1 >= 0:
        array[i1] = v1
    if i2 >= 0:
        array[i2] = v2


@always_inline
def refill(array, i0, i1, i2):
    """Reload the file, after a callee may have rewritten the array."""
    v0 = array[i0] if i0 >= 0 else None
    v1 = array[i1] if i1 >= 0 else None
    v2 = array[i2] if i2 >= 0 else None
    return v0, v1, v2


class PEDriver(object):
    """Declares how an interpreter may be partially evaluated, ahead of time.

    The shape mirrors JitDriver: one object naming the variables and their
    roles, and a marker called at the point it describes.

        pedriver = PEDriver(static="opcode", split="pc stack_ptr",
                            holes="oparg2 send_argc",
                            never=SELF_MODIFYING, min_size=20)

        def _interp_step(opcode, oparg, oparg2, send_argc, pc, stack_ptr,
                         s0, s1, s2, method, frame, stack):
            pedriver.pe_merge_point(
                opcode=opcode, oparg=oparg, oparg2=oparg2,
                send_argc=send_argc, pc=pc, stack_ptr=stack_ptr,
                s0=s0, s1=s1, s2=s2,
                method=method, frame=frame, stack=stack)
            ...

    ``static`` is fixed when the interpreter is translated -- the instruction
    being executed, so there is one residual template per instruction.
    ``split`` is unknown then but known once a program is chosen, and the
    residual code branches on it.  ``holes`` are late-static too but only flow
    through, so they become typed slots rather than branches.  Everything else
    named at the merge point stays dynamic.  Name lists may be written as one
    space-separated string.

    ``never`` names instructions the evaluator must leave alone -- ones that
    rewrite their own bytecode, where a generated program would be stale the
    moment it ran.  ``min_size`` is the smallest program worth generating, in
    instructions: what specializing saves is the dispatch removed from every
    instruction executed, so it grows with the method, while what it costs --
    a JitCode in the binary, one more guard at every trace start -- does not.
    ``worth_generating`` takes a predicate instead, for a judgement size cannot
    make.

    Calling the merge point checks that its keywords are exactly the step
    function's arguments, and binds this declaration to the function it is
    called from, the way jit_merge_point binds a JitDriver to its loop.
    """

    name = "pedriver"

    def __init__(self, static, split=(), holes=(), never=(), min_size=0,
                 worth_generating=None, value_file=None):
        self.static = _names(static)
        self.split = _names(split)
        self.holes = _names(holes)
        self.never = tuple(never)
        self.link_policy = worth_generating or _at_least(min_size)
        self.value_file = value_file
        self._make_extregistryentries()

    def _freeze_(self):
        # A declaration, not data: the annotator treats it as a constant, which
        # is what lets pe_merge_point resolve to its registry entry.
        return True

    def pe_merge_point(_self, **livevars):
        # special-cased by ExtRegistryEntry below
        pass

    def bind(self, func):
        """Attach this declaration to a step function.

        Called for you when the merge point is annotated; also usable directly
        for a function the annotator never sees, as in a test.
        """
        func._pe_entry_point_ = True
        func._pe_static_args_ = self.static
        func._pe_split_args_ = self.split
        func._pe_hole_args_ = self.holes
        func._pe_skip_keys_ = self.never
        func._pe_link_policy_ = self.link_policy
        if self.value_file is not None:
            array, scalars, late_static = self.value_file
            func._pe_value_file_ = (array, _names(scalars),
                                    _names(late_static))
        return func

    def _make_extregistryentries(self):
        # As in JitDriver: an ExtRegistryEntry cannot be declared for a method
        # of a frozen object, so the bound method is attached back to self.
        self.pe_merge_point = self.pe_merge_point

        class Entry(ExtPEMergePoint):
            _about_ = self.pe_merge_point


def _at_least(instructions):
    if not instructions:
        return None

    def worth_generating(program, code):
        return len(program.blocks) >= instructions

    return worth_generating


class ExtPEMergePoint(ExtRegistryEntry):
    """Bind a PEDriver to the function its merge point is called from.

    Nothing is emitted: the partial evaluator reads the declaration off the
    function, so the marker leaves no trace in the residual code.
    """

    def compute_result_annotation(self, **kwds_s):
        from rpython.annotator import model as annmodel

        driver = self.instance.im_self
        graph = self.bookkeeper.position_key[0]
        declared = set(driver.static + driver.split + driver.holes)
        given = set(name[2:] for name in kwds_s)
        missing = declared - given
        if missing:
            raise PEHintError(
                "%s names %s, which pe_merge_point was not given"
                % (driver.name, ", ".join(sorted(missing))))
        arguments = set(graph.func.func_code.co_varnames[
            :graph.func.func_code.co_argcount])
        stray = given - arguments
        if stray:
            raise PEHintError(
                "pe_merge_point was given %s, which %s does not take"
                % (", ".join(sorted(stray)), graph.func.__name__))
        driver.bind(graph.func)
        return annmodel.s_None

    def specialize_call(self, hop, **kwds_i):
        from rpython.rtyper.lltypesystem import lltype
        # Nothing is emitted: the marker exists to be read by the annotator.
        hop.exception_cannot_occur()
        return hop.inputconst(lltype.Void, None)


class PEHintError(Exception):
    """Raised when a pe_merge_point disagrees with its driver."""
