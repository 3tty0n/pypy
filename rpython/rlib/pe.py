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

from rpython.rlib.objectmodel import always_inline, specialize

FILE_SIZE = 3

#: Slot numbers at or above this live in the array rather than the file.
OUT_OF_FILE = FILE_SIZE


def value_file(array, file, late_static):
    """Declare an array whose late-static accesses are held in scalars.

    ``array`` names the argument holding the array, ``file`` the arguments
    holding the scalars, and ``late_static`` the arguments whose values must be
    known at link time for the slot numbers to be constants.  Every name in
    ``late_static`` has to be a split argument; that is what makes the tests in
    this module fold, and without it the file costs more than it saves.
    """
    def decorate(func):
        func._pe_value_file_ = (array, tuple(file), tuple(late_static))
        return func

    return decorate


def check_value_file(func):
    """Raise if a declared value file cannot actually be held in scalars.

    Returns the declaration, or None when the function does not declare one.
    """
    declaration = getattr(func, "_pe_value_file_", None)
    if declaration is None:
        return None
    array, file, late_static = declaration
    split = getattr(func, "_pe_split_args_", ())
    for name in late_static:
        if name not in split:
            raise ValueError(
                "%r declares %r as late-static for its value file, but the "
                "split arguments are %r -- an index that is not known at link "
                "time leaves the file's tests in the residual code"
                % (func, name, split))
    if len(file) != FILE_SIZE:
        raise ValueError(
            "%r declares %d value-file slots, but this module implements %d"
            % (func, len(file), FILE_SIZE))
    argnames = func.func_code.co_varnames[:func.func_code.co_argcount]
    for name in (array,) + tuple(file) + tuple(late_static):
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
