"""Scalar shadow file for array slots whose indices are late-static."""

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

    late_static names must be split args, so slot numbers fold to
    constants; pass the result to PEDriver's value_file."""
    return (array, _names(scalars), _names(late_static))


def check_value_file(func):
    """Raise if a declared value file cannot be held in scalars."""
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
    """The value in slot (a constant), falling back to array[index]."""
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
    """Store value in slot (a constant), or array[index]; returns the file."""
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
    """Move the file up by one, value entering at slot 0.

    A negative evict_index means the file is not yet full: nothing evicted."""
    if evict_index >= 0:
        array[evict_index] = v2
    return value, v0, v1


@always_inline
def shift_out(fill_index, array, v0, v1, v2):
    """Move the file down by one, dropping slot 0.

    A negative fill_index means nothing left to read; slot becomes empty."""
    if fill_index >= 0:
        return v1, v2, array[fill_index]
    return v1, v2, None


@always_inline
def spill(array, i0, i1, i2, v0, v1, v2):
    """Write the file back to array; a negative index skips that slot."""
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
    """Declares how an interpreter may be partially evaluated, ahead of
    time.  Mirrors JitDriver: static is fixed at translation (one
    template per instruction), split is known once a program is chosen
    and branches the residual code, holes are late-static but flow
    through untested, never lists self-modifying instructions to skip,
    and min_size/worth_generating gate which programs are worth it."""

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
        """Attach this declaration to a step function."""
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
    """Bind a PEDriver to the function its merge point is called from."""

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


def residualize(func):
    """Mark a helper to be inlined into the PE entry graph at PE time,
    leaving the interpreter's own call to it untouched."""
    func._pe_residualize_ = True
    return func
