from pypy.interpreter.error import oefmt
from rpython.rtyper.lltypesystem import rffi, lltype
from pypy.module.cpyext.api import (
    cpython_api, cpython_struct, bootstrap_function, build_type_checkers,
    PyVarObjectFields, Py_ssize_t, CONST_STRING, CANNOT_FAIL)
from pypy.module.cpyext.pyerrors import PyErr_BadArgument
from pypy.module.cpyext.pyobject import (
    PyObject, PyObjectP, Py_DecRef, make_ref, from_ref, track_reference,
    make_typedescr, get_typedescr, as_pyobj, Py_IncRef, get_w_obj_and_decref)

##
## Implementation of PyBytesObject
## ================================
##
## The problem
## -----------
##
## PyBytes_AsString() must return a (non-movable) pointer to the underlying
## ob_sval, whereas pypy strings are movable.  C code may temporarily store
## this address and use it, as long as it owns a reference to the PyObject.
## There is no "release" function to specify that the pointer is not needed
## any more.
##
## Also, the pointer may be used to fill the initial value of string. This is
## valid only when the string was just allocated, and is not used elsewhere.
##
## Solution
## --------
##
## PyBytesObject contains two additional members: the ob_size and a pointer to a
## char ob_sval; it may be NULL.
##
## - A string allocated by pypy will be converted into a PyBytesObject with a
##   NULL buffer.  The first time PyBytes_AsString() is called, memory is
##   allocated (with flavor='raw') and content is copied.
##
## - A string allocated with PyBytes_FromStringAndSize(NULL, size) will
##   allocate a PyBytesObject structure, and a buffer with the specified
##   size+1, but the reference won't be stored in the global map; there is no
##   corresponding object in pypy.  When from_ref() or Py_INCREF() is called,
##   the pypy string is created, and added to the global map of tracked
##   objects.  The buffer is then supposed to be immutable.
##
##-  A buffer obtained from PyBytes_AS_STRING() could be mutable iff
##   there is no corresponding pypy object for the string
##
## - _PyBytes_Resize() works only on not-yet-pypy'd strings, and returns a
##   similar object.
##
## - PyBytes_Size() doesn't need to force the object.
##
## - There could be an (expensive!) check in from_ref() that the buffer still
##   corresponds to the pypy gc-managed string.
##

PyBytesObjectStruct = lltype.ForwardReference()
PyBytesObject = lltype.Ptr(PyBytesObjectStruct)
PyBytesObjectFields = PyVarObjectFields + \
    (("ob_shash", rffi.LONG), ("ob_sstate", rffi.INT), ("ob_sval", rffi.CArray(lltype.Char)))
cpython_struct("PyBytesObject", PyBytesObjectFields, PyBytesObjectStruct)

@bootstrap_function
def init_bytesobject(space):
    "Type description of PyBytesObject"
    make_typedescr(space.w_str.layout.typedef,
                   basestruct=PyBytesObject.TO,
                   attach=bytes_attach,
                   dealloc=bytes_dealloc,
                   realize=bytes_realize)

PyBytes_Check, PyBytes_CheckExact = build_type_checkers("Bytes", "w_bytes")

def new_empty_str(space, length):
    """
    Allocate a PyBytesObject and its ob_sval, but without a corresponding
    interpreter object.  The ob_sval may be mutated, until bytes_realize() is
    called.  Refcount of the result is 1.
    """
    typedescr = get_typedescr(space.w_bytes.layout.typedef)
    py_obj = typedescr.allocate(space, space.w_bytes)
    py_str = rffi.cast(PyBytesObject, py_obj)
    py_str.c_ob_shash = -1
    py_str.c_ob_sstate = rffi.cast(rffi.INT, 0) # SSTATE_NOT_INTERNED
    return py_str

def bytes_attach(space, py_obj, w_obj):
    """
    Copy RPython string object contents to a PyBytesObject. The
    c_ob_sval must not be modified.
    """
    py_str = rffi.cast(PyBytesObject, py_obj)
    s = space.str_w(w_obj)
    if py_str.c_ob_size  < len(s):
        raise oefmt(space.w_ValueError,
            "bytes_attach called on object with ob_size %d but trying to store %d",
            py_str.c_ob_size, len(s)) 
    rffi.c_memcpy(py_str.c_ob_sval, rffi.str2charp(s), len(s))
    py_str.c_ob_sval[len(s)] = '\0'
    py_str.c_ob_shash = space.hash_w(w_obj)
    py_str.c_ob_sstate = rffi.cast(rffi.INT, 1) # SSTATE_INTERNED_MORTAL

def bytes_realize(space, py_obj):
    """
    Creates the string in the interpreter. The PyBytesObject ob_sval must not
    be modified after this call.
    """
    py_str = rffi.cast(PyBytesObject, py_obj)
    s = rffi.charpsize2str(py_str.c_ob_sval, py_str.c_ob_size)
    w_type = from_ref(space, rffi.cast(PyObject, py_obj.c_ob_type))
    w_obj = space.allocate_instance(W_BytesObject, w_type)
    w_obj.__init__(s)
    py_str.c_ob_shash = space.hash_w(w_obj)
    py_str.c_ob_sstate = rffi.cast(rffi.INT, 1) # SSTATE_INTERNED_MORTAL
    track_reference(space, py_obj, w_obj)
    return w_obj

@cpython_api([PyObject], lltype.Void, header=None)
def bytes_dealloc(space, py_obj):
    """Frees allocated PyBytesObject resources.
    """
    from pypy.module.cpyext.object import PyObject_dealloc
    PyObject_dealloc(space, py_obj)

#_______________________________________________________________________

@cpython_api([CONST_STRING, Py_ssize_t], PyObject, result_is_ll=True)
def PyBytes_FromStringAndSize(space, char_p, length):
    if char_p:
        s = rffi.charpsize2str(char_p, length)
        return make_ref(space, space.wrapbytes(s))
    else:
        return rffi.cast(PyObject, new_empty_str(space, length))

@cpython_api([CONST_STRING], PyObject)
def PyBytes_FromString(space, char_p):
    s = rffi.charp2str(char_p)
    return space.wrapbytes(s)

@cpython_api([PyObject], rffi.CCHARP, error=0)
def PyBytes_AsString(space, ref):
    return _PyBytes_AsString(space, ref)

def _PyBytes_AsString(space, ref):
    if from_ref(space, rffi.cast(PyObject, ref.c_ob_type)) is space.w_str:
        pass    # typecheck returned "ok" without forcing 'ref' at all
    elif not PyBytes_Check(space, ref):   # otherwise, use the alternate way
        raise oefmt(space.w_TypeError,
            "expected bytes, %T found", from_ref(space, ref))
    ref_str = rffi.cast(PyBytesObject, ref)
    if not pyobj_has_w_obj(ref):
        # XXX Force the ref?
        bytes_realize(space, ref)
    return ref_str.c_ob_sval

@cpython_api([rffi.VOIDP], rffi.CCHARP, error=0)
def PyBytes_AS_STRING(space, void_ref):
    ref = rffi.cast(PyObject, void_ref)
    # if no w_str is associated with this ref,
    # return the c-level ptr as RW
    if not pyobj_has_w_obj(ref):
        py_str = rffi.cast(PyBytesObject, ref)
        return py_str.c_ob_sval
    return _PyBytes_AsString(space, ref)

@cpython_api([PyObject, rffi.CCHARPP, rffi.CArrayPtr(Py_ssize_t)], rffi.INT_real, error=-1)
def PyBytes_AsStringAndSize(space, ref, data, length):
    if not PyBytes_Check(space, ref):
        raise oefmt(space.w_TypeError,
            "expected bytes, %T found", from_ref(space, ref))
    if not pyobj_has_w_obj(ref):
        # force the ref
        bytes_realize(space, ref)
    ref_str = rffi.cast(PyBytesObject, ref)
    data[0] = ref_str.c_ob_sval
    if length:
        length[0] = ref_str.c_ob_size
    else:
        i = 0
        while ref_str.c_ob_sval[i] != '\0':
            i += 1
        if i != ref_str.c_ob_size:
            raise oefmt(space.w_TypeError,
                        "expected string without null bytes")
    return 0

@cpython_api([PyObject], Py_ssize_t, error=-1)
def PyBytes_Size(space, ref):
    if from_ref(space, rffi.cast(PyObject, ref.c_ob_type)) is space.w_str:
        ref = rffi.cast(PyBytesObject, ref)
        return ref.c_ob_size
    else:
        w_obj = from_ref(space, ref)
        return space.len_w(w_obj)

@cpython_api([PyObjectP, Py_ssize_t], rffi.INT_real, error=-1)
def _PyBytes_Resize(space, ref, newsize):
    """A way to resize a string object even though it is "immutable". Only use this to
    build up a brand new string object; don't use this if the string may already be
    known in other parts of the code.  It is an error to call this function if the
    refcount on the input string object is not one. Pass the address of an existing
    string object as an lvalue (it may be written into), and the new size desired.
    On success, *string holds the resized string object and 0 is returned;
    the address in *string may differ from its input value.  If the reallocation
    fails, the original string object at *string is deallocated, *string is
    set to NULL, a memory exception is set, and -1 is returned.
    """
    # XXX always create a new string so far
    if pyobj_has_w_obj(ref[0]):
        raise oefmt(space.w_SystemError,
                    "_PyBytes_Resize called on already created string")
    py_str = rffi.cast(PyBytesObject, ref[0])
    try:
        py_newstr = new_empty_str(space, newsize)
    except MemoryError:
        Py_DecRef(space, ref[0])
        ref[0] = lltype.nullptr(PyObject.TO)
        raise
    to_cp = newsize
    oldsize = py_str.c_ob_size
    if oldsize < newsize:
        to_cp = oldsize
    for i in range(to_cp):
        py_newstr.c_ob_sval[i] = py_str.c_ob_sval[i]
    Py_DecRef(space, ref[0])
    ref[0] = rffi.cast(PyObject, py_newstr)
    return 0

@cpython_api([PyObject, PyObject], rffi.INT, error=CANNOT_FAIL)
def _PyBytes_Eq(space, w_str1, w_str2):
    return space.eq_w(w_str1, w_str2)

@cpython_api([PyObjectP, PyObject], lltype.Void, error=None)
def PyBytes_Concat(space, ref, w_newpart):
    """Create a new string object in *string containing the contents of newpart
    appended to string; the caller will own the new reference.  The reference to
    the old value of string will be stolen.  If the new string cannot be created,
    the old reference to string will still be discarded and the value of
    *string will be set to NULL; the appropriate exception will be set."""

    old = ref[0]
    if not old:
        return

    ref[0] = lltype.nullptr(PyObject.TO)
    w_str = get_w_obj_and_decref(space, old)
    if w_newpart is not None and PyBytes_Check(space, old):
        # XXX: should use buffer protocol
        w_newstr = space.add(w_str, w_newpart)
        ref[0] = make_ref(space, w_newstr)

@cpython_api([PyObjectP, PyObject], lltype.Void, error=None)
def PyBytes_ConcatAndDel(space, ref, newpart):
    """Create a new string object in *string containing the contents of newpart
    appended to string.  This version decrements the reference count of newpart."""
    try:
        PyBytes_Concat(space, ref, newpart)
    finally:
        Py_DecRef(space, newpart)

@cpython_api([PyObject, PyObject], PyObject)
def _PyBytes_Join(space, w_sep, w_seq):
    return space.call_method(w_sep, 'join', w_seq)

@cpython_api([PyObject], PyObject)
def PyBytes_FromObject(space, w_obj):
    """Return the bytes representation of object obj that implements
    the buffer protocol."""
    if space.is_w(space.type(w_obj), space.w_bytes):
        return w_obj
    buffer = space.buffer_w(w_obj, space.BUF_FULL_RO)
    return space.wrapbytes(buffer.as_str())
