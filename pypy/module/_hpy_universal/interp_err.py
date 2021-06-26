from rpython.rlib.nonconst import NonConstant
from rpython.rtyper.lltypesystem import rffi, lltype, ll2ctypes
from rpython.rlib.objectmodel import we_are_translated
from pypy.interpreter.error import OperationError, oefmt
from pypy.module._hpy_universal.apiset import API
from pypy.module._hpy_universal.bridge import BRIDGE
from pypy.module._hpy_universal import llapi
from pypy.module._hpy_universal.interp_unicode import _maybe_utf8_to_w

## HPy exceptions in PyPy
## ======================
##
## HPy exceptions are implemented using normal RPython exceptions, which means
## that e.g. HPyErr_SetString simply raises an OperationError: see
## e.g. test_exception_transform.test_llhelper_can_raise for a test which
## ensure that exceptions correctly propagate.
##
## Moreover, we need to ensure that it is NOT possible to call RPython code
## when an RPython exception is set, else you get unexpected results. The plan
## is to document that it's forbidden to call most HPy functions if an
## exception has been set, apart for few functions, such as:
##
##     - HPyErr_Occurred()
##     - HPyErr_Fetch()
##     - HPyErr_Clear()
##
## We need to enforce this in debug mode.
##
## ~~~ Implementation ~~~
##
## HPyErr_SetString, HPyErr_Occurred and HPyErr_Clear are implemented in C. See also:
##    - src/hpyerr.c for the source code
##    - state.py:setup.ctx which explicitly stores the C functions in the ctx


## ~~~ @BRIDGE Functions ~~~
## These functions are called from hpyerr.c, and are used only in tests

@BRIDGE.func("void _hpy_err_SetString(HPyContext ctx, HPy type, const char *message)")
def _hpy_err_SetString(space, handles, ctx, h_exc_type, utf8):
    w_obj = _maybe_utf8_to_w(space, utf8)
    w_exc_type = handles.deref(h_exc_type)
    raise OperationError(w_exc_type, w_obj)

@BRIDGE.func("void _hpy_err_SetObject(HPyContext ctx, HPy type, HPy value)")
def _hpy_err_SetObject(space, handles, ctx, h_exc_type, h_exc_value):
    w_exc_type = handles.deref(h_exc_type)
    w_obj = handles.deref(h_exc_value)
    raise OperationError(w_exc_type, w_obj)

@BRIDGE.func("int hpy_err_Occurred_rpy(void)", error_value=API.int(-1))
def hpy_err_Occurred_rpy(space, handles):
    if we_are_translated():
        # this function should never been called after translation. We can't
        # simply put an assert else the annotator complains that the function
        # returns Void, hack hack hack
        if NonConstant(False):
            return API.int(-42)
        assert False
    #
    # this is a bit of a hack: it will never aim to be correct in 100% of
    # cases, but since it's used only for tests, it's enough.  If an
    # exception was raised by an HPy call, it must be stored in
    # ll2ctypes._callback_exc_info, waiting to be properly re-raised as
    # soon as we exit the C code, by
    # ll2ctypes:get_ctypes_trampoline:invoke_via_ctypes
    res = ll2ctypes._callback_exc_info is not None
    return API.int(res)

@BRIDGE.func("void hpy_err_Clear(void)")
def hpy_err_Clear(space, handles):
    assert not we_are_translated()
    ll2ctypes._callback_exc_info = None


## ~~~ API Functions ~~~~
## The following are normal @API functions, so they contain the "real"
## implementation.

@API.func("HPy HPyErr_NoMemory(HPyContext ctx)")
def HPyErr_NoMemory(space, handles, ctx):
    # hack to convince the annotator that this function returns an HPy (i.e.,
    # a Signed)
    if NonConstant(False):
        return -42
    raise OperationError(space.w_MemoryError, space.w_None)
