#ifndef RPYTHON_LL2CTYPES
#  include "common_header.h"
#  include "structdef.h"
#  include "forwarddecl.h"
#  include "preimpl.h"
#  include "src/exception.h"
#endif

#include <stdio.h>
#include "hpy.h"
#include "hpyerr.h"
#include "bridge.h"


void pypy_HPy_FatalError(HPyContext *ctx, const char *message)
{
    fprintf(stderr, "Fatal Python error: %s\n", message);
    abort();
}

int pypy_HPyErr_Occurred(HPyContext *ctx)
{
#ifdef RPYTHON_LL2CTYPES
    /* before translation */
    return hpy_err_Occurred_rpy();
#else
    /* after translation */
    return RPyExceptionOccurred();
#endif
}

void pypy_HPyErr_SetString(HPyContext *ctx, HPy type, const char *message)
{
#ifndef RPYTHON_LL2CTYPES /* after translation */
    // it is allowed to call this function with an exception set: for now, we
    // just ensure that the exception is cleared before setting it again in
    // hpy_err_SetString. In the future, we might have to add some logic for
    // chaining exceptions.
    RPyClearException();
#endif
    hpy_err_SetString(ctx, type, message);
}

void pypy_HPyErr_SetObject(HPyContext *ctx, HPy type, HPy value)
{
  #ifndef RPYTHON_LL2CTYPES /* after translation */
      // it is allowed to call this function with an exception set: for now, we
      // just ensure that the exception is cleared before setting it again in
      // hpy_err_SetString. In the future, we might have to add some logic for
      // chaining exceptions.
      RPyClearException();
  #endif
      hpy_err_SetObject(ctx, type, value);
}

void pypy_HPyErr_Clear(HPyContext *ctx)
{
#ifdef RPYTHON_LL2CTYPES
    /* before translation */
    hpy_err_Clear();
#else
    /* after translation */
    RPyClearException();
#endif
}
