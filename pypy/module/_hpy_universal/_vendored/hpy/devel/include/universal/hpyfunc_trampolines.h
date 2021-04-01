#ifndef HPY_UNIVERSAL_HPYFUNC_TRAMPOLINES_H
#define HPY_UNIVERSAL_HPYFUNC_TRAMPOLINES_H

/* This file should be autogenerated */

typedef struct {
    cpy_PyObject *self;
    cpy_PyObject *result;
} _HPyFunc_args_NOARGS;

typedef struct {
    cpy_PyObject *self;
    cpy_PyObject *arg;
    cpy_PyObject *result;
} _HPyFunc_args_O;

typedef struct {
    cpy_PyObject *self;
    cpy_PyObject *args;
    cpy_PyObject *result;
} _HPyFunc_args_VARARGS;

typedef struct {
    cpy_PyObject *self;
    cpy_PyObject *args;
    cpy_PyObject *kw;
    cpy_PyObject *result;
} _HPyFunc_args_KEYWORDS;

typedef struct {
    cpy_PyObject *self;
    cpy_PyObject *args;
    cpy_PyObject *kw;
    int result;
} _HPyFunc_args_INITPROC;


#define _HPyFunc_TRAMPOLINE_HPyFunc_NOARGS(SYM, IMPL)                   \
    static cpy_PyObject *                                               \
    SYM(cpy_PyObject *self, cpy_PyObject *noargs)                       \
    {                                                                   \
        _HPyFunc_args_NOARGS a = { self };                              \
        _HPy_CallRealFunctionFromTrampoline(                            \
            _ctx_for_trampolines, HPyFunc_NOARGS, IMPL, &a);            \
        return a.result;                                                \
    }

#define _HPyFunc_TRAMPOLINE_HPyFunc_O(SYM, IMPL)                        \
    static cpy_PyObject *                                               \
    SYM(cpy_PyObject *self, cpy_PyObject *arg)                          \
    {                                                                   \
        _HPyFunc_args_O a = { self, arg };                              \
        _HPy_CallRealFunctionFromTrampoline(                            \
            _ctx_for_trampolines, HPyFunc_O, IMPL, &a);                 \
        return a.result;                                                \
    }


#define _HPyFunc_TRAMPOLINE_HPyFunc_VARARGS(SYM, IMPL)                  \
    static cpy_PyObject *                                               \
    SYM(cpy_PyObject *self, cpy_PyObject *args)                         \
    {                                                                   \
        _HPyFunc_args_VARARGS a = { self, args };                       \
        _HPy_CallRealFunctionFromTrampoline(                            \
            _ctx_for_trampolines, HPyFunc_VARARGS, IMPL, &a);           \
        return a.result;                                                \
    }


#define _HPyFunc_TRAMPOLINE_HPyFunc_KEYWORDS(SYM, IMPL)                 \
    static cpy_PyObject *                                               \
    SYM(cpy_PyObject *self, cpy_PyObject *args, cpy_PyObject *kw)       \
    {                                                                   \
        _HPyFunc_args_KEYWORDS a = { self, args, kw };                  \
        _HPy_CallRealFunctionFromTrampoline(                            \
            _ctx_for_trampolines, HPyFunc_KEYWORDS, IMPL, &a);          \
        return a.result;                                                \
    }

#define _HPyFunc_TRAMPOLINE_HPyFunc_INITPROC(SYM, IMPL)                 \
    static int                                                          \
    SYM(cpy_PyObject *self, cpy_PyObject *args, cpy_PyObject *kw)       \
    {                                                                   \
        _HPyFunc_args_INITPROC a = { self, args, kw };                  \
        _HPy_CallRealFunctionFromTrampoline(                            \
            _ctx_for_trampolines, HPyFunc_INITPROC, IMPL, &a);          \
        return a.result;                                                \
    }

/* special case: this function is used as 'tp_dealloc', but from the user
   point of view the slot is HPy_tp_destroy. */
#define _HPyFunc_TRAMPOLINE_HPyFunc_DESTROYFUNC(SYM, IMPL)              \
    static void                                                         \
    SYM(cpy_PyObject *self)                                             \
    {                                                                   \
        _HPy_CallDestroyAndThenDealloc(                                 \
            _ctx_for_trampolines, IMPL, self);                          \
    }


#endif // HPY_UNIVERSAL_HPYFUNC_TRAMPOLINES_H