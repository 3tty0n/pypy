from rpython.rtyper.lltypesystem import lltype, rffi
from rpython.rlib.objectmodel import specialize, always_inline, we_are_translated
from rpython.rlib.unroll import unrolling_iterable
from pypy.module._hpy_universal import llapi

CONSTANTS = [
    ('NULL', lambda space: None),
    # Constants
    ('None', lambda space: space.w_None),
    ('True', lambda space: space.w_True),
    ('False', lambda space: space.w_False),
    ('NotImplemented', lambda space: space.w_NotImplemented),
    ('Ellipsis', lambda space: space.w_Ellipsis),
    # Exceptions
    ('BaseException', lambda space: space.w_BaseException),
    ('Exception', lambda space: space.w_Exception),
    ('StopAsyncIteration', lambda space: space.w_StopAsyncIteration),
    ('StopIteration', lambda space: space.w_StopIteration),
    ('GeneratorExit', lambda space: space.w_GeneratorExit),
    ('ArithmeticError', lambda space: space.w_ArithmeticError),
    ('LookupError', lambda space: space.w_LookupError),
    ('AssertionError', lambda space: space.w_AssertionError),
    ('AttributeError', lambda space: space.w_AttributeError),
    ('BufferError', lambda space: space.w_BufferError),
    ('EOFError', lambda space: space.w_EOFError),
    ('FloatingPointError', lambda space: space.w_FloatingPointError),
    ('OSError', lambda space: space.w_OSError),
    ('ImportError', lambda space: space.w_ImportError),
    ('ModuleNotFoundError', lambda space: space.w_ModuleNotFoundError),
    ('IndexError', lambda space: space.w_IndexError),
    ('KeyError', lambda space: space.w_KeyError),
    ('KeyboardInterrupt', lambda space: space.w_KeyboardInterrupt),
    ('MemoryError', lambda space: space.w_MemoryError),
    ('NameError', lambda space: space.w_NameError),
    ('OverflowError', lambda space: space.w_OverflowError),
    ('RuntimeError', lambda space: space.w_RuntimeError),
    ('RecursionError', lambda space: space.w_RecursionError),
    ('NotImplementedError', lambda space: space.w_NotImplementedError),
    ('SyntaxError', lambda space: space.w_SyntaxError),
    ('IndentationError', lambda space: space.w_IndentationError),
    ('TabError', lambda space: space.w_TabError),
    ('ReferenceError', lambda space: space.w_ReferenceError),
    ('SystemError', lambda space: space.w_SystemError),
    ('SystemExit', lambda space: space.w_SystemExit),
    ('TypeError', lambda space: space.w_TypeError),
    ('UnboundLocalError', lambda space: space.w_UnboundLocalError),
    ('UnicodeError', lambda space: space.w_UnicodeError),
    ('UnicodeEncodeError', lambda space: space.w_UnicodeEncodeError),
    ('UnicodeDecodeError', lambda space: space.w_UnicodeDecodeError),
    ('UnicodeTranslateError', lambda space: space.w_UnicodeTranslateError),
    ('ValueError', lambda space: space.w_ValueError),
    ('ZeroDivisionError', lambda space: space.w_ZeroDivisionError),
    ('BlockingIOError', lambda space: space.w_BlockingIOError),
    ('BrokenPipeError', lambda space: space.w_BrokenPipeError),
    ('ChildProcessError', lambda space: space.w_ChildProcessError),
    ('ConnectionError', lambda space: space.w_ConnectionError),
    ('ConnectionAbortedError', lambda space: space.w_ConnectionAbortedError),
    ('ConnectionRefusedError', lambda space: space.w_ConnectionRefusedError),
    ('ConnectionResetError', lambda space: space.w_ConnectionResetError),
    ('FileExistsError', lambda space: space.w_FileExistsError),
    ('FileNotFoundError', lambda space: space.w_FileNotFoundError),
    ('InterruptedError', lambda space: space.w_InterruptedError),
    ('IsADirectoryError', lambda space: space.w_IsADirectoryError),
    ('NotADirectoryError', lambda space: space.w_NotADirectoryError),
    ('PermissionError', lambda space: space.w_PermissionError),
    ('ProcessLookupError', lambda space: space.w_ProcessLookupError),
    ('TimeoutError', lambda space: space.w_TimeoutError),
    # Warnings
    ('Warning', lambda space: space.w_Warning),
    ('UserWarning', lambda space: space.w_UserWarning),
    ('DeprecationWarning', lambda space: space.w_DeprecationWarning),
    ('PendingDeprecationWarning', lambda space: space.w_PendingDeprecationWarning),
    ('SyntaxWarning', lambda space: space.w_SyntaxWarning),
    ('RuntimeWarning', lambda space: space.w_RuntimeWarning),
    ('FutureWarning', lambda space: space.w_FutureWarning),
    ('ImportWarning', lambda space: space.w_ImportWarning),
    ('UnicodeWarning', lambda space: space.w_UnicodeWarning),
    ('BytesWarning', lambda space: space.w_BytesWarning),
    ('ResourceWarning', lambda space: space.w_ResourceWarning),
    # Types
    ('BaseObjectType', lambda space: space.w_object),
    ('TypeType', lambda space: space.w_type),
    ('LongType', lambda space: space.w_int),
    ('UnicodeType', lambda space: space.w_unicode),
    ('TupleType', lambda space: space.w_tuple),
    ('ListType', lambda space: space.w_list),
]

class AbstractHandleManager(object):

    def new(self, w_object):
        raise NotImplementedError

    def close(self, index):
        raise NotImplementedError

    def deref(self, index):
        raise NotImplementedError

    def consume(self, index):
        raise NotImplementedError

    def dup(self, index):
        raise NotImplementedError

    def attach_release_callback(self, index, cb):
        raise NotImplementedError

    def using(self, *w_objs):
        """
        context-manager to new/close one or more handles
        """
        # Here we are using some RPython trickery to create different classes
        # depending on the number of w_objs. The idea is that the whole class is
        # optimized away and what's left is a series of calls to handles.new() and
        # handles.close()
        # XXX: That doesn't work: __enter__ and __exit__ can't be inlined
        UsingContextManager = make_UsingContextManager(len(w_objs))
        return UsingContextManager(self, w_objs)


class HandleManager(AbstractHandleManager):

    def __init__(self, uctx, space):
        self.ctx = uctx
        self.handles_w = [build_value(space) for name, build_value in CONSTANTS]
        self.release_callbacks = [None] * len(self.handles_w)
        self.free_list = []

    def new(self, w_object):
        if len(self.free_list) == 0:
            index = len(self.handles_w)
            self.handles_w.append(w_object)
            self.release_callbacks.append(None)
        else:
            index = self.free_list.pop()
            self.handles_w[index] = w_object
            # releasers[index] is already set to None by close()
        return index

    def close(self, index):
        assert index > 0
        if self.release_callbacks[index] is not None:
            w_obj = self.deref(index)
            for f in self.release_callbacks[index]:
                f.release(index, w_obj)
            self.release_callbacks[index] = None
        self.handles_w[index] = None
        self.free_list.append(index)

    def deref(self, index):
        assert index > 0
        return self.handles_w[index]

    def consume(self, index):
        """
        Like close, but also return the w_object which was pointed by the handled
        """
        assert index > 0
        w_object = self.handles_w[index]
        self.close(index)
        return w_object

    def dup(self, index):
        w_object = self.handles_w[index]
        return self.new(w_object)

    def attach_release_callback(self, index, cb):
        if self.release_callbacks[index] is None:
            self.release_callbacks[index] = [cb]
        else:
            self.release_callbacks[index].append(cb)


class DebugHandleManager(AbstractHandleManager):

    def __init__(self, dctx, u_handles):
        self.ctx = dctx
        self.u_handles = u_handles

    def new(self, w_object):
        uh = self.u_handles.new(w_object)
        return llapi.hpy_debug_open_handle(self.ctx, uh)

    def close(self, dh):
        uh = llapi.hpy_debug_unwrap_handle(dh)
        llapi.hpy_debug_close_handle(self.ctx, dh)
        self.u_handles.close(uh)

    def deref(self, dh):
        uh = llapi.hpy_debug_unwrap_handle(dh)
        return self.u_handles.deref(uh)

    def consume(self, dh):
        uh = llapi.hpy_debug_unwrap_handle(dh)
        llapi.hpy_debug_close_handle(self.ctx, dh)
        return self.u_handles.consume(uh)


class HandleReleaseCallback(object):

    def release(self, h, w_obj):
        raise NotImplementedError


class FreeNonMovingBuffer(HandleReleaseCallback):
    """
    Callback to call rffi.free_nonmovingbuffer_ll
    """

    def __init__(self, llbuf, llstring, flag):
        self.llbuf = llbuf
        self.llstring = llstring
        self.flag = flag

    def release(self, h, w_obj):
        rffi.free_nonmovingbuffer_ll(self.llbuf, self.llstring, self.flag)


@specialize.memo()
def make_UsingContextManager(N):
    INDICES = unrolling_iterable(range(N))
    class UsingContextManager(object):

        @always_inline
        def __init__(self, mgr, w_objects):
            self.mgr = mgr
            self.w_objects = w_objects
            self.handles = (0,) * N

        #@always_inline
        def __enter__(self):
            handles = ()
            for i in INDICES:
                h = self.mgr.new(self.w_objects[i])
                handles += (h,)
            self.handles = handles

            # if we have only one handle, return it directly. This makes it
            # possible to write this:
            #     with handles.using(w1) as h1:
            #         ...
            # AND this
            #     with handles.using(w1, w2) as (h1, h2):
            #         ...
            if N == 1:
                return self.handles[0]
            else:
                return handles

        #@always_inline
        def __exit__(self, etype, evalue, tb):
            for i in INDICES:
                self.mgr.close(self.handles[i])

    return UsingContextManager
