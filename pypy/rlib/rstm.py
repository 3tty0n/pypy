import threading
from pypy.translator.stm import stmgcintf
from pypy.rlib.debug import ll_assert, fatalerror
from pypy.rlib.objectmodel import keepalive_until_here, specialize
from pypy.rlib.objectmodel import we_are_translated
from pypy.rlib.rposix import get_errno, set_errno
from pypy.rpython.lltypesystem import lltype, llmemory, rffi, rclass
from pypy.rpython.lltypesystem.lloperation import llop
from pypy.rpython.annlowlevel import (cast_instance_to_base_ptr,
                                      llhelper)

def is_inevitable():
    return we_are_translated() and stmgcintf.StmOperations.is_inevitable()

def become_inevitable():
    llop.stm_become_inevitable(lltype.Void)

def should_break_transaction():
    return we_are_translated() and (
        stmgcintf.StmOperations.should_break_transaction())

def set_transaction_length(length):
    stmgcintf.StmOperations.set_transaction_length(length)

def increment_atomic():
    stmgcintf.StmOperations.add_atomic(+1)

def decrement_atomic():
    stmgcintf.StmOperations.add_atomic(-1)

def is_atomic():
    return stmgcintf.StmOperations.get_atomic()

def before_external_call():
    if not is_atomic():
        e = get_errno()
        llop.stm_stop_transaction(lltype.Void)
        stmgcintf.StmOperations.commit_transaction()
        set_errno(e)
before_external_call._dont_reach_me_in_del_ = True
before_external_call._transaction_break_ = True

def after_external_call():
    if not is_atomic():
        e = get_errno()
        stmgcintf.StmOperations.begin_inevitable_transaction()
        llop.stm_start_transaction(lltype.Void)
        set_errno(e)
after_external_call._dont_reach_me_in_del_ = True
after_external_call._transaction_break_ = True

def enter_callback_call():
    token = stmgcintf.StmOperations.descriptor_init()
    if token != 1:
        after_external_call()
    else:
        ll_assert(not is_atomic(), "new thread: is_atomic() != 0")
        stmgcintf.StmOperations.begin_inevitable_transaction()
        # the StmGCTLS is not built yet.  leave it to gc_thread_start()
    return token
enter_callback_call._dont_reach_me_in_del_ = True
enter_callback_call._transaction_break_ = True

def leave_callback_call(token):
    if token != 1:
        before_external_call()
    else:
        # the StmGCTLS is already destroyed, done by gc_thread_die()
        # (we don't care if is_atomic() or not, we'll commit now)
        stmgcintf.StmOperations.commit_transaction()
        stmgcintf.StmOperations.descriptor_done()
leave_callback_call._dont_reach_me_in_del_ = True
leave_callback_call._transaction_break_ = True

# ____________________________________________________________

def make_perform_transaction(func, CONTAINERP):
    #
    def _stm_callback(llcontainer, retry_counter):
        if not is_atomic():
            llop.stm_start_transaction(lltype.Void)
        llcontainer = rffi.cast(CONTAINERP, llcontainer)
        try:
            res = func(llcontainer, retry_counter)
            llcontainer.got_exception = lltype.nullptr(rclass.OBJECT)
        except Exception, e:
            res = 0     # stop perform_transaction() and returns
            lle = cast_instance_to_base_ptr(e)
            llcontainer.got_exception = lle
        if not is_atomic():
            llop.stm_stop_transaction(lltype.Void)
        return res
    #
    def perform_transaction(llcontainer):
        before_external_call()
        adr_of_top = llop.gc_adr_of_root_stack_top(llmemory.Address)
        llcallback = llhelper(stmgcintf.StmOperations.CALLBACK_TX,
                              _stm_callback)
        stmgcintf.StmOperations.perform_transaction(llcallback, llcontainer,
                                                    adr_of_top)
        after_external_call()
        keepalive_until_here(llcontainer)
    perform_transaction._transaction_break_ = True
    #
    return perform_transaction
