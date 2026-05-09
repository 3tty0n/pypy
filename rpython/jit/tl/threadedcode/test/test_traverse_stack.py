import py
from rpython.jit.tl.threadedcode.traverse_stack import *

def test_t_empty():
    assert t_empty().t_is_empty()
    assert (not TStack(2, TStack(3, None)).t_is_empty())
    assert (not TStack(42, None).t_is_empty())

def test_t_pop_and_t_push():
    tstack = t_empty()
    tstack = t_push(2, tstack)
    tstack = t_push(3, tstack)
    pc, tstack = tstack.t_pop()
    assert pc == 3
    pc, tstack = tstack.t_pop()
    assert pc == 2 and tstack is t_empty()
