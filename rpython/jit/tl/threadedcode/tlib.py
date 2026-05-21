from rpython.rlib import jit


def emit_jump(pc, t):
    from rpython.rtyper.lltypesystem import lltype
    from rpython.rtyper.lltypesystem.lloperation import llop
    llop.jit_emit_jump(lltype.Void, t)
    return pc


def emit_ret(pc, w_x):
    from rpython.rtyper.lltypesystem import lltype
    from rpython.rtyper.lltypesystem.lloperation import llop
    llop.jit_emit_ret(lltype.Void, w_x)
    return pc
