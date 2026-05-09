"""
Inline depth policy for TLA tier-1 (CALL* hint bytes, entry defaults, global cap).
"""
from rpython.rlib import jit

# Hint byte on CALL_H / CALL_ASSEMBLER_H; sixth byte on CALL_N_H
INLINE_HINT_DEFAULT = 0
INLINE_HINT_FORCE_SHALLOW = 1
INLINE_HINT_ALLOW_DEEP_1 = 2

_global_inline_cap = 0
_entry_inline_hints = {}

def set_global_inline_cap(n):
    global _global_inline_cap
    _global_inline_cap = int(n)

def get_global_inline_cap():
    return _global_inline_cap

def register_entry_inline_hint(entry_pc, value):
    _entry_inline_hints[int(entry_pc)] = int(value)

def clear_entry_inline_hints():
    _entry_inline_hints.clear()

@jit.dont_look_inside
def _entry_extra(callee_pc):
    return _entry_inline_hints.get(int(callee_pc), 0)

def compute_child_inline_budget(parent_budget, callee_pc, hint_byte):
    cap = get_global_inline_cap()
    site = 0
    if hint_byte == INLINE_HINT_FORCE_SHALLOW:
        site = 0
    elif hint_byte == INLINE_HINT_ALLOW_DEEP_1:
        site = 1 if cap > 0 else 0
    elif hint_byte == INLINE_HINT_DEFAULT:
        site = _entry_extra(callee_pc)
        if site > cap:
            site = cap
    else:
        site = 0
    if site > cap:
        site = cap
    return max(parent_budget - 1, 0) + site
