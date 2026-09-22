"""Negative control: the same rule, written by someone who knows the target.

An audit that accepts everything establishes nothing, so this file exists to
be rejected.  It is the rule of layout_rule.py with three things added, each
of which a person who had read the transition code might write without
noticing that it is the adapter:

  * a test of what the source representation currently is;
  * a name from the machinery that produces the target;
  * a tile-sized constant chosen to match the kernel that will run.

benchmark/paper/audit_rules.py must fail this file.  If it ever passes, the
firewall has stopped working and the audit of layout_rule.py means nothing.
"""
from rpython.metatensor import kernels


def blocked_shape(rows, cols, heads):
    return heads * rows, cols // heads


def apply(t, heads):
    if t.is_virtual():                      # a test of the source
        kernels.add_output(t.kernel, 0)     # a name from the target
    if t.cols % 4096 == 0:                  # a tile-sized constant
        return t.head_split(heads)
    return t.head_split(heads)
