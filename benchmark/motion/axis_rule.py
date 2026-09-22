"""A change rule: the same vector, addressed along the other axis.

An interpreter rule, not a transition.  It says what a layout change *means*
and nothing about how it is carried out.

    A square matrix m of side n and a vector v of n values combine
    elementwise.  The vector is addressed along one of the two axes:

        along columns:  out[i, j] = m[i, j] + v[j]
        along rows:     out[i, j] = m[i, j] + v[i]

    The change class is the move from one addressing to the other.  The
    values of v are the same values; what changes is which index reads them.

That equation is the whole rule.  It names no storage, no device, no kernel,
no tile, no broadcast constant, and it does not ask whether m exists yet.

`apply` hands the rule to the interpreter in the interpreter's own
vocabulary: a vector shaped [1, n] is read along columns and one shaped
[n, 1] along rows, which is the interpreter's existing way of saying which
index reads a vector.  The meta-tracer specialises it with whatever else it
is tracing.

Checked by benchmark/paper/audit_rules.py.
"""


def index_map(axis, i, j):
    """Which element of v the output at (i, j) reads."""
    return j if axis == 'columns' else i


def vector_shape(axis, n):
    return [1, n] if axis == 'columns' else [n, 1]


def apply(m, v):
    """The rule, in the interpreter's vocabulary.  `v` already carries the
    axis in its shape, which is how the interpreter spells the distinction."""
    return m.add(v)


def reference(mvals, vvals, n, axis):
    """The rule on plain numbers, so an implementation is checked against the
    equation rather than against another implementation."""
    out = 0.0
    for i in range(n):
        for j in range(n):
            out += mvals[i * n + j] + vvals[index_map(axis, i, j)]
    return out
