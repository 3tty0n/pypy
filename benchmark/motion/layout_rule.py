"""A change rule: the same values, addressed as blocks instead of as a matrix.

This is an interpreter rule, not a transition.  It says what a device/layout
change *means*, in the vocabulary a tensor interpreter already has, and says
nothing about how the change is carried out, what the value was stored in
before, or what will consume it after.

    A tensor is a map from index to value.  For a matrix of `rows` rows and
    `cols` columns, and a block count `heads` dividing `cols`, the blocked
    form has `heads * rows` rows and `cols // heads` columns, and

        blocked[b * rows + i, j]  =  matrix[i, b * (cols // heads) + j]

    The inverse rule reads the same equation right to left.

The equation is the whole rule.  It mentions no storage, no device, no
kernel, no layout descriptor, and no property of whatever produced the
matrix - in particular it does not ask whether the matrix exists yet.

`index_map` is the rule as arithmetic, which is what the reference
implementation and the audit both read.  `apply` is the rule handed to the
interpreter, in the interpreter's own vocabulary; the meta-tracer specialises
it along with everything else it is tracing.

Firewall: this file is checked by benchmark/paper/audit_rules.py, which fails
if it imports anything outside the declared vocabulary, mentions any name
belonging to the target representation, or uses a constant that only means
something once the target is known.  The check is the record; no claim is
made here about who wrote the file.
"""


def index_map(rows, cols, heads, b, i, j):
    """Where blocked[b, i, j] comes from in the matrix: (row, column)."""
    return i, b * (cols // heads) + j


def blocked_shape(rows, cols, heads):
    return heads * rows, cols // heads


def apply(t, heads):
    """The rule, in the interpreter's vocabulary."""
    return t.head_split(heads)


def unapply(t, heads):
    """The same equation, read right to left."""
    return t.head_merge(heads)


def reference(values, rows, cols, heads):
    """The rule executed on plain numbers, for checking an implementation of
    it against the equation rather than against another implementation."""
    br, bc = blocked_shape(rows, cols, heads)
    out = [0.0] * (br * bc)
    for b in range(heads):
        for i in range(rows):
            for j in range(bc):
                r, c = index_map(rows, cols, heads, b, i, j)
                out[(b * rows + i) * bc + j] = values[r * cols + c]
    return out
