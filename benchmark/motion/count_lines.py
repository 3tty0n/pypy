#!/usr/bin/env python3
"""Hand-written lines on each side of the leave comparison, counted one way.

A line counts when it holds code: blank lines, comment-only lines and
docstrings do not.  Everything else a side needs is counted, including
annotations (_immutable_fields_, jit.promote, dont_look_inside,
unroll_safe) and the application's policy declaration.  The one exclusion is
the harness both sides share - the stand-in for the runtime's notification
channel (Group.arm/poll on the MOTION side, Channel in the adapter) - which
is listed below by name, and its state lines carry `# [harness]`, so the
exclusion is visible.

    count_lines.py [--json]
"""
import ast
import io
import json
import os
import sys
import tokenize

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

# (label, file, what to count).  what: 'all', or a list of
# ('exclude'|'only', [qualified names]), or ('marked', tag) for lines carrying
# a `# [tag]` comment.
MOTION = [
    ('change rule', 'pypy/module/_metatensor/rule_leave.py', 'all'),
    ('rule vocabulary', 'pypy/module/_metatensor/motion_iface.py', 'all'),
    ('domain helpers + annotations',
     'pypy/module/_metatensor/interp_group.py',
     ('exclude', ['W_Group.arm', 'W_Group.descr_arm', 'W_Group.poll'])),
    ('module registration', 'pypy/module/_metatensor/moduledef.py',
     ('marked', 'motion')),
    ('policy declaration', 'benchmark/applevel/leave_probe.py',
     ('marked', 'policy')),
    ('policy handed to the group', 'benchmark/applevel/leave_probe.py',
     ('marked', 'motion-policy')),
]
ADAPTER = [
    ('contributor log + dedup + transition',
     'benchmark/motion/adapter_leave.py', ('only', ['ManualAggregator'])),
    ('policy declaration', 'benchmark/applevel/leave_probe.py',
     ('marked', 'policy')),
]
HARNESS = ['W_Group.arm', 'W_Group.descr_arm', 'W_Group.poll',
           'W_Group.__init__ lines marked [harness]', 'Channel']


def docstring_lines(tree):
    lines = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef)):
            body = node.body
            if (body and isinstance(body[0], ast.Expr) and
                    isinstance(body[0].value, ast.Constant) and
                    isinstance(body[0].value.value, str)):
                lines.update(range(body[0].lineno, body[0].end_lineno + 1))
    return lines


def spans(tree, names):
    """Line ranges of the named classes/functions (Class.method allowed)."""
    out = []

    def visit(node, prefix):
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.ClassDef, ast.FunctionDef)):
                q = prefix + child.name
                if q in names:
                    first = min([d.lineno for d in child.decorator_list] +
                                [child.lineno])
                    out.append((first, child.end_lineno))
                visit(child, q + '.')
    visit(tree, '')
    return out


def code_lines(src):
    lines = set()
    for tok in tokenize.generate_tokens(io.StringIO(src).readline):
        if tok.type in (tokenize.COMMENT, tokenize.NL, tokenize.NEWLINE,
                        tokenize.INDENT, tokenize.DEDENT,
                        tokenize.ENDMARKER):
            continue
        lines.update(range(tok.start[0], tok.end[0] + 1))
    return lines


def count(path, what):
    src = open(os.path.join(ROOT, path)).read()
    tree = ast.parse(src)
    lines = code_lines(src) - docstring_lines(tree)
    if what == 'all':
        return len(lines)
    harness = {i + 1 for i, l in enumerate(src.splitlines())
               if '# [harness]' in l}
    lines = lines - harness
    kind, arg = what
    if kind == 'marked':
        tagged = {i + 1 for i, l in enumerate(src.splitlines())
                  if ('# [%s]' % arg) in l}
        return len(lines & tagged)
    ranges = spans(tree, arg)
    inside = {n for a, b in ranges for n in range(a, b + 1)}
    if kind == 'only':
        return len(lines & inside)
    return len(lines - inside)


def side(items):
    rows = [(label, path, count(path, what)) for label, path, what in items]
    return rows, sum(r[2] for r in rows)


def main(argv):
    m_rows, n_rule = side(MOTION)
    a_rows, n_adapter = side(ADAPTER)
    if '--json' in argv:
        print(json.dumps({'N_rule': n_rule, 'N_adapter': n_adapter,
                          'motion': m_rows, 'adapter': a_rows,
                          'excluded_harness': HARNESS}))
        return 0
    for title, rows, total in (('MOTION (rule + everything around it)',
                                m_rows, n_rule),
                               ('hand-written adapter', a_rows, n_adapter)):
        print(title)
        for label, path, n in rows:
            print('  %4d  %-38s %s' % (n, label, path))
        print('  %4d  total\n' % total)
    print('excluded from both (shared harness): %s' % ', '.join(HARNESS))
    print('N_rule=%d N_adapter=%d' % (n_rule, n_adapter))
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv))
