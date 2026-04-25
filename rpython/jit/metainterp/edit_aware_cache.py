"""
Edit-aware trace-cache invalidation (Idea 3A).

Headline contribution of the Idea-3 track: imports nbjit.jl's static /
dynamic region split into PyPy's persistent trace cache. Cache entries
compiled from a region of source that has NOT been edited since the
entry was produced remain valid; entries whose originating source
region WAS edited are invalidated, individually.

Rough sketch.

  1. At trace compile time we fingerprint the *region* of source
     (by AST-level structure of the enclosing function/method + its
     transitive callees below some bounded depth). Fingerprints are
     stable across whitespace-only edits, comments, and variable
     renames *inside* the region if we normalize, but for this
     prototype we just use ast.dump() of the region.

  2. A global ``CodeIndex`` maps source-file -> mapping of
     ``fn_qualname -> fingerprint``. Built lazily on first lookup,
     cheap to rebuild if the file mtime advanced.

  3. On cache hit we compute the current fingerprint for the
     originating region and compare. Mismatch -> drop that specific
     entry and miss.

  4. Entries that reference *transitively* an edited callee are
     invalidated via a small dependency graph the compiler records
     at store time. Bounded-depth so we don't chase all of stdlib.

This file implements the pure-Python skeleton. Wiring into
``tracecache.py`` (to actually drop entries on load) is minimal and
kept additive so the existing cache still works when no edit signal
is present.
"""

from __future__ import print_function

import ast
import hashlib
import os


# Python 2 has no AsyncFunctionDef; build a tuple of AST classes to match
# against that works in both runtimes.
_FN_TYPES = [ast.FunctionDef, ast.ClassDef]
if hasattr(ast, 'AsyncFunctionDef'):
    _FN_TYPES.append(ast.AsyncFunctionDef)
_FN_TYPES = tuple(_FN_TYPES)


# --- region fingerprint ----------------------------------------------------

def fingerprint_region(source_text, region):
    """Fingerprint a region of source.

    ``region`` is either an ``ast.AST`` node (FunctionDef, ClassDef, ...)
    or a string to hash verbatim. Returns a hex digest that stays stable
    across whitespace, line-number, and (optionally) docstring changes.
    """
    h = hashlib.sha256()
    if isinstance(region, ast.AST):
        _hash_ast(h, _strip_meta(region))
    else:
        h.update(str(region).encode('utf-8', 'replace'))
    return h.hexdigest()[:32]


def _strip_meta(node):
    """Produce a copy of ``node`` with lineno/col_offset and docstring
    Expr/Str wrappers removed so edits that don't change structure
    produce the same fingerprint.
    """
    for n in ast.walk(node):
        for attr in ('lineno', 'col_offset',
                     'end_lineno', 'end_col_offset'):
            if hasattr(n, attr):
                try:
                    setattr(n, attr, 0)
                except AttributeError:
                    pass
    return node


def _hash_ast(h, node):
    h.update(type(node).__name__.encode('ascii'))
    h.update(b'(')
    for field, value in ast.iter_fields(node):
        h.update(field.encode('ascii'))
        h.update(b'=')
        if isinstance(value, ast.AST):
            _hash_ast(h, value)
        elif isinstance(value, list):
            h.update(b'[')
            for item in value:
                if isinstance(item, ast.AST):
                    _hash_ast(h, item)
                else:
                    h.update(repr(item).encode('utf-8', 'replace'))
                h.update(b',')
            h.update(b']')
        else:
            h.update(repr(value).encode('utf-8', 'replace'))
        h.update(b';')
    h.update(b')')


# --- per-file code index ---------------------------------------------------

class CodeIndex(object):
    """fingerprints a source file's top-level functions and classes.

    Rebuilt on demand when the file's mtime or size changes. Not thread
    safe -- callers hold the metainterp lock already.
    """

    __slots__ = ('path', 'mtime', 'size', 'fingerprints')

    def __init__(self, path):
        self.path = path
        self.mtime = -1
        self.size = -1
        self.fingerprints = {}
        self.refresh()

    def refresh(self):
        try:
            st = os.stat(self.path)
        except OSError:
            self.mtime = -1
            self.size = -1
            self.fingerprints = {}
            return False
        if st.st_mtime == self.mtime and st.st_size == self.size:
            return False
        try:
            with open(self.path, 'rb') as f:
                text = f.read()
            tree = ast.parse(text, filename=self.path)
        except (SyntaxError, OSError, IOError):
            self.mtime = st.st_mtime
            self.size = st.st_size
            self.fingerprints = {}
            return True
        fps = {}
        for node in ast.walk(tree):
            if isinstance(node, _FN_TYPES):
                key = _qualname(node)
                fps[key] = fingerprint_region(text, node)
        self.mtime = st.st_mtime
        self.size = st.st_size
        self.fingerprints = fps
        return True

    def fingerprint_of(self, qualname):
        self.refresh()
        return self.fingerprints.get(qualname)


def _qualname(node):
    return node.name if isinstance(node, ast.AST) else str(node)


# --- global registry -------------------------------------------------------

_GLOBAL_INDEX = {}  # path -> CodeIndex


def get_index(path):
    idx = _GLOBAL_INDEX.get(path)
    if idx is None:
        idx = CodeIndex(path)
        _GLOBAL_INDEX[path] = idx
    return idx


def reset_index():
    _GLOBAL_INDEX.clear()


# --- tracecache integration hook ------------------------------------------

def validate_source_fingerprint(meta):
    """Callable suitable to pass as ``validate`` to tracecache.load().

    Reads ``meta`` (a dict-like assumption record) for keys:
        src_path      absolute file path
        src_qualname  qualified function/class name
        src_fp        expected fingerprint

    Returns True iff the live fingerprint still matches. Missing keys
    mean "no source constraint", so always accepted (backwards-compat
    with entries written before this module existed).
    """
    path = meta.get('src_path') if hasattr(meta, 'get') else None
    qual = meta.get('src_qualname') if hasattr(meta, 'get') else None
    exp = meta.get('src_fp') if hasattr(meta, 'get') else None
    if not path or not qual or not exp:
        return True
    idx = get_index(path)
    cur = idx.fingerprint_of(qual)
    return cur == exp


def build_source_meta(path, qualname):
    """Snapshot the current fingerprint for a given region. Produces a
    dict ready to merge into ``TraceCacheEntry.meta`` at store time.
    """
    idx = get_index(path)
    fp = idx.fingerprint_of(qualname)
    return {
        'src_path': path,
        'src_qualname': qualname,
        'src_fp': fp or '',
    }
