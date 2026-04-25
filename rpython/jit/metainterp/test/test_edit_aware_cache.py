"""Unit tests for edit-aware cache invalidation (Idea 3A)."""

import os
import tempfile

from rpython.jit.metainterp import edit_aware_cache as eac


def _write(path, text):
    with open(path, 'w') as f:
        f.write(text)


def test_fingerprint_is_stable_across_whitespace(tmpdir):
    p = str(tmpdir.join('a.py'))
    _write(p, 'def f(x):\n    return x + 1\n')
    eac.reset_index()
    fp1 = eac.get_index(p).fingerprint_of('f')
    # Same semantics, different whitespace.
    _write(p, 'def f(x):\n\n    return  x + 1\n')
    # Force mtime bump so refresh() re-reads.
    os.utime(p, (os.stat(p).st_mtime + 1, os.stat(p).st_mtime + 1))
    fp2 = eac.get_index(p).fingerprint_of('f')
    assert fp1 == fp2


def test_fingerprint_changes_on_real_edit(tmpdir):
    p = str(tmpdir.join('a.py'))
    _write(p, 'def f(x):\n    return x + 1\n')
    eac.reset_index()
    fp1 = eac.get_index(p).fingerprint_of('f')
    _write(p, 'def f(x):\n    return x + 2\n')
    os.utime(p, (os.stat(p).st_mtime + 1, os.stat(p).st_mtime + 1))
    fp2 = eac.get_index(p).fingerprint_of('f')
    assert fp1 != fp2


def test_validate_accepts_when_source_unchanged(tmpdir):
    p = str(tmpdir.join('a.py'))
    _write(p, 'def f(x):\n    return x + 1\n')
    eac.reset_index()
    meta = eac.build_source_meta(p, 'f')
    assert eac.validate_source_fingerprint(meta) is True


def test_validate_rejects_after_edit(tmpdir):
    p = str(tmpdir.join('a.py'))
    _write(p, 'def f(x):\n    return x + 1\n')
    eac.reset_index()
    meta = eac.build_source_meta(p, 'f')
    # Edit the source; bump mtime.
    _write(p, 'def f(x):\n    return x * 2\n')
    os.utime(p, (os.stat(p).st_mtime + 1, os.stat(p).st_mtime + 1))
    assert eac.validate_source_fingerprint(meta) is False


def test_validate_ignores_entries_without_source_meta():
    # Backwards compat: entries stored before 3A existed have no
    # src_path/qualname. Must always validate True.
    assert eac.validate_source_fingerprint({}) is True
    assert eac.validate_source_fingerprint({'other': 'stuff'}) is True


def test_many_functions_only_edited_one_invalidates(tmpdir):
    p = str(tmpdir.join('a.py'))
    _write(p, 'def a():\n    return 1\n\ndef b():\n    return 2\n\n'
              'def c():\n    return 3\n')
    eac.reset_index()
    idx = eac.get_index(p)
    fp_a = idx.fingerprint_of('a')
    fp_b = idx.fingerprint_of('b')
    fp_c = idx.fingerprint_of('c')
    # Only edit b.
    _write(p, 'def a():\n    return 1\n\ndef b():\n    return 99\n\n'
              'def c():\n    return 3\n')
    os.utime(p, (os.stat(p).st_mtime + 1, os.stat(p).st_mtime + 1))
    idx2 = eac.get_index(p)
    assert idx2.fingerprint_of('a') == fp_a  # still valid
    assert idx2.fingerprint_of('c') == fp_c  # still valid
    assert idx2.fingerprint_of('b') != fp_b  # invalidated
