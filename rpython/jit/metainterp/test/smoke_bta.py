"""Quick smoke test for BTA in untranslated mode."""
from __future__ import print_function
import sys
import os
import subprocess

# Work around close_fds overflow on macOS + PyPy2
_orig_popen = subprocess.Popen.__init__
def _patched_popen(self, *a, **kw):
    kw.pop('close_fds', None)
    return _orig_popen(self, *a, **kw)
subprocess.Popen.__init__ = _patched_popen

# Ensure repo root is on path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from rpython.jit.metainterp.pyjitpl import PyjitplCounters
PyjitplCounters.reset()

# Run a tiny hot loop that the JIT will trace
def test_loop():
    n = 1000
    acc = 0
    for i in range(n):
        acc = acc + 1
    return acc

result = test_loop()
print('result:', result)
PyjitplCounters.report()
print('bta_skipped:', PyjitplCounters._bta_skipped)
