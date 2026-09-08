"""Tests for patch 0001 (pypy/interpreter/pycode.py).

Upstream home: append these to pypy/interpreter/test/test_code.py.
Run: pytest patches/tests/test_pycode_globals_jit.py
"""
from rpython.rlib import jit as rjit


def _compile(space, src):
    return space.createcompiler().compile(src, '<test>', 'exec', 0)


def _frame(space, code, w_globals):
    return space.FrameClass(space, code, w_globals, None)


class TestFrameStoresGlobal:
    def test_caches_first_globals_when_not_jitted(self):
        space = self.space
        code = _compile(space, "x = 1")
        w_g = space.newdict(module=True)
        assert code.frame_stores_global(w_g) is False
        assert code.w_globals is w_g
        # a frame running in that same globals reads them off the code
        frame = _frame(space, code, w_g)
        assert frame.get_w_globals() is w_g

    def test_second_globals_still_stored_on_the_frame(self):
        space = self.space
        code = _compile(space, "x = 1")
        w_g1 = space.newdict(module=True)
        w_g2 = space.newdict(module=True)
        code.frame_stores_global(w_g1)
        assert code.frame_stores_global(w_g2) is True
        assert code.w_globals is w_g1
        frame = _frame(space, code, w_g2)
        assert frame.get_w_globals() is w_g2

    def test_jitted_does_not_write_the_quasi_immutable(self, monkeypatch):
        # the point of the patch: from compiled code the field is left
        # alone (writing it forces the quasi-immut and aborts the trace)
        space = self.space
        code = _compile(space, "x = 1")
        w_g = space.newdict(module=True)
        monkeypatch.setattr(rjit, 'we_are_jitted', lambda: True)
        assert code.frame_stores_global(w_g) is True
        assert code.w_globals is None

    def test_jitted_frame_still_finds_its_globals(self, monkeypatch):
        space = self.space
        code = _compile(space, "x = 1")
        w_g = space.newdict(module=True)
        monkeypatch.setattr(rjit, 'we_are_jitted', lambda: True)
        frame = _frame(space, code, w_g)
        assert frame.get_w_globals() is w_g
        # and the code object is still uncommitted afterwards, so a later
        # interpreted frame may claim it
        assert code.w_globals is None
        monkeypatch.undo()
        assert code.frame_stores_global(w_g) is False
        assert code.w_globals is w_g

    def test_eval_created_code_runs_correctly(self):
        # end-to-end: code objects made at run time see the right globals
        space = self.space
        w_res = space.appexec([], """():
            def make(n):
                g = {'n': n}
                c = compile('r = n * 2', '<eval>', 'exec')
                exec c in g
                return g['r']
            return [make(i) for i in range(3)]
        """)
        assert space.unwrap(w_res) == [0, 2, 4]
