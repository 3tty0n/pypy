"""JIT-level regression test for patch 0001.

Needs a translated pypy-c with the JIT:
    pypy/goal/pypy-c pytest.py patches/tests/test_pypy_c_quasiimmut_globals.py

Upstream home: pypy/module/pypyjit/test_pypy_c/test_globals.py.
Fails on unpatched pypy (hundreds of aborts), passes with 0001.
"""
from pypy.module.pypyjit.test_pypy_c.test_00_model import BaseTestPyPyC


class TestQuasiImmutGlobals(BaseTestPyPyC):
    def test_runtime_created_code_does_not_abort_traces(self):
        # every callee's code object runs its first frame from inside the
        # traced loop; writing PyCode.w_globals there forces the
        # quasi-immutable and throws the trace away
        def main(n):
            funcs = []
            src = "def f(a):\n    return a + 1\n"
            for i in range(n):
                g = {}
                exec src in g
                funcs.append(g['f'])
            total = 0
            i = 0
            while i < n:
                total += funcs[i](i)
                i += 1
            return total

        log = self.run(main, [1000])
        assert log.result == sum(range(1, 1001))
        assert log.jit_summary.abort.force_quasiimmut == 0

    def test_globals_still_reachable_from_compiled_code(self):
        # the frame keeps its own globals instead; reads must still work
        def main(n):
            src = ("def f(a):\n"
                   "    return a + BASE\n")
            funcs = []
            for i in range(n):
                g = {'BASE': i}
                exec src in g
                funcs.append(g['f'])
            total = 0
            i = 0
            while i < n:
                total += funcs[i](i)
                i += 1
            return total

        log = self.run(main, [1000])
        assert log.result == sum(2 * i for i in range(1000))
        assert log.jit_summary.abort.force_quasiimmut == 0

    def test_shared_code_object_unchanged(self):
        # the common case (one code object, one globals dict) still takes
        # the cached path: no extra work, no aborts
        def main(n):
            def f(a):
                return a + 1
            total = 0
            i = 0
            while i < n:
                total += f(i)
                i += 1
            return total

        log = self.run(main, [1000])
        assert log.result == sum(range(1, 1001))
        assert log.jit_summary.abort.force_quasiimmut == 0
        loop, = log.loops_by_filename(self.filepath)
        assert 'quasiimmut_field' not in log.opnames(loop.allops())
