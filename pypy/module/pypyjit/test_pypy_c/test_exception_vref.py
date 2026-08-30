"""JIT-level regression test for patch 0002.

Needs a translated pypy-c with the JIT:
    pypy/goal/pypy-c pytest.py patches/tests/test_pypy_c_exception_vref.py

Upstream home: pypy/module/pypyjit/test_pypy_c/test_exception.py.
The first test fails on unpatched pypy (thousands of forcings), the rest
guard the cases where forcing is still required.
"""
from pypy.module.pypyjit.test_pypy_c.test_00_model import BaseTestPyPyC


class TestExceptionVRef(BaseTestPyPyC):
    def test_caught_exception_keeps_frames_virtual(self):
        # exception used as control flow, caught two frames up, traceback
        # never looked at: nothing needs the frames to exist
        def main(n):
            def h(i):
                raise ValueError(i)

            def g(i):
                h(i)

            total = 0
            i = 0
            while i < n:
                try:
                    g(i)
                except ValueError as e:
                    total += e.args[0]
                i += 1
            return total

        log = self.run(main, [1000])
        assert log.result == sum(range(1000))
        assert log.jit_summary.virtualizables_forced == 0
        loop, = log.loops_by_filename(self.filepath)
        ops = log.opnames(loop.allops())
        # one force_token per frame that leaves with an exception before
        # the patch (5 here), only the guard's own after it
        assert ops.count('force_token') <= 2

    def test_traceback_kept_still_works(self):
        # the escaping case: the traceback outlives the frames, so the
        # frames must be materialized and the f_back chain stay walkable
        def main(n):
            import sys

            def h(i):
                raise ValueError(i)

            def g(i):
                h(i)

            names = None
            i = 0
            while i < n:
                try:
                    g(i)
                except ValueError:
                    tb = sys.exc_info()[2]
                i += 1
            while tb.tb_next is not None:
                tb = tb.tb_next
            frame = tb.tb_frame
            names = []
            while frame is not None and len(names) < 3:
                names.append(frame.f_code.co_name)
                frame = frame.f_back
            return names

        log = self.run(main, [1000])
        assert log.result == ['h', 'g', 'main']

    def test_getframe_f_back_from_compiled_code(self):
        # frame.escaped path, untouched by the patch
        def main(n):
            import sys

            def h():
                f = sys._getframe()
                return f.f_back.f_code.co_name

            def g():
                return h()

            last = None
            i = 0
            while i < n:
                last = g()
                i += 1
            return repr(last)

        log = self.run(main, [1000])
        assert log.result == 'g'

    def test_normal_return_unchanged(self):
        # no exception: frames stay virtual as before
        def main(n):
            def g(i):
                return i + 1

            total = 0
            i = 0
            while i < n:
                total += g(i)
                i += 1
            return total

        log = self.run(main, [1000])
        assert log.result == sum(range(1, 1001))
        assert log.jit_summary.virtualizables_forced == 0
