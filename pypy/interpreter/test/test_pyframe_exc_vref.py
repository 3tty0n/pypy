"""App-level behaviour that patch 0002 must not change.

leave() no longer forces a frame's vref on every exception exit, only when
the vref escaped.  These lock down the cases that make a vref escape: a
traceback outliving its frames, sys._getframe(), re-raise, generators, with.

Upstream home: append to pypy/interpreter/test/test_pyframe.py.
Run: pytest patches/tests/test_pyframe_exc_vref.py
(Untranslated this is exercise, not proof -- the JIT-level proof is
patches/tests/test_pypy_c_exception_vref.py and the two new tests in
rpython/jit/metainterp/test/test_virtualref.py.)
"""


class AppTestExceptionFrames:
    def test_traceback_outlives_its_frames(self):
        import sys

        def h():
            raise ValueError("boom")

        def g():
            h()

        def f():
            g()

        try:
            f()
        except ValueError:
            tb = sys.exc_info()[2]
        # walk the chain long after every frame returned
        names = []
        t = tb
        while t is not None:
            names.append(t.tb_frame.f_code.co_name)
            t = t.tb_next
        assert names == ['test_traceback_outlives_its_frames', 'f', 'g', 'h']

    def test_tb_frame_f_back_chain(self):
        import sys

        def h():
            raise ValueError

        def g():
            h()

        def f():
            g()

        try:
            f()
        except ValueError:
            tb = sys.exc_info()[2]
        while tb.tb_next is not None:
            tb = tb.tb_next
        frame = tb.tb_frame                      # h's frame
        back = []
        while frame is not None:
            back.append(frame.f_code.co_name)
            frame = frame.f_back
        assert back[:4] == ['h', 'g', 'f',
                            'test_tb_frame_f_back_chain']

    def test_getframe_f_back_during_exception(self):
        import sys

        def h():
            try:
                raise ValueError
            except ValueError:
                f = sys._getframe()
                return [f.f_code.co_name, f.f_back.f_code.co_name]

        def g():
            return h()

        assert g() == ['h', 'g']

    def test_reraise_keeps_traceback(self):
        import sys

        def h():
            raise ValueError

        def g():
            try:
                h()
            except ValueError:
                raise

        try:
            g()
        except ValueError:
            tb = sys.exc_info()[2]
        names = []
        while tb is not None:
            names.append(tb.tb_frame.f_code.co_name)
            tb = tb.tb_next
        assert names[-1] == 'h'

    def test_exception_in_generator(self):
        import sys

        def gen(n):
            for i in range(n):
                if i == 3:
                    raise ValueError(i)
                yield i

        got = []
        try:
            for x in gen(10):
                got.append(x)
        except ValueError:
            tb = sys.exc_info()[2]
        assert got == [0, 1, 2]
        while tb.tb_next is not None:
            tb = tb.tb_next
        assert tb.tb_frame.f_code.co_name == 'gen'

    def test_exception_through_with_block(self):
        import sys

        class CM(object):
            entered = exited = False

            def __enter__(self):
                self.entered = True
                return self

            def __exit__(self, *args):
                self.exited = True
                return False

        cm = CM()

        def f():
            with cm:
                raise ValueError

        try:
            f()
        except ValueError:
            tb = sys.exc_info()[2]
        assert cm.entered and cm.exited
        while tb.tb_next is not None:
            tb = tb.tb_next
        assert tb.tb_frame.f_code.co_name == 'f'

    def test_caught_and_dropped_exception_is_invisible(self):
        # the case the patch optimises: nothing ever looks at the frames
        def h(i):
            raise ValueError(i)

        def g(i):
            h(i)

        total = 0
        for i in range(100):
            try:
                g(i)
            except ValueError as e:
                total += int(str(e))
        assert total == sum(range(100))
