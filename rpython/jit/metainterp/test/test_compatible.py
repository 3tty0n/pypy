from rpython.jit.metainterp.test.support import LLJitMixin
from rpython.rlib import jit
from rpython.rtyper.lltypesystem import lltype, rffi


class TestCompatible(LLJitMixin):
    def test_simple(self):
        S = lltype.GcStruct('S', ('x', lltype.Signed))
        p1 = lltype.malloc(S)
        p1.x = 5

        p2 = lltype.malloc(S)
        p2.x = 5

        p3 = lltype.malloc(S)
        p3.x = 6
        driver = jit.JitDriver(greens = [], reds = ['n', 'x'])

        class A(object):
            pass

        c = A()
        c.count = 0
        @jit.elidable_compatible()
        def g(s, ignored):
            c.count += 1
            return s.x

        def f(n, x):
            while n > 0:
                driver.can_enter_jit(n=n, x=x)
                driver.jit_merge_point(n=n, x=x)
                n -= g(x, "abc")

        def main():
            g(p1, "def") # make annotator not make argument constant
            f(100, p1)
            f(100, p2)
            f(100, p3)
            return c.count

        x = self.meta_interp(main, [])

        assert x < 25
        # trace, two bridges, a finish bridge
        self.check_trace_count(4)

    def test_exception(self):
        S = lltype.GcStruct('S', ('x', lltype.Signed))
        p1 = lltype.malloc(S)
        p1.x = 5

        p2 = lltype.malloc(S)
        p2.x = 5

        p3 = lltype.malloc(S)
        p3.x = 6
        driver = jit.JitDriver(greens = [], reds = ['n', 'x'])
        @jit.elidable_compatible()
        def g(s):
            if s.x == 6:
                raise Exception
            return s.x

        def f(n, x):
            while n > 0:
                driver.can_enter_jit(n=n, x=x)
                driver.jit_merge_point(n=n, x=x)
                try:
                    n -= g(x)
                except:
                    n -= 1

        def main():
            f(100, p1)
            f(100, p2)
            f(100, p3)

        self.meta_interp(main, [])
        # XXX check number of bridges


    def test_quasi_immutable(self):
        from rpython.rlib.objectmodel import we_are_translated
        class C(object):
            _immutable_fields_ = ['version?']

        class Version(object):
            def __init__(self, cls):
                self.cls = cls
        p1 = C()
        p1.version = Version(p1)
        p1.x = 1
        p2 = C()
        p2.version = Version(p2)
        p2.x = 1
        p3 = C()
        p3.version = Version(p3)
        p3.x = 3

        driver = jit.JitDriver(greens = [], reds = ['n', 'x'])

        class Counter(object):
            pass

        c = Counter()
        c.count = 0
        @jit.elidable_compatible()
        def g(cls, v):
            if we_are_translated():
                c.count += 1
            return cls.x

        def f(n, x):
            res = 0
            while n > 0:
                driver.can_enter_jit(n=n, x=x)
                driver.jit_merge_point(n=n, x=x)
                x = jit.hint(x, promote_compatible=True)
                res = g(x, x.version)
                n -= res
            return res

        def main(x):
            res = f(100, p1)
            assert res == 1
            res = f(100, p2)
            assert res == 1
            res = f(100, p3)
            assert res == 3
            # invalidate p1 or p2
            if x:
                p1.x = 2
                p1.version = Version(p1)
                res = f(100, p1)
                assert res == 2
                p1.x = 1
                p1.version = Version(p1)
            else:
                p2.x = 2
                p2.version = Version(p2)
                res = f(100, p2)
                assert res == 2
                p2.x = 1
                p2.version = Version(p2)
            return c.count
        main(True)
        main(False)

        x = self.meta_interp(main, [True])
        assert x < 30

        x = self.meta_interp(main, [False])
        assert x < 30
        # XXX check number of bridges


    def test_dont_record_repeated_guard_compatible(self):
        class A:
            pass
        class B(A):
            pass
        @jit.elidable_compatible()
        def extern(x):
            return isinstance(x, A)
        @jit.dont_look_inside
        def pick(n):
            if n:
                x = a
            else:
                x = b
            return x
        a = A()
        b = B()
        def fn(n):
            x = pick(n)
            return extern(x) + extern(x) + extern(x)

        res = self.interp_operations(fn, [1])
        assert res == 3
        self.check_operations_history(guard_compatible=1)


    def test_too_many_bridges(self):
        S = lltype.GcStruct('S', ('x', lltype.Signed))
        p1 = lltype.malloc(S)
        p1.x = 5

        p2 = lltype.malloc(S)
        p2.x = 5

        p3 = lltype.malloc(S)
        p3.x = 6
        driver = jit.JitDriver(greens = [], reds = ['n', 'x'])

        class A(object):
            pass

        c = A()
        c.count = 0
        @jit.elidable_compatible()
        def g(s, ignored):
            c.count += 1
            return s.x

        def f(n, x):
            while n > 0:
                driver.can_enter_jit(n=n, x=x)
                driver.jit_merge_point(n=n, x=x)
                n -= g(x, 7)

        def main():
            g(p1, 9) # make annotator not make argument constant
            f(100, p1)
            f(100, p3) # not compatible, so make a bridge
            f(100, p2) # compatible with loop again, too bad
            return c.count

        x = self.meta_interp(main, [])

        assert x < 30
        # trace, two bridges, a finish bridge
        self.check_trace_count(4)


    def test_order_of_chained_guards(self):
        class Obj(object):
            def __init__(self):
                self.m = Map()
        class Map(object):
            pass

        p1 = Obj()
        p1.m.x = 5
        p1.m.y = 5

        p2 = Obj()
        p2.m.x = 5
        p2.m.y = 5

        p3 = Obj()
        p3.m.x = 5
        p3.m.y = 6
        driver = jit.JitDriver(greens = [], reds = ['n', 'x'])

        class A(object):
            pass

        c = A()
        c.count = 0
        @jit.elidable_compatible()
        def check1(m, ignored):
            c.count += 1
            return m.x

        @jit.elidable_compatible()
        def check2(m, ignored):
            c.count += 1
            return m.y

        def f(n, x):
            while n > 0:
                driver.can_enter_jit(n=n, x=x)
                driver.jit_merge_point(n=n, x=x)
                n -= check1(x.m, 7) + check2(x.m, 7)

        def main():
            check1(p1.m, 9) # make annotator not make argument constant
            f(100, p1)
            f(100, p3) # not compatible, so make a bridge
            f(100, p2) # compatible with loop again, too bad
            return c.count

        x = self.meta_interp(main, [])

        # trace, two bridges, a finish bridge
        self.check_trace_count(4)
        assert x < 50
