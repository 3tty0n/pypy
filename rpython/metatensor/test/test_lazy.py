"""The deferred-execution baseline must compute what the eager path computes
and build the kernels the fusion pass builds."""

import os
os.environ.setdefault('RTENSOR_CPU', '1')

from rpython.jit.metainterp.optimizeopt import metatensor as metatensor_opt
from rpython.jit.metainterp.test.support import LLJitMixin
from rpython.rlib.jit import JitDriver
from rpython.metatensor import core, kernels, lazy, ops, runtime
from rpython.metatensor.ops import (tensor_add, tensor_item, tensor_mul,
                                    tensor_relu, tensor_sum)
from rpython.metatensor.core import from_list
from rpython.metatensor.device import host

kernels.init_device()


class Lazily(object):
    def __enter__(self):
        self.old = core.lazy_knob.on
        core.lazy_knob.on = True
        lazy.live.refs = []
        return self

    def __exit__(self, *args):
        core.lazy_knob.on = self.old
        lazy.live.refs = []


def values(t):
    # what the app level does to read a tensor: force, then download
    h = host(ops.tensor_force(t))
    return [h[i] for i in range(t.size)]


def chain(x, w, b):
    h = ops.add(ops.mul(x, w), b)
    return ops.relu(h)


def test_chain_matches_eager():
    x = from_list([1.0, -2.0, 3.0, -4.0])
    w = from_list([0.5, 0.5, 0.5, 0.5])
    b = from_list([1.0, 1.0, 1.0, 1.0])
    want = values(chain(x, w, b))
    with Lazily():
        got = chain(x, w, b)
        assert got.lazy
        assert got.size == 4
        assert values(got) == want
        assert not got.lazy


def test_one_kernel_for_the_whole_chain():
    x = from_list([1.0, -2.0, 3.0, -4.0])
    w = from_list([0.5, 0.5, 0.5, 0.5])
    b = from_list([1.0, 1.0, 1.0, 1.0])
    with Lazily():
        before = kernels.counter.n
        values(chain(x, w, b))
        # one kernel name handed out, not one per operation
        assert kernels.counter.n - before <= 1


def test_kernel_key_is_the_one_the_pass_builds():
    x = from_list([1.0, 2.0, 3.0, 4.0])
    w = from_list([0.5, 0.5, 0.5, 0.5])
    b = from_list([1.0, 1.0, 1.0, 1.0])
    seen = []
    old = kernels.compile_or_reuse

    def spy(kernel):
        seen.append(kernels.kernel_key(kernel))
        return old(kernel)
    kernels.compile_or_reuse = spy
    lazy.kernels.compile_or_reuse = spy
    try:
        with Lazily():
            values(chain(x, w, b))
    finally:
        kernels.compile_or_reuse = old
        lazy.kernels.compile_or_reuse = old
    # leaves x, w, b are 0, 1, 2; mul -> 3, add -> 4, relu -> 5
    assert seen == ['3,4,1:0:1:0,0:3:2:0,2:4:-1:0,d0']


def test_reduction_shape_without_forcing():
    x = core.zeros([2, 4])
    for i in range(8):
        x.host[i] = float(i)
    with Lazily():
        s = ops.sum(ops.mul(x, x), 1)
        assert s.lazy
        assert s.size == 2
        assert values(s) == [14.0, 126.0]


def test_scalar_is_folded_into_the_kernel_body():
    from rpython.metatensor.devops import scalar_of
    x = from_list([1.0, 2.0, 3.0, 4.0])
    two = scalar_of(2.0, core.F64)
    with Lazily():
        t = ops.mul(x, two)
        lz = lazy._lazy_of(t)
        assert lz.leaves == [x]
        assert lz.consts == [two]
        assert values(t) == [2.0, 4.0, 6.0, 8.0]


def test_leaf_cap_forces_instead_of_widening():
    xs = [from_list([float(i + 1)] * 4) for i in range(8)]
    with Lazily():
        t = xs[0]
        for i in range(1, 8):
            t = ops.add(t, xs[i])
        # eight distinct leaves cannot ride in one launch
        assert values(t) == [36.0] * 4


def test_in_place_write_does_not_change_a_pending_read():
    x = from_list([1.0, 2.0, 3.0, 4.0])
    one = from_list([1.0, 1.0, 1.0, 1.0])
    with Lazily():
        pending = ops.mul(x, x)
        ops.assign(x, ops.add(x, one))
        assert values(x) == [2.0, 3.0, 4.0, 5.0]
        assert values(pending) == [1.0, 4.0, 9.0, 16.0]


def test_interior_node_read_later_is_recomputed():
    """The pass would make this an extra output of the kernel it already
    emitted.  A runtime library learns too late, so it recomputes from the
    leaves; the value has to be the same and the read has to cost a launch."""
    x = from_list([1.0, -2.0, 3.0, -4.0])
    w = from_list([0.5, 0.5, 0.5, 0.5])
    b = from_list([1.0, 1.0, 1.0, 1.0])
    with Lazily():
        inner = ops.add(ops.mul(x, w), b)
        outer = ops.relu(inner)
        assert values(outer) == [1.5, 0.0, 2.5, 0.0]
        assert inner.lazy
        assert values(inner) == [1.5, 0.0, 2.5, -1.0]


def test_dropped_intermediate_does_not_stay_live():
    x = from_list([1.0, 2.0, 3.0, 4.0])
    w = from_list([1.0, 1.0, 1.0, 1.0])
    with Lazily():
        for _ in range(50):
            values(ops.relu(ops.add(ops.mul(x, w), w)))
        import gc
        gc.collect()
        lazy.barrier()
        assert len(lazy.live.refs) == 0


class TestSameKernelsAsTheFusionPass(LLJitMixin):
    """The point of the baseline: same emitter, same kernels, different place
    to keep the DAG.  If the keys ever diverge the two arms are not comparable."""

    def _keys(self, run):
        seen = []
        old = kernels.compile_or_reuse

        def spy(kernel):
            key = kernels.kernel_key(kernel)
            if key not in seen:
                seen.append(key)
            return old(kernel)
        kernels.compile_or_reuse = spy
        lazy.kernels.compile_or_reuse = spy
        metatensor_opt.kernels.compile_or_reuse = spy
        try:
            run()
        finally:
            kernels.compile_or_reuse = old
            lazy.kernels.compile_or_reuse = old
            metatensor_opt.kernels.compile_or_reuse = old
        return seen

    def test_keys_match(self):
        driver = JitDriver(greens=[], reds=['n', 'w', 'b', 'acc'])

        def body(w, b):
            return ops.relu(ops.add(ops.mul(w, b), b))

        def f(n):
            w = from_list([1.0, -2.0, 3.0, -4.0])
            b = from_list([0.5, 0.5, 0.5, 0.5])
            acc = 0.0
            while n > 0:
                driver.jit_merge_point(n=n, acc=acc, w=w, b=b)
                acc += ops.item(ops.sum(body(w, b), -1))
                n -= 1
            return acc

        fused = self._keys(lambda: self.meta_interp(f, [10]))

        def lazily():
            w = from_list([1.0, -2.0, 3.0, -4.0])
            b = from_list([0.5, 0.5, 0.5, 0.5])
            with Lazily():
                h = body(w, b)
                ops.item(ops.sum(h, -1))
        deferred = self._keys(lazily)
        assert fused
        assert deferred == fused


def test_mark_step_forces_the_roots_but_not_the_interior():
    """A deferred library needs a per-iteration materialization point or the
    DAG grows across iterations.  Only the roots have to be forced: an
    interior node is computed inside its root's kernel."""
    x = from_list([1.0, 2.0, 3.0, 4.0])
    w = from_list([0.5, 0.5, 0.5, 0.5])
    with Lazily():
        inner = ops.mul(x, w)
        root = ops.relu(inner)
        lazy.mark_step()
        assert not root.lazy
        assert inner.lazy
        assert values(inner) == [0.5, 1.0, 1.5, 2.0]


def test_a_loop_without_a_mark_step_keeps_growing():
    x = from_list([1.0, 2.0, 3.0, 4.0])
    w = from_list([0.5, 0.5, 0.5, 0.5])
    with Lazily():
        h = x
        for _ in range(20):
            h = ops.relu(ops.mul(h, w))
        assert h.lazy
        depth = 0
        lz = lazy._lazy_of(h)
        while lz is not None:
            depth += 1
            lz = lazy._lazy_of(lz.a)
        assert depth == 40
    with Lazily():
        h = x
        for _ in range(20):
            h = ops.relu(ops.mul(h, w))
            lazy.mark_step()
        assert not h.lazy
