"""Deferred execution at runtime: the lazy-library baseline.

MetaTensor keeps the tensor DAG in the tracing JIT's virtual objects, so the
DAG is built once, when a trace is optimized, and what is left in the loop
body is a single launch of an already compiled kernel.  This module builds the
same DAG out of runtime objects instead, the way a conventional deferred
tensor library does, and hands it to the same kernel emitter, the same
in-process and on-disk kernel caches, the same device allocator and the same
launcher.  The two arms therefore differ only in where the DAG lives and when
it is built.  On the same DAG they build the same kernel, which the cache keys
confirm; they do not always see the same DAG, because a trace ends at the loop
back edge and a deferred library ends at whatever the program materializes.

Turned on by METATENSOR_LAZY=1.  It is only meaningful with the JIT's 'tensor'
optimization pass disabled (--jit enable_opts=... without 'tensor'), because
with the pass on the primitives in ops.py are absorbed and never execute.

One asymmetry is not an implementation accident and is the point of the
comparison.  The fusion pass can turn an interior node into an extra output
of a kernel it has already emitted, because the trace has not run yet.  A
runtime library cannot: by the time it learns that an interior value is
wanted, the launch has happened.  Here that interior value is recomputed from
its leaves instead, which is sound because an in-place write materializes
every deferred tensor first, and costs one extra launch where the pass costs
nothing.  Deciding before the launch instead is not open to it either: what
the program still holds is only known to the collector, so it would have to
make an output of every interior node.
"""

import os

from rpython.rlib import rweakref
from rpython.rtyper.annlowlevel import (cast_base_ptr_to_instance,
                                        cast_instance_to_base_ptr)
from rpython.rtyper.lltypesystem import lltype
from rpython.rtyper.rclass import OBJECTPTR

from rpython.metatensor import core, device, kernels, runtime
from rpython.metatensor.core import (ARITY, AXIS_ALL, HOSTARRAY, NDTYPES,
                                     NULLTENSOR, SHAPEARRAY, TENSOR,
                                     TENSORARRAY, _shape1, cols, is_reduction)
from rpython.metatensor.devops import scalars

NULLOBJ = lltype.nullptr(OBJECTPTR.TO)


def enabled():
    return core.lazy_knob.on


class Stats(object):
    fallbacks = 0
    forces = 0
    nodes = 0
    barriers = 0
    pruned = 0
stats = Stats()

# How long the live set may get before it is swept for entries whose tensor
# the program has dropped.  A lazy library has to keep this set, because an
# in-place write has to materialize whatever still reads the destination.
# The limit doubles when a sweep frees little: the collector runs on its own
# schedule, so a set that is still mostly alive would otherwise be swept on
# every single operation.
LIVE_SWEEP = 256


def _prune():
    kept = []
    refs = live.refs
    for i in range(len(refs)):
        other = refs[i]()
        if other is not None and other.out.lazy:
            kept.append(refs[i])
    stats.pruned += len(refs) - len(kept)
    live.refs = kept
    live.limit = LIVE_SWEEP
    if 2 * len(kept) > LIVE_SWEEP:
        live.limit = 2 * len(kept)


class Lazy(object):
    """One deferred operation.  `out` is the tensor it will produce; that
    tensor points back here through its `lazy` field, so the two die together
    once the program drops the tensor and the weak live set forgets it.

    The class-level defaults are what the annotator sees in a program that
    never turns deferred execution on, where force() is reachable through the
    hook but no Lazy is ever built."""

    opcode = 0
    param = 0
    # True once this node is an operand of another deferred node.  A
    # materialization point only has to force the roots; the interior nodes
    # are computed as part of the root's kernel.
    interior = False
    a = NULLTENSOR
    b = NULLTENSOR
    out = NULLTENSOR
    leaves = None
    consts = None

    def __init__(self, opcode, a, b, param, leaves, consts):
        self.interior = False
        self.opcode = opcode
        self.a = a
        self.b = b
        self.param = param
        self.leaves = leaves
        self.consts = consts
        self.out = NULLTENSOR


class Live(object):
    def __init__(self):
        self.refs = []
        self.limit = LIVE_SWEEP
live = Live()


def _lazy_of(t):
    if not t.lazy:
        return None
    return cast_base_ptr_to_instance(Lazy, t.lazy)


def _is_const_scalar(t):
    """A 0-d tensor from the scalar memo, the runtime counterpart of the
    constant pointer the fusion pass folds into the kernel body."""
    if t.size != 1 or not t.host:
        return False
    dt = t.dtype
    if dt < 0 or dt >= NDTYPES:
        return False
    v = t.host[0]
    if not (v == v and v - v == 0.0):
        return False
    return scalars.tensors[dt].get(v, NULLTENSOR) == t


def _index_of(seq, t):
    for i in range(len(seq)):
        if seq[i] == t:
            return i
    return -1


def _merge(t, leaves, consts):
    lz = _lazy_of(t)
    if lz is not None:
        lz.interior = True
        for i in range(len(lz.leaves)):
            if _index_of(leaves, lz.leaves[i]) < 0:
                leaves.append(lz.leaves[i])
        for i in range(len(lz.consts)):
            if _index_of(consts, lz.consts[i]) < 0:
                consts.append(lz.consts[i])
    elif _is_const_scalar(t):
        if _index_of(consts, t) < 0:
            consts.append(t)
    elif _index_of(leaves, t) < 0:
        leaves.append(t)


def _operands(opcode, a, b, leaves, consts):
    _merge(a, leaves, consts)
    if ARITY[opcode] == 2:
        _merge(b, leaves, consts)


def _width(leaves, consts):
    n = len(leaves)
    if n == 0:
        n = len(consts)
    return n


def _result_shape(opcode, a, b, param):
    """What the launcher will produce for this node, computed the way
    launch_gpu computes it, so that a deferred tensor answers size, shape and
    dtype queries without being materialized."""
    if is_reduction(opcode):
        n = a.size
        c = cols(a)
        if param == 0:
            outlen = c
        elif param == 1:
            outlen = n // c if c > 0 else n
        else:
            outlen = 1
        if outlen <= 0:
            outlen = 1
        return outlen, _shape1(outlen), a.dtype
    big = a
    if ARITY[opcode] == 2 and b.size > a.size:
        big = b
    return big.size, big.shape, big.dtype


def lazy_op(opcode, a, b, param):
    leaves, consts = [], []
    _operands(opcode, a, b, leaves, consts)
    if _width(leaves, consts) > core.max_inputs():
        force(a)
        if ARITY[opcode] == 2:
            force(b)
        leaves, consts = [], []
        _operands(opcode, a, b, leaves, consts)
        if _width(leaves, consts) > core.max_inputs():
            # Forcing could not cut it, the operands are all constants.  Run
            # the node on its own, so its result becomes a single leaf, which
            # is what the pass does when it emits the residual call.
            return runtime.eval_op(opcode, a, b, param)
    n, shape, dtype = _result_shape(opcode, a, b, param)
    t = lltype.malloc(TENSOR)
    t.size = n
    t.shape = shape
    t.dptr = 0
    t.host = lltype.nullptr(HOSTARRAY)
    t.extra = lltype.nullptr(TENSORARRAY)
    t.dtype = dtype
    t.buf = NULLOBJ
    lz = Lazy(opcode, a, b, param, leaves, consts)
    lz.out = t
    t.lazy = cast_instance_to_base_ptr(lz)
    if len(live.refs) >= live.limit:
        _prune()
    live.refs.append(rweakref.ref(lz))
    stats.nodes += 1
    return t


def _emit(lz, leaves, consts, opcodes, lefts, rights, params, infos):
    idx = [-1, -1]
    nargs = ARITY[lz.opcode]
    for i in range(nargs):
        arg = lz.a if i == 0 else lz.b
        sub = _lazy_of(arg)
        if sub is not None:
            idx[i] = _emit(sub, leaves, consts, opcodes, lefts, rights,
                           params, infos)
        else:
            j = _index_of(leaves, arg)
            if j < 0:
                j = len(leaves) + _index_of(consts, arg)
            idx[i] = j
    opcodes.append(lz.opcode)
    lefts.append(idx[0])
    rights.append(idx[1])
    params.append(lz.param)
    infos.append(lz)
    return len(leaves) + len(consts) + len(opcodes) - 1


def _collect_leaves(lz, leaves, consts, split):
    nargs = ARITY[lz.opcode]
    for i in range(nargs):
        arg = lz.a if i == 0 else lz.b
        sub = _lazy_of(arg)
        if sub is not None:
            _collect_leaves(sub, leaves, consts, split)
        elif split and _is_const_scalar(arg):
            if _index_of(consts, arg) < 0:
                consts.append(arg)
        elif _index_of(leaves, arg) < 0:
            leaves.append(arg)


def _adopt(t, r):
    t.size = r.size
    t.shape = r.shape
    t.dptr = r.dptr
    t.host = r.host
    t.buf = r.buf
    t.extra = r.extra
    t.lazy = NULLOBJ


def _launch(kernel, leaves):
    """The same two steps runtime.launch takes for the fusion pass: the
    compiled kernel if there is one, the node array interpreted otherwise."""
    if kernel.fn != 0:
        nmax = 0
        for i in range(len(leaves)):
            if leaves[i].size > nmax:
                nmax = leaves[i].size
        if kernel.n == 0 or kernel.n == nmax:
            r = runtime.launch_gpu(kernel, leaves)
            if r:
                return r
    device.note_unfused(kernel.fn, runtime.refusal(kernel, leaves))
    values = []
    for i in range(len(leaves)):
        values.append(leaves[i])
    for j in range(len(kernel.consts)):
        values.append(runtime.cached_scalar(kernel.consts[j], kernel.dtype))
    nodes = kernel.nodes
    for i in range(len(nodes)):
        node = nodes[i]
        opcode = node.opcode
        assert opcode >= 0
        ia = node.a
        assert ia >= 0
        ib = node.b
        right = values[ib] if ib >= 0 else NULLTENSOR
        values.append(runtime.eval_op(opcode, values[ia], right, node.p))
    result = values[len(values) - 1]
    nout = len(kernel.outputs)
    if nout > 0:
        result.extra = lltype.malloc(TENSORARRAY, nout)
        for k in range(nout):
            idx = kernel.outputs[k]
            assert idx >= 0
            result.extra[k] = values[idx]
    return result


def _materialize(lz, t):
    stats.forces += 1
    leaves, consts = [], []
    opcodes, lefts, rights, params, infos = [], [], [], [], []
    _collect_leaves(lz, leaves, consts, True)
    if not leaves:
        leaves, consts = [], []
        _collect_leaves(lz, leaves, consts, False)
    if not leaves or len(leaves) > core.MAX_INPUTS_LIMIT:
        _adopt(t, _eval_tree(lz))
        return
    _emit(lz, leaves, consts, opcodes, lefts, rights, params, infos)
    big = leaves[0]
    for i in range(1, len(leaves)):
        if leaves[i].size > big.size:
            big = leaves[i]
    kernel = kernels.new_kernel(len(leaves), len(opcodes), big.dtype)
    kernel.consts = lltype.malloc(HOSTARRAY, len(consts))
    for i in range(len(consts)):
        kernel.consts[i] = consts[i].host[0]
    for i in range(len(opcodes)):
        kernels.set_node(kernel, i, opcodes[i], lefts[i], rights[i], params[i])
    # Mirror what the pass bakes into the kernel: it can only put a size in
    # when the trace promoted one, which is exactly when the size policy is
    # still static.  Once the policy has demoted, both arms compile a
    # size-independent kernel and the cache keys stay equal.
    kernel.n = big.size if core.policy.static else 0
    kernel.cols = cols(big) if core.policy.static_cols else 0
    kernels.compile_or_reuse(kernel)
    _adopt(t, _launch(kernel, leaves))


def _eval_tree(lz):
    """Fallback for a chain the launcher cannot take: run it node by node,
    which is what the single-op path does anyway."""
    a = force(lz.a)
    b = lz.b
    if ARITY[lz.opcode] == 2:
        b = force(b)
    return runtime.eval_op(lz.opcode, a, b, lz.param)


def force(t):
    """Materialize t.

    A failed force falls back to evaluating the chain node by node, the way
    the runtime already does when a launch is refused.  Note where this is
    called from: not host() and dev(), which run underneath the launcher's
    elidable-or-memoryerror oopspec.  The raise and write analyzers are
    syntactic, so a kernel compile reachable from there would break that
    oopspec's declared effect and make the library ops random-effecting,
    which would stop the pass fusing across them."""
    lz = _lazy_of(t)
    if lz is None:
        return t
    try:
        _materialize(lz, t)
    except Exception:
        stats.fallbacks += 1
        _adopt(t, _eval_tree(lz))
    return t


def mark_step():
    """The materialization point a deferred library needs and a trace gets for
    free.  Without one the DAG grows across loop iterations instead of being
    executed, which is why PyTorch/XLA makes the program call mark_step.  Only
    the roots are forced; an interior node is computed inside its root's
    kernel."""
    refs = live.refs
    for i in range(len(refs)):
        other = refs[i]()
        if other is not None and not other.interior and other.out.lazy:
            force(other.out)
    _prune()


def barrier():
    """An in-place write is about to happen: materialize every tensor that is
    still deferred, the runtime counterpart of OptTensor.force_live.  It is
    all of them and not just the readers of the destination because a write
    can reach a deferred operand through a reshaped view, which is a different
    tensor object over the same buffer."""
    stats.barriers += 1
    refs = live.refs
    live.refs = []
    live.limit = LIVE_SWEEP
    for i in range(len(refs)):
        other = refs[i]()
        if other is not None and other.out.lazy:
            force(other.out)


device.lazy_force = force


