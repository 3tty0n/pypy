""" Direct (untranslated, in-process) test of aarch64 stitch_bridge, used to
debug Phase 1 bridge reuse.  Builds a loop with two guards that share the same
failarg layout, compiles a bridge for the first guard, then stitches the second
guard to that same bridge and runs it.  This exercises the trampoline
(rebuild_faillocs + remap + branch + patch_trace) on real hardware without a
full translation. """
from rpython.jit.backend.detect_cpu import getcpuclass
from rpython.jit.metainterp.history import (BasicFailDescr, BasicFinalDescr,
                                            JitCellToken, TargetToken)
from rpython.jit.tool.oparser import parse
from rpython.rtyper.lltypesystem import lltype, llmemory, rffi

CPU = getcpuclass()


class FakeStats(object):
    pass


class _Version(object):
    def __init__(self, inputargs):
        self.inputargs = inputargs


def test_stitch_bridge_same_layout():
    cpu = CPU(rtyper=None, stats=FakeStats())
    cpu.setup_once()
    targettoken = TargetToken()
    fd1 = BasicFailDescr(1)
    fd2 = BasicFailDescr(2)
    fdfinal = BasicFinalDescr(3)
    loop = parse("""
    [i0]
    label(i0, descr=targettoken)
    i1 = int_add(i0, 1)
    i2 = int_le(i1, 100)
    guard_true(i2, descr=fd1) [i1]
    i3 = int_le(i1, 3)
    guard_true(i3, descr=fd2) [i1]
    jump(i1, descr=targettoken)
    """, namespace={'targettoken': targettoken, 'fd1': fd1, 'fd2': fd2})
    looptoken = JitCellToken()
    cpu.compile_loop(loop.inputargs, loop.operations, looptoken)

    bridge = parse("""
    [i1]
    i9 = int_add(i1, 1000)
    finish(i9, descr=fdfinal)
    """, namespace={'fdfinal': fdfinal})
    asminfo = cpu.compile_bridge(fd1, bridge.inputargs, bridge.operations,
                                 looptoken)
    assert asminfo is not None
    print("asminfo.asmaddr=0x%x rawstart=0x%x" % (asminfo.asmaddr,
                                                  asminfo.rawstart))

    version = _Version(bridge.inputargs)
    ok = cpu.stitch_bridge(fd2, (asminfo, fd1, version, looptoken))
    print("stitch_bridge returned:", ok)

    deadframe = cpu.execute_token(looptoken, 0)
    fail = cpu.get_latest_descr(deadframe)
    res = cpu.get_int_value(deadframe, 0)
    print("fail.identifier=%s res=%s" % (fail.identifier, res))
    # i0=0 -> i1 climbs 1,2,3,4; at i1=4 guard fd2 (i1<=3) fails.  If the stitch
    # works that failure runs the bridge -> 4+1000 == 1004, fail == fdfinal(3).
    assert ok is True
    assert fail.identifier == 3
    assert res == 1004


def test_stitch_bridge_four_args():
    # raytrace's failing stitch had n=4 integer failargs; force a non-trivial
    # remap by keeping several values live across both guards.
    cpu = CPU(rtyper=None, stats=FakeStats())
    cpu.setup_once()
    tt = TargetToken()
    fd1 = BasicFailDescr(1)
    fd2 = BasicFailDescr(2)
    fdfinal = BasicFinalDescr(3)
    loop = parse("""
    [i0]
    label(i0, descr=tt)
    i1 = int_add(i0, 1)
    i2 = int_add(i1, 1)
    i3 = int_add(i2, 1)
    i4 = int_le(i1, 100)
    guard_true(i4, descr=fd1) [i1, i2, i3, i0]
    i5 = int_le(i1, 3)
    guard_true(i5, descr=fd2) [i1, i2, i3, i0]
    jump(i1, descr=tt)
    """, namespace={'tt': tt, 'fd1': fd1, 'fd2': fd2})
    looptoken = JitCellToken()
    cpu.compile_loop(loop.inputargs, loop.operations, looptoken)

    bridge = parse("""
    [i1, i2, i3, i0]
    i9 = int_add(i1, i2)
    i10 = int_add(i9, i3)
    i11 = int_add(i10, i0)
    i12 = int_add(i11, 1000)
    finish(i12, descr=fdfinal)
    """, namespace={'fdfinal': fdfinal})
    asminfo = cpu.compile_bridge(fd1, bridge.inputargs, bridge.operations,
                                 looptoken)
    assert asminfo is not None

    version = _Version(bridge.inputargs)
    ok = cpu.stitch_bridge(fd2, (asminfo, fd1, version, looptoken))
    print("4-arg stitch returned:", ok)

    deadframe = cpu.execute_token(looptoken, 0)
    fail = cpu.get_latest_descr(deadframe)
    res = cpu.get_int_value(deadframe, 0)
    print("4-arg fail.identifier=%s res=%s" % (fail.identifier, res))
    # failing iter has i1=4,i2=5,i3=6,i0=3 -> 4+5+6+3+1000 = 1018
    assert ok is True
    assert fail.identifier == 3
    assert res == 1018


def test_stitch_bridge_inner_guard_fails():
    # the stitched-to bridge has its own guard that FAILS for the second guard's
    # runtime values -> the stitched bridge must deopt correctly.
    cpu = CPU(rtyper=None, stats=FakeStats())
    cpu.setup_once()
    tt = TargetToken()
    fd1 = BasicFailDescr(1)
    fd2 = BasicFailDescr(2)
    fdg = BasicFailDescr(7)
    fdfinal = BasicFinalDescr(3)
    loop = parse("""
    [i0]
    label(i0, descr=tt)
    i1 = int_add(i0, 1)
    i4 = int_le(i1, 100)
    guard_true(i4, descr=fd1) [i1, i0]
    i5 = int_le(i1, 3)
    guard_true(i5, descr=fd2) [i1, i0]
    jump(i1, descr=tt)
    """, namespace={'tt': tt, 'fd1': fd1, 'fd2': fd2})
    looptoken = JitCellToken()
    cpu.compile_loop(loop.inputargs, loop.operations, looptoken)

    bridge = parse("""
    [i1, i0]
    i9 = int_add(i1, 1)
    guard_value(i1, 99, descr=fdg) [i1, i0]
    finish(i9, descr=fdfinal)
    """, namespace={'fdg': fdg, 'fdfinal': fdfinal})
    asminfo = cpu.compile_bridge(fd1, bridge.inputargs, bridge.operations,
                                 looptoken)
    assert asminfo is not None

    version = _Version(bridge.inputargs)
    ok = cpu.stitch_bridge(fd2, (asminfo, fd1, version, looptoken))

    deadframe = cpu.execute_token(looptoken, 0)
    fail = cpu.get_latest_descr(deadframe)
    res = cpu.get_int_value(deadframe, 0)
    print("inner-guard fail.identifier=%s res=%s" % (fail.identifier, res))
    # at i1=4 fd2 fails -> stitch -> bridge -> guard_value(i1,99) fails (4!=99)
    # -> deopt via fdg with [i1=4, i0=3]
    assert ok is True
    assert fail.identifier == 7
    assert res == 4


def test_stitch_gcmap_inspection():
    """ Compare the gcmap (and the ref value seen) at the bridge's malloc
    slowpath when the bridge is entered (a) normally from its own guard fd1 vs
    (b) via a stitch from sibling guard fd2.  If they differ, the stitch leaves
    a wrong GC view -> that is the ref segfault root cause, isolated WITHOUT a
    full translation. """
    from rpython.jit.backend.llsupport.test.test_gc_integration import (
        GCDescrFastpathMalloc, getmap, unpack_gcmap)
    from rpython.jit.backend.aarch64 import assembler as aarch64_asm

    captured = []

    def check(frame):
        bits = unpack_gcmap(frame)
        vals = [rffi.cast(lltype.Signed, frame.jf_frame[i]) for i in bits]
        captured.append((getmap(frame), bits, vals))

    cpu = CPU(None, None)
    cpu.gc_ll_descr = GCDescrFastpathMalloc(check)
    cpu.setup_once()

    S = lltype.GcStruct('S', ('x', lltype.Signed))
    s = lltype.malloc(S, immortal=True)
    s.x = 12345
    p0ref = lltype.cast_opaque_ptr(llmemory.GCREF, s)
    p0addr = rffi.cast(lltype.Signed, p0ref)

    tt = TargetToken()
    fd1 = BasicFailDescr(1)
    fd2 = BasicFailDescr(2)
    fdfinal = BasicFinalDescr(3)
    loop = parse("""
    [i0, p0]
    label(i0, p0, descr=tt)
    i2 = int_add(i0, 1)
    i4 = int_le(i2, 100)
    guard_true(i4, descr=fd1) [p0, i2]
    i5 = int_le(i2, 3)
    guard_true(i5, descr=fd2) [p0, i2]
    jump(i2, p0, descr=tt)
    """, namespace={'tt': tt, 'fd1': fd1, 'fd2': fd2})
    looptoken = JitCellToken()
    cpu.compile_loop(loop.inputargs, loop.operations, looptoken)

    bridge = parse("""
    [p0, i2]
    p9 = call_malloc_nursery(72)
    finish(p0, descr=fdfinal)
    """, namespace={'fdfinal': fdfinal})
    asminfo = cpu.compile_bridge(fd1, bridge.inputargs, bridge.operations,
                                 looptoken)
    assert asminfo is not None

    nurs = rffi.cast(lltype.Signed, cpu.gc_ll_descr.nursery)

    # (a) normal: i0=200 -> i2=201 -> fd1 fails -> its bridge -> malloc slowpath
    cpu.gc_ll_descr.addrs[0] = nurs
    cpu.execute_token(looptoken, 200, p0ref)

    # (b) stitch fd2 -> the same bridge (allow the ref stitch just here), then
    # make fd2 fail (i0=5 -> i2=6: fd1 passes, fd2 fails)
    version = _Version(bridge.inputargs)
    aarch64_asm.PHASE1_DIAG_ALLOW_REF = True
    try:
        ok = cpu.stitch_bridge(fd2, (asminfo, fd1, version, looptoken))
    finally:
        aarch64_asm.PHASE1_DIAG_ALLOW_REF = False
    cpu.gc_ll_descr.addrs[0] = nurs
    cpu.execute_token(looptoken, 5, p0ref)

    print("stitch ok:", ok)
    print("p0addr = 0x%x" % p0addr)
    for label, cap in zip(["NORMAL(fd1)", "STITCH(fd2)"], captured):
        gcmap, bits, vals = cap
        print("%s gcmap=%s bits=%s vals=%s" %
              (label, gcmap, bits, ["0x%x" % v for v in vals]))
    try:
        assert len(captured) == 2, captured
        # Finding: the stitched bridge sees the SAME gcmap and the correct ref
        # value as a normal entry -- so ref-stitch is NOT a gcmap problem.
        assert captured[0][0] == captured[1][0], "gcmap differs: %s" % (captured,)
        assert p0addr in captured[1][2], "stitched ref value wrong: %s" % (captured,)
    finally:
        lltype.free(cpu.gc_ll_descr.nursery, flavor='raw')
        lltype.free(cpu.gc_ll_descr.addrs, flavor='raw')


def test_stitch_gcmap_multiref():
    """ Like test_stitch_gcmap_inspection but with several refs live across the
    guards (forcing some onto the stack), to see whether a more complex remap
    leaves a wrong gcmap / wrong ref value at the stitched bridge's GC point. """
    from rpython.jit.backend.llsupport.test.test_gc_integration import (
        GCDescrFastpathMalloc, getmap, unpack_gcmap)
    from rpython.jit.backend.aarch64 import assembler as aarch64_asm

    captured = []

    def check(frame):
        bits = unpack_gcmap(frame)
        vals = [rffi.cast(lltype.Signed, frame.jf_frame[i]) for i in bits]
        captured.append((getmap(frame), sorted(vals)))

    cpu = CPU(None, None)
    cpu.gc_ll_descr = GCDescrFastpathMalloc(check)
    cpu.setup_once()

    S = lltype.GcStruct('S', ('x', lltype.Signed))
    refs = []
    addrs = []
    for k in range(6):
        s = lltype.malloc(S, immortal=True); s.x = k
        r = lltype.cast_opaque_ptr(llmemory.GCREF, s)
        refs.append(r); addrs.append(rffi.cast(lltype.Signed, r))

    tt = TargetToken()
    fd1 = BasicFailDescr(1)
    fd2 = BasicFailDescr(2)
    fdfinal = BasicFinalDescr(3)
    loop = parse("""
    [i0, p0, p1, p2, p3, p4, p5]
    label(i0, p0, p1, p2, p3, p4, p5, descr=tt)
    i2 = int_add(i0, 1)
    i4 = int_le(i2, 100)
    guard_true(i4, descr=fd1) [p0, p1, p2, p3, p4, p5, i2]
    i5 = int_le(i2, 3)
    guard_true(i5, descr=fd2) [p0, p1, p2, p3, p4, p5, i2]
    jump(i2, p0, p1, p2, p3, p4, p5, descr=tt)
    """, namespace={'tt': tt, 'fd1': fd1, 'fd2': fd2})
    looptoken = JitCellToken()
    cpu.compile_loop(loop.inputargs, loop.operations, looptoken)

    bridge = parse("""
    [p0, p1, p2, p3, p4, p5, i2]
    p9 = call_malloc_nursery(72)
    finish(p0, descr=fdfinal)
    """, namespace={'fdfinal': fdfinal})
    asminfo = cpu.compile_bridge(fd1, bridge.inputargs, bridge.operations,
                                 looptoken)
    nurs = rffi.cast(lltype.Signed, cpu.gc_ll_descr.nursery)

    cpu.gc_ll_descr.addrs[0] = nurs
    cpu.execute_token(looptoken, 200, *refs)

    version = _Version(bridge.inputargs)
    aarch64_asm.PHASE1_DIAG_ALLOW_REF = True
    try:
        ok = cpu.stitch_bridge(fd2, (asminfo, fd1, version, looptoken))
    finally:
        aarch64_asm.PHASE1_DIAG_ALLOW_REF = False
    cpu.gc_ll_descr.addrs[0] = nurs
    cpu.execute_token(looptoken, 5, *refs)

    print("multiref stitch ok:", ok)
    for label, cap in zip(["NORMAL", "STITCH"], captured):
        print("%s gcmap=%s nrefs=%d vals=%s" %
              (label, cap[0], len(cap[1]), ["0x%x" % v for v in cap[1]]))
    try:
        assert len(captured) == 2, captured
        # only p0 stays live across the malloc (finish(p0)); the key check is
        # that the stitched entry presents the SAME gcmap and the SAME live ref
        # value(s) as a normal entry.
        assert captured[0][0] == captured[1][0], "GCMAP DIFFERS:\n  %s\n  %s" % (captured[0], captured[1])
        assert captured[0][1] == captured[1][1], "LIVE REFS DIFFER:\n  %s\n  %s" % (captured[0][1], captured[1][1])
        assert addrs[0] in captured[1][1], "p0 wrong in stitch: 0x%x not in %s" % (addrs[0], ["0x%x"%v for v in captured[1][1]])
    finally:
        lltype.free(cpu.gc_ll_descr.nursery, flavor='raw')
        lltype.free(cpu.gc_ll_descr.addrs, flavor='raw')
