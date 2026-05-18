"""Byte-for-byte verification of the NEON .2D encoders against the system
assembler (the encoding oracle).  Unlike test_instr_builder, this MUST run
on darwin -- raw SIMD machine code is silent memory corruption if a bit is
wrong, so the oracle is mandatory on the target platform.
"""
from hypothesis import given, settings, strategies as st
from rpython.jit.backend.aarch64 import codebuilder
from rpython.jit.backend.aarch64.test.gen import assemble


class CodeBuilder(codebuilder.InstrBuilder):
    def __init__(self, arch_version=7):
        self.arch_version = arch_version
        self.buffer = []

    def writechar(self, char):
        self.buffer.append(char)

    def currpos(self):
        return 0

    def hexdump(self):
        return ''.join(self.buffer)


# Values that exercise every bit of a 5-bit register field plus the
# zero and all-ones extremes -> deterministically proves no field
# overlap / correct placement without 32^3 assembler calls.
VREGS = [0, 1, 2, 4, 8, 16, 31]
XREGS = [0, 1, 2, 4, 8, 16, 30]


def _check(res, exp, ctx):
    assert res == exp, "%s: encoder=%r oracle=%r" % (ctx, res, exp)


def test_oracle_self_check():
    # If the oracle itself is broken (e.g. template not assembling on this
    # platform) fail loudly here rather than mis-attributing it to encoders.
    assert assemble("nop") == '\x1f\x20\x03\xd5'
    assert len(assemble("fadd v0.2d, v1.2d, v2.2d")) == 4


def test_FADD_2d_exhaustive_curated():
    for d in VREGS:
        for n in VREGS:
            for m in VREGS:
                cb = CodeBuilder()
                cb.FADD_2d(d, n, m)
                _check(cb.hexdump(),
                       assemble("fadd v%d.2d, v%d.2d, v%d.2d" % (d, n, m)),
                       "FADD_2d(%d,%d,%d)" % (d, n, m))


def test_FMUL_2d_exhaustive_curated():
    for d in VREGS:
        for n in VREGS:
            for m in VREGS:
                cb = CodeBuilder()
                cb.FMUL_2d(d, n, m)
                _check(cb.hexdump(),
                       assemble("fmul v%d.2d, v%d.2d, v%d.2d" % (d, n, m)),
                       "FMUL_2d(%d,%d,%d)" % (d, n, m))


def test_LD1_2d_exhaustive_curated():
    for t in VREGS:
        for n in XREGS:
            cb = CodeBuilder()
            cb.LD1_2d(t, n)
            _check(cb.hexdump(),
                   assemble("ld1 {v%d.2d}, [x%d]" % (t, n)),
                   "LD1_2d(%d,%d)" % (t, n))


def test_ST1_2d_exhaustive_curated():
    for t in VREGS:
        for n in XREGS:
            cb = CodeBuilder()
            cb.ST1_2d(t, n)
            _check(cb.hexdump(),
                   assemble("st1 {v%d.2d}, [x%d]" % (t, n)),
                   "ST1_2d(%d,%d)" % (t, n))


def test_DUP_2d_exhaustive_curated():
    for d in VREGS:
        for n in VREGS:
            cb = CodeBuilder()
            cb.DUP_2d(d, n)
            _check(cb.hexdump(),
                   assemble("dup v%d.2d, v%d.d[0]" % (d, n)),
                   "DUP_2d(%d,%d)" % (d, n))


@settings(max_examples=40, deadline=None)
@given(d=st.integers(0, 31), n=st.integers(0, 31), m=st.integers(0, 31))
def test_FADD_2d_fuzz(d, n, m):
    cb = CodeBuilder()
    cb.FADD_2d(d, n, m)
    assert cb.hexdump() == assemble("fadd v%d.2d, v%d.2d, v%d.2d" % (d, n, m))


@settings(max_examples=40, deadline=None)
@given(d=st.integers(0, 31), n=st.integers(0, 31), m=st.integers(0, 31))
def test_FMUL_2d_fuzz(d, n, m):
    cb = CodeBuilder()
    cb.FMUL_2d(d, n, m)
    assert cb.hexdump() == assemble("fmul v%d.2d, v%d.2d, v%d.2d" % (d, n, m))


@settings(max_examples=40, deadline=None)
@given(t=st.integers(0, 31), n=st.integers(0, 30))
def test_LD1_2d_fuzz(t, n):
    cb = CodeBuilder()
    cb.LD1_2d(t, n)
    assert cb.hexdump() == assemble("ld1 {v%d.2d}, [x%d]" % (t, n))


@settings(max_examples=40, deadline=None)
@given(t=st.integers(0, 31), n=st.integers(0, 30))
def test_ST1_2d_fuzz(t, n):
    cb = CodeBuilder()
    cb.ST1_2d(t, n)
    assert cb.hexdump() == assemble("st1 {v%d.2d}, [x%d]" % (t, n))


def test_DUP_d_exhaustive_curated():
    for d in VREGS:
        for n in VREGS:
            for s in (0, 1):
                cb = CodeBuilder()
                cb.DUP_d(d, n, s)
                _check(cb.hexdump(),
                       assemble("dup d%d, v%d.d[%d]" % (d, n, s)),
                       "DUP_d(%d,%d,%d)" % (d, n, s))


def test_INS_d_exhaustive_curated():
    for d in VREGS:
        for n in VREGS:
            for di in (0, 1):
                for si in (0, 1):
                    cb = CodeBuilder()
                    cb.INS_d(d, n, di, si)
                    _check(cb.hexdump(),
                           assemble("ins v%d.d[%d], v%d.d[%d]"
                                    % (d, di, n, si)),
                           "INS_d(%d,%d,%d,%d)" % (d, n, di, si))
