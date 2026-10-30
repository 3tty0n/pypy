import py
import pytest

from rpython.jit.codewriter import support
from rpython.jit.codewriter.codewriter import CodeWriter
from rpython.jit.codewriter.genextension import GenExtension
from rpython.jit.codewriter.genextrun import generate_run_function
from rpython.jit.codewriter.test.test_codewriter import (
    FakeCPU, FakeJitDriverSD, FakePolicy)


class _AssemblerSnapshot(object):
    pass

    def __init__(self, assembler):
        self.__dict__.update(assembler.__dict__)
        self.startpoints = dict(assembler.startpoints)
        self.label_positions = dict(assembler.label_positions)


def _tla_jitcodes(genextrun='off', decorate=False):
    from rpython.jit.tl.tla import tla
    from rpython.jit.codewriter.test import test_codewriter

    test_codewriter.FakeFieldDescr.is_always_pure = lambda self: False
    test_codewriter.FakeArrayDescr.is_always_pure = lambda self: False

    if hasattr(tla.Frame, '_virtualizable_'):
        del tla.Frame._virtualizable_
        tla.jitdriver.virtualizables = []

    def entry(x):
        frame = tla.Frame("\x00\x00")
        frame.push(tla.W_IntObject(x))
        return frame.interp().getrepr()

    rtyper = support.annotate(entry, [5])
    graphs = rtyper.annotator.translator.graphs
    portal = [g for g in graphs if g.name == 'Frame.interp'][0]
    jd = FakeJitDriverSD(portal)
    jd.jitdriver = tla.jitdriver
    jd.index = 0
    jd.mainjitcode = None
    cw = CodeWriter(FakeCPU(rtyper), [jd])
    cw.genextrun = genextrun

    captured = []
    assemble = cw.assembler.assemble

    def spy(ssarepr, jitcode=None, num_regs=None):
        res = assemble(ssarepr, jitcode, num_regs)
        snap = _AssemblerSnapshot(cw.assembler)
        captured.append((ssarepr, jitcode if jitcode is not None else res,
                         snap))
        return res

    cw.assembler.assemble = spy
    if decorate:
        tla.Frame.interp.im_func.generate_jit_extension = True
    try:
        cw.find_all_graphs(FakePolicy())
        cw.make_jitcodes(verbose=False)
    finally:
        if decorate:
            del tla.Frame.interp.im_func.generate_jit_extension
    return cw.assembler, captured


def test_genext_on_tla(tmpdir):
    assembler, captured = _tla_jitcodes()
    ssarepr, jitcode, snap = [(s, j, a) for s, j, a in captured
                              if s.name == 'Frame.interp'][0]
    GenExtension(snap, ssarepr, jitcode).generate()
    source = jitcode._genext_source
    tmpdir.join("tla_interp_shortcut.py").write(source)
    assert "def jit_shortcut" in source


def test_genext_run_mode(tmpdir):
    assembler, captured = _tla_jitcodes()
    ssarepr, jitcode, snap = [(s, j, a) for s, j, a in captured
                              if s.name == 'Frame.interp'][0]
    genext = GenExtension(snap, ssarepr, jitcode)
    genext.generate()
    generate_run_function(genext)


def test_pipeline_generates_run_functions_for_all_jitcodes():
    assembler, captured = _tla_jitcodes(genextrun='all')
    assert captured
    for ssarepr, jitcode, snap in captured:
        assert jitcode.genext_run_function is not None
        assert 'def jit_run(bh):' in jitcode._genext_run_source


def test_pipeline_critical_skips_undecorated_jitcodes():
    assembler, captured = _tla_jitcodes(genextrun='critical')
    assert captured
    for ssarepr, jitcode, snap in captured:
        assert not ssarepr.genextension
        assert jitcode.genext_run_function is None


def test_pipeline_critical_generates_decorated_jitcode():
    assembler, captured = _tla_jitcodes(genextrun='critical', decorate=True)
    by_name = dict((ssarepr.name, (ssarepr, jitcode))
                   for ssarepr, jitcode, snap in captured)
    ssarepr, jitcode = by_name['Frame.interp']
    assert ssarepr.genextension
    assert jitcode.genext_run_function is not None
    for name, (ssarepr, jitcode) in by_name.items():
        if name != 'Frame.interp':
            assert not ssarepr.genextension
            assert jitcode.genext_run_function is None


def test_pipeline_off_by_default():
    assembler, captured = _tla_jitcodes()
    assert captured
    for ssarepr, jitcode, snap in captured:
        assert jitcode.genext_run_function is None


def test_genext_on_residual(tmpdir):
    from rpython.translator.backendopt.test.test_native_pipeline import (
        _toy_setup, OP_DEC_JUMP, OP_HALT)

    code = (chr(OP_DEC_JUMP) + chr(0) + chr(OP_DEC_JUMP) + chr(0) +
            chr(OP_HALT) + chr(0))
    program, emitter = _toy_setup(code)
    jitcode, _positions = emitter.emit(program, "orig")
    assembler = emitter.codewriter.assembler
    GenExtension(assembler, emitter.last_ssarepr, jitcode).generate()
    source = jitcode._genext_source
    tmpdir.join("residual_shortcut.py").write(source)
    assert "def jit_shortcut" in source
    assert jitcode.genext_function is not None
