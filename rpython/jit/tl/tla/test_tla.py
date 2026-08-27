import py
from rpython.jit.tl.tla import tla
from rpython.jit.tl.tla import offline

def test_stack():
    f = tla.Frame('')
    f.push(1)
    f.push(2)
    f.push(3)
    assert f.pop() == 3
    assert f.pop() == 2
    assert f.pop() == 1
    py.test.raises(AssertionError, f.pop)


def test_W_IntObject():
    w_a = tla.W_IntObject(0)
    w_b = tla.W_IntObject(10)
    w_c = tla.W_IntObject(32)
    assert not w_a.is_true()
    assert w_b.is_true()
    assert w_c.is_true()
    assert w_b.add(w_c).intvalue == 42
    assert w_b.getrepr() == '10'


def assemble(mylist):
    return ''.join([chr(x) for x in mylist])

def interp(mylist, w_arg):
    bytecode = assemble(mylist)
    return tla.run(bytecode, w_arg)

def test_interp():
    code = [
        tla.RETURN
        ]
    res = interp(code, tla.W_IntObject(42))
    assert res.intvalue == 42

def test_pop():
    code = [
        tla.CONST_INT, 99,
        tla.POP,
        tla.RETURN
        ]
    res = interp(code, tla.W_IntObject(42))
    assert res.intvalue == 42

def test_dup():
    code = [
        tla.DUP,
        tla.ADD,
        tla.RETURN
        ]
    res = interp(code, tla.W_IntObject(41))
    assert res.intvalue == 2 * 41

def test_bogus_return():
    code = [
        tla.CONST_INT, 123,
        tla.RETURN # stack depth == 2 here, error!
        ]
    py.test.raises(AssertionError, "interp(code, tla.W_IntObject(234))")

def test_add():
    code = [
        tla.CONST_INT, 20,
        tla.ADD,
        tla.RETURN
        ]
    res = interp(code, tla.W_IntObject(22))
    assert res.intvalue == 42

def test_sub():
    code = [
        tla.CONST_INT, 20,
        tla.SUB,
        tla.RETURN
        ]
    res = interp(code, tla.W_IntObject(22))
    assert res.intvalue == 2

def test_jump_if():
    code = [
        tla.JUMP_IF, 5,   # jump to target
        tla.CONST_INT, 123,
        tla.RETURN,
        tla.CONST_INT, 234,  # target
        tla.RETURN
        ]
    res = interp(code, tla.W_IntObject(0))
    assert res.intvalue == 123

    res = interp(code, tla.W_IntObject(1))
    assert res.intvalue == 234


def test_newstr():
    code = [
        tla.POP,
        tla.NEWSTR, ord('x'),
        tla.RETURN
        ]
    res = interp(code, tla.W_IntObject(0))
    assert isinstance(res, tla.W_StringObject)
    assert res.strvalue == 'x'

# ____________________________________________________________
# EXERCISES
# ____________________________________________________________


def test_add_strings():
    py.test.skip('exercise!')
    code = [
        tla.NEWSTR, ord('d'),
        tla.ADD,
        tla.NEWSTR, ord('!'),
        tla.ADD,
        tla.RETURN
        ]
    res = interp(code, tla.W_StringObject('Hello worl'))
    assert res.strvalue == 'Hello world!'

def test_mul():
    py.test.skip('exercise!')
    code = [
        tla.CONST_INT, 2,
        tla.MUL,
        tla.RETURN
        ]
    res = interp(code, tla.W_IntObject(21))
    assert res.intvalue == 42

def test_mul_strings():
    py.test.skip('exercise!')
    code = [
        tla.CONST_INT, 3,
        tla.MUL,
        tla.RETURN
        ]
    res = interp(code, tla.W_StringObject('foo '))
    assert res.strvalue == 'foo foo foo '

def test_div_float():
    py.test.skip('exercise!')
    code = [
        tla.CONST_INT, 2,
        tla.DIV,
        tla.RETURN
        ]
    res = interp(code, tla.W_IntObject(5))
    assert isinstance(res, tla.W_FloatObject)
    assert res.floatval == 2.5

# ____________________________________________________________

from rpython.jit.metainterp.test.support import LLJitMixin

class TestLLtype(LLJitMixin):
    def test_loop(self):
        code = [
                tla.DUP,
                tla.CONST_INT, 1,
                tla.SUB,
                tla.DUP,
                tla.JUMP_IF, 1,
                tla.POP,
                tla.CONST_INT, 1,
                tla.SUB,
                tla.DUP,
                tla.JUMP_IF, 0,
                tla.RETURN
            ]
        def interp_w(intvalue):
            w_result = interp(code, tla.W_IntObject(intvalue))
            assert isinstance(w_result, tla.W_IntObject)
            return w_result.intvalue
        res = self.meta_interp(interp_w, [42], listops=True)
        assert res == 0


def test_offline_pe_catalog_and_linked_tla_loop():
    from rpython.jit.codewriter.codewriter import CodeWriter
    from rpython.jit.codewriter.test.test_codewriter import FakeCPU
    from rpython.translator.translator import TranslationContext

    t = TranslationContext()
    t.buildannotator().build_types(tla.run, [str, tla.W_IntObject])
    t.buildrtyper().specialize()
    catalog = offline.build_template_catalog(t)

    assert set(catalog.keys()) == set(range(len(tla.OPNAMES)))
    for opcode in catalog.keys():
        template = catalog.lookup(opcode)
        # A few opcode semantics keep their own int_eq (type/stack checks).
        assert sum(op.opname == "int_eq"
                   for op in template.operations) < 9

    bytecode = assemble([
        tla.CONST_INT, 1,
        tla.SUB,
        tla.DUP,
        tla.JUMP_IF, 0,
        tla.RETURN,
    ])
    linked = offline.link_bytecode(catalog, bytecode)

    assert set(linked.blocks) == set([0, 2, 3, 4, 6])
    assert 0 in linked.loop_headers
    assert (4, 0) in linked.backedges
    assert linked.blocks[6].has_finish

    codewriter = CodeWriter(FakeCPU(t.rtyper), [])
    codewriter.callcontrol.candidate_graphs = set(t.graphs)
    lowered = linked.lower(codewriter, "linked-tla-countdown")
    dump = lowered.jitcode.dump()
    assert "goto_if_not" in dump
    assert "strgetitem" not in dump
    assert set(lowered.entry_positions) == set(linked.blocks)


class TestOfflineTracing(LLJitMixin):
    def test_compare_tracing_work(self):
        from rpython.rlib.jit import Counters
        from rpython.jit.metainterp.jitprof import Profiler
        from rpython.jit.metainterp.warmspot import get_stats
        from rpython.jit.metainterp import pyjitpl
        from rpython.translator.translator import TranslationContext

        code = [
            tla.CONST_INT, 1,
            tla.SUB,
            tla.DUP,
            tla.JUMP_IF, 0,
            tla.RETURN,
        ]
        bytecode = assemble(code)

        def interp_w(intvalue):
            w_result = interp(code, tla.W_IntObject(intvalue))
            assert isinstance(w_result, tla.W_IntObject)
            return w_result.intvalue

        baseline = self.meta_interp(
            interp_w, [42], listops=True, ProfilerClass=Profiler)
        assert baseline == 0
        baseline_profiler = pyjitpl._warmrunnerdesc.metainterp_sd.profiler
        baseline_time = baseline_profiler.get_times(Counters.TRACING)
        baseline_ops = baseline_profiler.get_counter(Counters.RECORDED_OPS)

        def install_linked(codewriter, jitdriver_sd, translator):
            return offline.lower_and_install(
                codewriter, jitdriver_sd, translator, bytecode)

        pe_result = self.meta_interp(
            interp_w, [42], listops=True, ProfilerClass=Profiler,
            pe_linked_setup=install_linked)
        assert pe_result == baseline
        pe_profiler = pyjitpl._warmrunnerdesc.metainterp_sd.profiler
        pe_time = pe_profiler.get_times(Counters.TRACING)
        pe_ops = pe_profiler.get_counter(Counters.RECORDED_OPS)

        assert get_stats().pe_metadata_count > 0
        print("TLA tracing baseline: %.9fs, %d recorded ops" %
              (baseline_time, baseline_ops))
        print("TLA tracing offline PE: %.9fs, %d recorded ops" %
              (pe_time, pe_ops))
        assert baseline_ops > 0 and pe_ops > 0
        assert pe_ops < baseline_ops


def test_interp_step_declares_pc_as_split_argument():
    interp_step = tla.Frame.interp_step.im_func
    assert interp_step._pe_static_args_ == ('opcode',)
    assert interp_step._pe_split_args_ == ('pc',)
