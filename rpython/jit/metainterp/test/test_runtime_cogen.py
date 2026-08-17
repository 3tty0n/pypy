"""In-process validation for runtime_cogen.generate_for_live_code, using
the TLA toy interpreter (rpython/jit/tl/tla) as a PEDriver-style example.
"""

from rpython.jit.metainterp.test.support import LLJitMixin
from rpython.jit.tl.tla import tla
from rpython.jit.tl.tla import offline as tla_offline
from rpython.translator.backendopt.jitcode_emitter import ProgramEmitter
from rpython.translator.backendopt.portal_linker import PortalLinker
from rpython.translator.backendopt.runtime_cogen import generate_for_live_code
from rpython.translator.backendopt.partialeval import make_rtyped_constant
from rpython.translator.backendopt.generating_extension import (
    GeneratingExtension)
from rpython.translator.translator import graphof
from rpython.rtyper.lltypesystem import lltype, llmemory


def _assemble(mylist):
    return ''.join([chr(x) for x in mylist])


COUNTDOWN = [
    tla.CONST_INT, 1,
    tla.SUB,
    tla.DUP,
    tla.JUMP_IF, 0,
    tla.RETURN,
]

PORTAL_SOURCES = (2, 1)          # (self, bytecode) among (pc, bytecode, self)
RUNTIME_NAMES = ("self", "bytecode")
JIT_MERGE_POINT_ARGS = ("pc", "bytecode", "self")
GUARD = (0, 1)                   # (pc, bytecode) indices into the boxes


def _tla_portal_linker(jitdriver_sd):
    return PortalLinker(
        jitdriver_sd, PORTAL_SOURCES, RUNTIME_NAMES,
        jit_merge_point_args=JIT_MERGE_POINT_ARGS,
        null_names=("bytecode",), static_name="opcode",
        split_names=("pc",), hole_names=("oparg",),
        name="linked-tla-cogen")


def _bytecode_ref(translator, bytecode):
    """Must go through the rtyper's own conversion, not a direct cast, to
    match the prebuilt low-level object the box already carries."""
    graph = graphof(translator, tla.Frame.interp_step.im_func)
    var = graph.startblock.inputargs[graph.signature[0].index("bytecode")]
    constant = make_rtyped_constant(translator, var, bytecode)
    return lltype.cast_opaque_ptr(llmemory.GCREF, constant.value)


class TestGenerateForLiveCode(LLJitMixin):

    def test_generate_for_live_code_installs_from_a_real_warmrunnerdesc(self):
        """Program installs on the real portal metadata, precompiled and
        ordinary generation agree byte-for-byte, and the guard is correct.
        """
        from rpython.rlib.nonconst import NonConstant

        bytecode = _assemble(COUNTDOWN)

        def interp_w(intvalue):
            # NonConstant: bytecode must stay a runtime value, not fold to
            # a compile-time constant, for jit_merge_point's split to work.
            w_result = tla.run(NonConstant(bytecode), tla.W_IntObject(intvalue))
            assert isinstance(w_result, tla.W_IntObject)
            return w_result.intvalue

        captured = {}

        def install(codewriter, jitdriver_sd, translator):
            extension = tla_offline.build_generating_extension(translator)
            linker = _tla_portal_linker(jitdriver_sd)
            ref = _bytecode_ref(translator, bytecode)

            used = dict((key, extension.templates[key]) for key in
                       (tla.CONST_INT, tla.SUB, tla.DUP, tla.JUMP_IF,
                        tla.RETURN))
            emitter = ProgramEmitter(
                codewriter, jitdriver_sd, "opcode", ("pc",), ("oparg",),
                RUNTIME_NAMES, jit_merge_point_args=JIT_MERGE_POINT_ARGS)
            emitter.precompile_fragments(used)
            before = len(emitter._fragments)
            assert before == 2 * len(used)

            precompiled = generate_for_live_code(
                extension, linker, codewriter, bytecode, GUARD, ref,
                emitter=emitter)
            assert precompiled is not None
            assert len(emitter._fragments) == before

            ordinary = generate_for_live_code(
                extension, linker, codewriter, bytecode, GUARD, ref)
            assert ordinary is not None

            captured["precompiled"] = precompiled
            captured["ordinary"] = ordinary
            captured["ref"] = ref
            captured["jitdriver_sd"] = jitdriver_sd
            return precompiled

        self.meta_interp(
            interp_w, [42], listops=True, pe_linked_setup=install,
            graph_and_interp_only=True)

        jitdriver_sd = captured["jitdriver_sd"]
        linked = jitdriver_sd.mainjitcode.pe_metadata.linked_programs
        assert captured["precompiled"] in linked
        assert captured["ordinary"] in linked

        precompiled = captured["precompiled"]
        ordinary = captured["ordinary"]
        assert precompiled.jitcode.code == ordinary.jitcode.code
        assert precompiled.jitcode.constants_i == ordinary.jitcode.constants_i

        ref = captured["ref"]
        assert precompiled.guard_ref == ref
        assert precompiled.guard_pc_index == GUARD[0]
        assert precompiled.guard_ref_index == GUARD[1]
        assert precompiled.matches_ref(ref)
        other_ref = lltype.nullptr(llmemory.GCREF.TO)
        assert not precompiled.matches_ref(other_ref)
        for pc in precompiled.guard_pcs:
            assert precompiled._covers(pc)
        assert precompiled.is_legit_entry_pc(0)

    def test_generate_for_live_code_executes_and_matches_the_plain_interpreter(self):
        """Runs meta_interp for real and checks the traced result against
        a plain, non-generated interpretation of the same bytecode."""
        from rpython.jit.metainterp.warmspot import get_stats
        from rpython.rlib.nonconst import NonConstant

        bytecode = _assemble(COUNTDOWN)

        def interp_w(intvalue):
            w_result = tla.run(NonConstant(bytecode), tla.W_IntObject(intvalue))
            assert isinstance(w_result, tla.W_IntObject)
            return w_result.intvalue

        baseline = self.meta_interp(interp_w, [42], listops=True)
        assert baseline == 0

        def install(codewriter, jitdriver_sd, translator):
            extension = tla_offline.build_generating_extension(translator)
            linker = _tla_portal_linker(jitdriver_sd)
            ref = _bytecode_ref(translator, bytecode)
            program = generate_for_live_code(
                extension, linker, codewriter, bytecode, GUARD, ref)
            assert program is not None
            return program

        pe_result = self.meta_interp(
            interp_w, [42], listops=True, pe_linked_setup=install)
        assert pe_result == baseline
        assert get_stats().pe_metadata_count > 0


def test_generate_for_live_code_declines_missing_template():
    """An instruction with no template is declined by the scan alone,
    without touching linker/guard/ref."""
    from rpython.translator.backendopt.test.test_partialeval_template_lowering \
        import byte_pair_decoder, get_graph

    OP_DEC_JUMP = 0
    OP_HALT = 1

    def interpret_one(opcode, oparg, pc, value):
        if opcode == OP_DEC_JUMP:
            if value > 0:
                return oparg, value - 1
            return pc + 2, value
        return -1, value

    interpret_one._pe_static_args_ = ("opcode",)
    interpret_one._pe_split_args_ = ("pc",)
    _graph, translator = get_graph(interpret_one, [int, int, int, int])
    extension = GeneratingExtension.from_step_function(
        translator, interpret_one, [OP_DEC_JUMP], byte_pair_decoder)

    code = chr(OP_DEC_JUMP) + chr(0) + chr(OP_HALT) + chr(0)
    result = generate_for_live_code(
        extension, None, None, code, guard=(0, 1), ref=None)
    assert result is None
    assert extension.last_blocked == (2, OP_HALT)

    from rpython.jit.codewriter.jitcode import PEJitCodeMetadata
    metadata = PEJitCodeMetadata(0, [], [], [], [], [0], [0])
    assert metadata._program_cache is None
    assert metadata.linked_program_for([]) is None
    assert metadata._program_cache is None
