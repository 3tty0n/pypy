"""In-process repro for the SOM ref-cache production crash."""

from rpython.jit.codewriter.jitcode import PEJitCodeMetadata
from rpython.jit.metainterp.test.support import LLJitMixin
from rpython.jit.tl.tla import tla_refcache as refcache
from rpython.jit.tl.tla import offline_refcache as refcache_offline
from rpython.translator.backendopt.jitcode_emitter import ProgramEmitter
from rpython.translator.backendopt.portal_linker import PortalLinker
from rpython.translator.backendopt.runtime_cogen import generate_for_live_code
from rpython.translator.backendopt.partialeval import make_rtyped_constant
from rpython.translator.translator import graphof
from rpython.rtyper.lltypesystem import lltype, llmemory


def _assemble(mylist):
    return ''.join([chr(x) for x in mylist])


# Bytecode: CONST_INT N; SUB 1 (pc=2, loop head); JUMP_IF 2; RETURN.
N = 4
COUNTDOWN = [
    refcache.CONST_INT, N,
    refcache.SUB, 1,
    refcache.JUMP_IF, 2,
    refcache.RETURN,
]

PORTAL_SOURCES = (2, 1, 3)        # (self, bytecode, top) among the boxes
RUNTIME_NAMES = ("self", "bytecode", "top")
JIT_MERGE_POINT_ARGS = ("pc", "bytecode", "self", "top")
GUARD = (0, 1)                    # (pc, bytecode) indices into the boxes


def _portal_linker(jitdriver_sd):
    return PortalLinker(
        jitdriver_sd, PORTAL_SOURCES, RUNTIME_NAMES,
        jit_merge_point_args=JIT_MERGE_POINT_ARGS,
        null_names=("bytecode",), static_name="opcode",
        split_names=("pc",), hole_names=("oparg",),
        name="linked-refcache")


def _bytecode_ref(translator, bytecode):
    graph = graphof(translator, refcache.Frame.interp_step.im_func)
    var = graph.startblock.inputargs[graph.signature[0].index("bytecode")]
    constant = make_rtyped_constant(translator, var, bytecode)
    return lltype.cast_opaque_ptr(llmemory.GCREF, constant.value)


class TestRefCacheGenerateForLiveCode(LLJitMixin):

    def test_executes_and_matches_the_plain_interpreter(self):
        """Baseline: install ahead of the first trace, no late trigger."""
        from rpython.rlib.nonconst import NonConstant

        bytecode = _assemble(COUNTDOWN)

        def interp_w(intvalue):
            w_result = refcache.run(NonConstant(bytecode), intvalue)
            return w_result.intvalue

        baseline = self.meta_interp(interp_w, [N], listops=True)
        assert baseline == 0

        def install(codewriter, jitdriver_sd, translator):
            extension = refcache_offline.build_generating_extension(translator)
            linker = _portal_linker(jitdriver_sd)
            ref = _bytecode_ref(translator, bytecode)
            program = generate_for_live_code(
                extension, linker, codewriter, bytecode, GUARD, ref)
            assert program is not None
            return program

        pe_result = self.meta_interp(
            interp_w, [N], listops=True, pe_linked_setup=install)
        assert pe_result == baseline

    def test_late_trigger_ssarepr_path_matches_interpreter(self):
        """SSARepr back end: the crash's own traceback runs through this."""
        from rpython.rlib.nonconst import NonConstant

        bytecode = _assemble(COUNTDOWN)

        def interp_w(intvalue):
            w_result = refcache.run(NonConstant(bytecode), intvalue)
            return w_result.intvalue

        baseline = self.meta_interp(interp_w, [N], listops=True)
        assert baseline == 0

        counter = [0]

        def install(codewriter, jitdriver_sd, translator):
            extension = refcache_offline.build_generating_extension(translator)
            linker = _portal_linker(jitdriver_sd)
            ref = _bytecode_ref(translator, bytecode)

            used = dict((key, extension.templates[key]) for key in
                       (refcache.CONST_INT, refcache.SUB,
                        refcache.JUMP_IF, refcache.RETURN))
            emitter = ProgramEmitter(
                codewriter, jitdriver_sd, "opcode", ("pc",), ("oparg",),
                RUNTIME_NAMES, jit_merge_point_args=JIT_MERGE_POINT_ARGS,
                null_names=("bytecode",))
            emitter.precompile_fragments(used)

            def runtime_cogen(gcref):
                counter[0] += 1
                program = generate_for_live_code(
                    extension, linker, codewriter, bytecode, GUARD, gcref,
                    emitter=emitter)
                if program is not None:
                    warmrunnerdesc = jitdriver_sd.warmstate.warmrunnerdesc
                    staticdata = warmrunnerdesc.metainterp_sd
                    staticdata.register_late_jitcode(
                        program.jitcode, codewriter)
                return program

            mainjitcode = linker.mainjitcode(codewriter)
            metadata = PEJitCodeMetadata(0, [], [], [], [], [], [])
            metadata.guard_ref_index = GUARD[1]
            metadata.runtime_cogen = runtime_cogen
            mainjitcode.pe_metadata = metadata
            return None

        pe_result = self.meta_interp(
            interp_w, [N], listops=True, pe_linked_setup=install)
        assert pe_result == baseline
        assert counter[0] >= 1

    def test_late_trigger_native_path_matches_interpreter(self):
        """Same as above, native_fragments.py/native_pipeline.py back end."""
        from rpython.rlib.nonconst import NonConstant

        bytecode = _assemble(COUNTDOWN)

        def interp_w(intvalue):
            w_result = refcache.run(NonConstant(bytecode), intvalue)
            return w_result.intvalue

        baseline = self.meta_interp(interp_w, [N], listops=True)
        assert baseline == 0

        counter = [0]

        def install(codewriter, jitdriver_sd, translator):
            extension = refcache_offline.build_generating_extension(translator)
            linker = _portal_linker(jitdriver_sd)
            ref = _bytecode_ref(translator, bytecode)

            used = dict((key, extension.templates[key]) for key in
                       (refcache.CONST_INT, refcache.SUB,
                        refcache.JUMP_IF, refcache.RETURN))
            emitter = ProgramEmitter(
                codewriter, jitdriver_sd, "opcode", ("pc",), ("oparg",),
                RUNTIME_NAMES, jit_merge_point_args=JIT_MERGE_POINT_ARGS,
                null_names=("bytecode",))
            emitter.precompile_fragments(used)
            native_table = emitter.native_table()

            def runtime_cogen(gcref):
                counter[0] += 1
                program = generate_for_live_code(
                    extension, linker, codewriter, bytecode, GUARD, gcref,
                    native_table=native_table)
                if program is not None:
                    warmrunnerdesc = jitdriver_sd.warmstate.warmrunnerdesc
                    staticdata = warmrunnerdesc.metainterp_sd
                    staticdata.register_late_jitcode(
                        program.jitcode, codewriter)
                return program

            mainjitcode = linker.mainjitcode(codewriter)
            metadata = PEJitCodeMetadata(0, [], [], [], [], [], [])
            metadata.guard_ref_index = GUARD[1]
            metadata.runtime_cogen = runtime_cogen
            mainjitcode.pe_metadata = metadata
            return None

        pe_result = self.meta_interp(
            interp_w, [N], listops=True, pe_linked_setup=install)
        assert pe_result == baseline
        assert counter[0] >= 1
