"""In-process validation for runtime_cogen.generate_for_live_code, using
the TLA toy interpreter (rpython/jit/tl/tla) as a PEDriver-style example.
"""

from rpython.jit.codewriter.jitcode import PEJitCodeMetadata
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


def test_late_jitcode_index_roundtrip_matches_real_production_path():
    """Round-trips the real register_late_jitcode/get_late_jitcode (not
    the deprecated MetaInterpStaticData alias) through _jitcode_at_pos."""
    from rpython.jit.codewriter.jitcode import (
        JitCode, register_late_jitcode, set_late_jitcode_base,
        _late_jitcodes_by_index)
    from rpython.jit.metainterp.resume import _jitcode_at_pos

    _late_jitcodes_by_index.clear()
    try:
        frozen = [JitCode("frozen-%d" % i) for i in range(5)]
        set_late_jitcode_base(len(frozen))

        late1 = JitCode("late-1")
        late1.setup()
        register_late_jitcode(late1, "liveness-chunk-1")
        assert late1.index == 5
        assert _jitcode_at_pos(frozen, late1.index) is late1

        late2 = JitCode("late-2")
        late2.setup()
        register_late_jitcode(late2, "liveness-chunk-2")
        assert late2.index == 6
        assert _jitcode_at_pos(frozen, late2.index) is late2
        assert _jitcode_at_pos(frozen, late1.index) is late1

        assert _jitcode_at_pos(frozen, 0) is frozen[0]
        assert _jitcode_at_pos(frozen, 4) is frozen[4]
    finally:
        _late_jitcodes_by_index.clear()


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

    def test_lookup_miss_triggers_runtime_cogen_with_no_program_preinstalled(self):
        """Lookup miss on an empty PEJitCodeMetadata triggers runtime_cogen
        and re-lookup after that hits the cache instead."""
        from rpython.jit.metainterp.history import ConstInt, ConstPtr
        from rpython.jit.metainterp.warmspot import get_stats
        from rpython.rlib.nonconst import NonConstant

        bytecode = _assemble(COUNTDOWN)

        def interp_w(intvalue):
            w_result = tla.run(NonConstant(bytecode), tla.W_IntObject(intvalue))
            assert isinstance(w_result, tla.W_IntObject)
            return w_result.intvalue

        baseline = self.meta_interp(interp_w, [42], listops=True)
        assert baseline == 0

        counter = [0]

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

            def runtime_cogen(gcref):
                counter[0] += 1
                return generate_for_live_code(
                    extension, linker, codewriter, bytecode, GUARD, gcref,
                    emitter=emitter)

            mainjitcode = linker.mainjitcode(codewriter)
            metadata = PEJitCodeMetadata(0, [], [], [], [], [], [])
            metadata.guard_ref_index = GUARD[1]
            metadata.runtime_cogen = runtime_cogen
            mainjitcode.pe_metadata = metadata

            boxes = [ConstInt(0), ConstPtr(ref)]
            program = metadata.linked_program_for(boxes)
            assert program is not None
            assert counter[0] == 1

            assert metadata.linked_program_for(boxes) is program
            assert counter[0] == 1

            from rpython.translator.backendopt.partialeval_template import (
                LoweredResidualProgram)
            return LoweredResidualProgram(None, program.jitcode, {})

        pe_result = self.meta_interp(
            interp_w, [42], listops=True, pe_linked_setup=install)
        assert pe_result == baseline
        assert counter[0] == 1
        assert get_stats().pe_metadata_count > 0
        assert get_stats().pe_metadata_count > 0

    def test_late_trigger_after_finish_setup_executes_and_matches_the_plain_interpreter(self):
        """First live trace triggers runtime_cogen after finish_setup
        already froze liveness_info/.index; needs register_late_jitcode."""
        from rpython.jit.metainterp.warmspot import get_stats
        from rpython.jit.metainterp import resume
        from rpython.rlib.nonconst import NonConstant

        bytecode = _assemble(COUNTDOWN)

        def interp_w(intvalue):
            total = 0
            i = 2
            while i <= intvalue:
                w_result = tla.run(NonConstant(bytecode), tla.W_IntObject(i))
                assert isinstance(w_result, tla.W_IntObject)
                total += w_result.intvalue
                i += 1
            return total

        baseline = self.meta_interp(interp_w, [12], listops=True)
        assert baseline == 0

        counter = [0]
        late_jitcode_box = []

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

            def runtime_cogen(gcref):
                counter[0] += 1
                program = generate_for_live_code(
                    extension, linker, codewriter, bytecode, GUARD, gcref,
                    emitter=emitter)
                if program is not None:
                    # Needs its own index/extended liveness_info: it was
                    # assembled after finish_setup already ran.
                    staticdata = jitdriver_sd.warmstate.warmrunnerdesc.metainterp_sd
                    staticdata.register_late_jitcode(program.jitcode, codewriter)
                    late_jitcode_box.append(program.jitcode)
                return program

            mainjitcode = linker.mainjitcode(codewriter)
            metadata = PEJitCodeMetadata(0, [], [], [], [], [], [])
            metadata.guard_ref_index = GUARD[1]
            metadata.runtime_cogen = runtime_cogen
            mainjitcode.pe_metadata = metadata
            return None

        resume_hits = [0]
        orig = resume.AbstractResumeDataReader._prepare_next_section

        def counting_prepare_next_section(self, info, jitcode):
            if late_jitcode_box and jitcode is late_jitcode_box[0]:
                resume_hits[0] += 1
            return orig(self, info, jitcode)

        resume.AbstractResumeDataReader._prepare_next_section = (
            counting_prepare_next_section)
        try:
            pe_result = self.meta_interp(
                interp_w, [12], listops=True, pe_linked_setup=install)
        finally:
            resume.AbstractResumeDataReader._prepare_next_section = orig
        assert pe_result == baseline
        assert counter[0] >= 1
        assert resume_hits[0] > 0, (
            "resume never reached the late jitcode -- COUNTDOWN's loop-exit "
            "guard should fail and resume through its liveness for real")
        assert get_stats().pe_metadata_count > 0

    def test_late_trigger_native_path_executes_and_matches_the_plain_interpreter(self):
        """Late trigger via generate_for_live_code's native_table path.
        Needs register_native_insn_coverage run first, or NativeAssembler
        silently declines and no linked program is ever produced."""
        from rpython.jit.metainterp.warmspot import get_stats
        from rpython.jit.metainterp import resume
        from rpython.rlib.nonconst import NonConstant

        bytecode = _assemble(COUNTDOWN)

        def interp_w(intvalue):
            total = 0
            i = 2
            while i <= intvalue:
                w_result = tla.run(NonConstant(bytecode), tla.W_IntObject(i))
                assert isinstance(w_result, tla.W_IntObject)
                total += w_result.intvalue
                i += 1
            return total

        baseline = self.meta_interp(interp_w, [12], listops=True)
        assert baseline == 0

        counter = [0]
        late_jitcode_box = []
        coverage_state = []

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
            native_table = emitter.native_table()
            coverage_state.append((codewriter, native_table))

            def runtime_cogen(gcref):
                counter[0] += 1
                program = generate_for_live_code(
                    extension, linker, codewriter, bytecode, GUARD, gcref,
                    native_table=native_table)
                if program is not None:
                    assert program.jitcode.own_liveness_info is not None
                    from rpython.jit.codewriter.jitcode import (
                        register_late_jitcode, set_late_jitcode_base)
                    staticdata = jitdriver_sd.warmstate.warmrunnerdesc.metainterp_sd
                    set_late_jitcode_base(len(staticdata.jitcodes))
                    register_late_jitcode(
                        program.jitcode, program.jitcode.own_liveness_info)
                    assert program.jitcode not in staticdata.jitcodes
                    late_jitcode_box.append(program.jitcode)
                return program

            mainjitcode = linker.mainjitcode(codewriter)
            metadata = PEJitCodeMetadata(0, [], [], [], [], [], [])
            metadata.guard_ref_index = GUARD[1]
            metadata.runtime_cogen = runtime_cogen
            mainjitcode.pe_metadata = metadata
            return None

        def jitcode_setup(mainjitcode):
            # Must run after make_jitcodes() and before finish_setup: the
            # only window a dict growth on assembler.insns is legal in.
            if not coverage_state:
                return
            codewriter, native_table = coverage_state[0]
            from rpython.translator.backendopt.jitcode_emitter import (
                stamp_descr_indices, register_native_insn_coverage)
            stamp_descr_indices(codewriter, native_table)
            register_native_insn_coverage(codewriter, native_table)

        resume_hits = [0]
        orig = resume.AbstractResumeDataReader._prepare_next_section

        def counting_prepare_next_section(self, info, jitcode):
            if late_jitcode_box and jitcode is late_jitcode_box[0]:
                resume_hits[0] += 1
            return orig(self, info, jitcode)

        resume.AbstractResumeDataReader._prepare_next_section = (
            counting_prepare_next_section)
        try:
            pe_result = self.meta_interp(
                interp_w, [12], listops=True, pe_linked_setup=install,
                pe_jitcode_setup=jitcode_setup)
        finally:
            resume.AbstractResumeDataReader._prepare_next_section = orig
        assert pe_result == baseline
        assert counter[0] >= 1
        assert get_stats().pe_metadata_count > 0
        assert resume_hits[0] > 0, (
            "resume never reached the late jitcode -- COUNTDOWN's loop-exit "
            "guard should fail and resume through its own_liveness_info "
            "for real")


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


def _threshold_test_ref():
    from rpython.rtyper.lltypesystem import lltype, llmemory

    return lltype.cast_opaque_ptr(
        llmemory.GCREF, lltype.malloc(lltype.GcStruct('S')))


def test_cogen_threshold_defers_generation_and_caching():
    """Below cogen_threshold, nothing may be cached under the ref: a
    cached None there would be a permanent decline, disabling cogen."""
    from rpython.jit.codewriter.jitcode import (
        JitCode, PEJitCodeMetadata, PELinkedProgram)
    from rpython.jit.metainterp.history import ConstInt, ConstPtr

    ref = _threshold_test_ref()
    jitcode = JitCode("threshold-test")
    jitcode.setup()
    program = PELinkedProgram(jitcode, [], [])
    program.guard_ref = ref

    calls = [0]

    def runtime_cogen(gcref):
        calls[0] += 1
        return program

    metadata = PEJitCodeMetadata(0, [], [], [], [], [], [])
    metadata.guard_ref_index = 1
    metadata.runtime_cogen = runtime_cogen
    metadata.cogen_threshold = 2

    boxes = [ConstInt(0), ConstPtr(ref)]

    assert metadata.linked_program_for(boxes) is None
    assert calls[0] == 0
    assert ref not in metadata._program_cache

    assert metadata.linked_program_for(boxes) is program
    assert calls[0] == 1
    assert metadata._program_cache[ref] is program

    assert metadata.linked_program_for(boxes) is program
    assert calls[0] == 1


def test_cogen_threshold_zero_generates_on_first_miss():
    """cogen_threshold=0 (the default): callback runs on the first miss."""
    from rpython.jit.codewriter.jitcode import (
        JitCode, PEJitCodeMetadata, PELinkedProgram)
    from rpython.jit.metainterp.history import ConstInt, ConstPtr

    ref = _threshold_test_ref()
    jitcode = JitCode("threshold-zero-test")
    jitcode.setup()
    program = PELinkedProgram(jitcode, [], [])
    program.guard_ref = ref

    calls = [0]

    def runtime_cogen(gcref):
        calls[0] += 1
        return program

    metadata = PEJitCodeMetadata(0, [], [], [], [], [], [])
    metadata.guard_ref_index = 1
    metadata.runtime_cogen = runtime_cogen
    assert metadata.cogen_threshold == 0

    boxes = [ConstInt(0), ConstPtr(ref)]
    assert metadata.linked_program_for(boxes) is program
    assert calls[0] == 1


def test_cogen_real_decline_is_still_cached_once_threshold_reached():
    """A real decline (None) is cached once the threshold lets it run."""
    from rpython.jit.codewriter.jitcode import PEJitCodeMetadata
    from rpython.jit.metainterp.history import ConstInt, ConstPtr

    ref = _threshold_test_ref()
    calls = [0]

    def runtime_cogen(gcref):
        calls[0] += 1
        return None

    metadata = PEJitCodeMetadata(0, [], [], [], [], [], [])
    metadata.guard_ref_index = 1
    metadata.runtime_cogen = runtime_cogen
    metadata.cogen_threshold = 1

    boxes = [ConstInt(0), ConstPtr(ref)]
    assert metadata.linked_program_for(boxes) is None
    assert calls[0] == 1
    assert metadata._program_cache[ref] is None

    assert metadata.linked_program_for(boxes) is None
    assert calls[0] == 1
