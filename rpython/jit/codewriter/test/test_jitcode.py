from rpython.jit.codewriter.jitcode import (JitCode, PEJitCodeMetadata,
    PELinkedProgram)
from rpython.jit.metainterp.history import ConstInt, ConstPtr
from rpython.rtyper.lltypesystem import lltype, llmemory


def test_num_regs():
    j = JitCode("test")
    j.setup(num_regs_i=12, num_regs_r=34, num_regs_f=56)
    assert j.num_regs_i() == 12
    assert j.num_regs_r() == 34
    assert j.num_regs_f() == 56
    j.setup(num_regs_i=0, num_regs_r=0, num_regs_f=0)
    assert j.num_regs_i() == 0
    assert j.num_regs_r() == 0
    assert j.num_regs_f() == 0
    j.setup(num_regs_i=255, num_regs_r=255, num_regs_f=255)
    assert j.num_regs_i() == 255
    assert j.num_regs_r() == 255
    assert j.num_regs_f() == 255


# ____________________________________________________________
# linked_program_for's ref-keyed cache (see PEJitCodeMetadata in jitcode.py)

def _new_ref():
    S = lltype.GcStruct('S')
    s = lltype.malloc(S)
    return lltype.cast_opaque_ptr(llmemory.GCREF, s)


def _make_program(ref_index, pc_index=-1, pcs=None):
    jitcode = JitCode("callee")
    jitcode.setup(code='')
    program = PELinkedProgram(jitcode, [], [])
    program.set_matcher(pc_index, pcs or [], ref_index, pcs or [], False)
    return program


def _counting_matcher(expected_ref):
    # Stands in for the real matcher (a full bytecode compare): records
    # every call so the tests can assert the cache stops it from re-running.
    calls = []
    def matcher(actual):
        calls.append(actual)
        return actual == expected_ref
    matcher.calls = calls
    return matcher


def test_linked_program_for_caches_ref_and_skips_matcher_on_hit():
    ref_a = _new_ref()
    metadata = PEJitCodeMetadata(0, [], [], [], [], [0], [0])
    decoy1 = _make_program(ref_index=0)
    decoy2 = _make_program(ref_index=0)
    match = _make_program(ref_index=0)
    decoy1.matcher = _counting_matcher(lltype.nullptr(llmemory.GCREF.TO))
    decoy2.matcher = _counting_matcher(lltype.nullptr(llmemory.GCREF.TO))
    match.matcher = _counting_matcher(ref_a)
    metadata.linked_programs = [decoy1, decoy2, match]

    boxes = [ConstPtr(ref_a)]
    assert metadata.linked_program_for(boxes) is match
    assert len(decoy1.matcher.calls) == 1
    assert len(decoy2.matcher.calls) == 1
    assert len(match.matcher.calls) == 1

    # second lookup, same ref: cache hit, no matcher re-invoked at all
    assert metadata.linked_program_for(boxes) is match
    assert len(decoy1.matcher.calls) == 1
    assert len(decoy2.matcher.calls) == 1
    assert len(match.matcher.calls) == 1


def test_linked_program_for_pc_only_failure_not_cached_as_none():
    ref_a = _new_ref()
    metadata = PEJitCodeMetadata(0, [], [], [], [], [0], [0])
    program = _make_program(ref_index=0, pc_index=1, pcs=[10])
    program.matcher = _counting_matcher(ref_a)
    metadata.linked_programs = [program]

    # ref matches but this call's pc isn't covered -> None, but the ref
    # must still get cached to *this* program, not to "no match".
    boxes_uncovered = [ConstPtr(ref_a), ConstInt(999)]
    assert metadata.linked_program_for(boxes_uncovered) is None
    assert len(program.matcher.calls) == 1

    # same ref, now with a covered pc: must find the program from cache,
    # without re-running matcher.
    boxes_covered = [ConstPtr(ref_a), ConstInt(10)]
    assert metadata.linked_program_for(boxes_covered) is program
    assert len(program.matcher.calls) == 1


def test_linked_program_for_non_matching_ref_caches_none():
    ref_a = _new_ref()
    ref_b = _new_ref()
    metadata = PEJitCodeMetadata(0, [], [], [], [], [0], [0])
    program = _make_program(ref_index=0)
    program.matcher = _counting_matcher(ref_a)
    metadata.linked_programs = [program]

    boxes = [ConstPtr(ref_b)]
    assert metadata.linked_program_for(boxes) is None
    assert len(program.matcher.calls) == 1

    # second lookup, same non-matching ref: cache hit on None, matcher
    # not re-run.
    assert metadata.linked_program_for(boxes) is None
    assert len(program.matcher.calls) == 1


def _counting_runtime_cogen(program_for_ref):
    calls = []
    def runtime_cogen(ref):
        calls.append(ref)
        return program_for_ref(ref)
    runtime_cogen.calls = calls
    return runtime_cogen


def test_runtime_cogen_not_invoked_when_ref_resolved_by_existing_program():
    ref_a = _new_ref()
    metadata = PEJitCodeMetadata(0, [], [], [], [], [0], [0])
    program = _make_program(ref_index=0)
    program.matcher = _counting_matcher(ref_a)
    metadata.linked_programs = [program]
    metadata.runtime_cogen = _counting_runtime_cogen(lambda ref: None)

    boxes = [ConstPtr(ref_a)]
    assert metadata.linked_program_for(boxes) is program
    assert metadata.runtime_cogen.calls == []


def test_runtime_cogen_invoked_once_and_decline_is_cached():
    ref_a = _new_ref()
    metadata = PEJitCodeMetadata(0, [], [], [], [], [0], [0])
    decoy = _make_program(ref_index=0)
    decoy.matcher = _counting_matcher(lltype.nullptr(llmemory.GCREF.TO))
    metadata.linked_programs = [decoy]
    metadata.runtime_cogen = _counting_runtime_cogen(lambda ref: None)

    boxes = [ConstPtr(ref_a)]
    assert metadata.linked_program_for(boxes) is None
    assert len(metadata.runtime_cogen.calls) == 1

    assert metadata.linked_program_for(boxes) is None
    assert len(metadata.runtime_cogen.calls) == 1


def test_runtime_cogen_success_is_cached_and_returned():
    ref_a = _new_ref()
    metadata = PEJitCodeMetadata(0, [], [], [], [], [0], [0])
    metadata.linked_programs = []
    metadata.match_ref_index = 0
    generated = _make_program(ref_index=0)
    generated.match_ref = ref_a

    def program_for_ref(ref):
        metadata.linked_programs = [generated]
        return generated
    metadata.runtime_cogen = _counting_runtime_cogen(program_for_ref)

    boxes = [ConstPtr(ref_a)]
    assert metadata.linked_program_for(boxes) is generated
    assert len(metadata.runtime_cogen.calls) == 1

    assert metadata.linked_program_for(boxes) is generated
    assert len(metadata.runtime_cogen.calls) == 1


def test_runtime_cogen_pc_only_miss_after_generation_still_caches_program():
    ref_a = _new_ref()
    metadata = PEJitCodeMetadata(0, [], [], [], [], [0], [0])
    metadata.linked_programs = []
    metadata.match_ref_index = 0
    generated = _make_program(ref_index=0, pc_index=1, pcs=[10])
    generated.match_ref = ref_a

    def program_for_ref(ref):
        metadata.linked_programs = [generated]
        return generated
    metadata.runtime_cogen = _counting_runtime_cogen(program_for_ref)

    boxes_uncovered = [ConstPtr(ref_a), ConstInt(999)]
    assert metadata.linked_program_for(boxes_uncovered) is None
    assert len(metadata.runtime_cogen.calls) == 1

    boxes_covered = [ConstPtr(ref_a), ConstInt(10)]
    assert metadata.linked_program_for(boxes_covered) is generated
    assert len(metadata.runtime_cogen.calls) == 1


def test_runtime_cogen_returning_wrong_ref_is_treated_as_decline():
    ref_a = _new_ref()
    ref_b = _new_ref()
    metadata = PEJitCodeMetadata(0, [], [], [], [], [0], [0])
    metadata.linked_programs = []
    metadata.match_ref_index = 0
    generated = _make_program(ref_index=0)
    generated.match_ref = ref_b

    metadata.runtime_cogen = _counting_runtime_cogen(lambda ref: generated)

    boxes = [ConstPtr(ref_a)]
    assert metadata.linked_program_for(boxes) is None
    assert len(metadata.runtime_cogen.calls) == 1


def test_soft_decline_is_not_cached_and_retries_with_backoff():
    """A gate-style decline (metadata.soft_decline set before returning
    None) must not be cached, but repeated gate checks are backed off."""
    ref_a = _new_ref()
    metadata = PEJitCodeMetadata(0, [], [], [], [], [0], [0])
    metadata.linked_programs = []
    metadata.match_ref_index = 0
    generated = _make_program(ref_index=0)
    generated.match_ref = ref_a

    calls = []

    def runtime_cogen(ref):
        calls.append(ref)
        if len(calls) < 3:
            metadata.soft_decline = True
            return None
        metadata.soft_decline = False
        metadata.linked_programs = [generated]
        return generated
    metadata.runtime_cogen = runtime_cogen

    boxes = [ConstPtr(ref_a)]
    assert metadata.linked_program_for(boxes) is None
    assert len(calls) == 1
    assert metadata.linked_program_for(boxes) is None
    assert len(calls) == 2
    assert metadata.linked_program_for(boxes) is None
    assert len(calls) == 2
    assert metadata.linked_program_for(boxes) is generated
    assert len(calls) == 3

    # Now cached: a further miss does not call runtime_cogen again.
    assert metadata.linked_program_for(boxes) is generated
    assert len(calls) == 3


def test_soft_decline_backoff_scales_from_cogen_threshold():
    ref_a = _new_ref()
    metadata = PEJitCodeMetadata(0, [], [], [], [], [0], [0])
    metadata.match_ref_index = 0
    metadata.cogen_threshold = 3
    calls = []

    def runtime_cogen(ref):
        calls.append(ref)
        metadata.soft_decline = True
        return None
    metadata.runtime_cogen = runtime_cogen

    boxes = [ConstPtr(ref_a)]
    for expected_calls in [0, 0, 1, 1, 1, 2, 2, 2, 2, 2, 2, 3]:
        assert metadata.linked_program_for(boxes) is None
        assert len(calls) == expected_calls


def test_is_linked_jitcode_uses_flag_set_by_attach():
    metadata = PEJitCodeMetadata(0, [], [], [], [], [0], [0])
    linked = JitCode("linked")
    linked.setup(code='')
    other = JitCode("other")
    other.setup(code='')
    assert not metadata.is_linked_jitcode(linked)
    assert not metadata.is_linked_jitcode(other)
    metadata.attach_linked_jitcode(linked, [], [])
    assert metadata.is_linked_jitcode(linked)
    assert not metadata.is_linked_jitcode(other)


def test_leave_pcs_and_installed_program():
    program = _make_program(0, pc_index=1, pcs=[0, 10])
    program.set_matcher(1, [0, 10], 0, [0], True, [20, 30])
    assert program.is_leave_pc(20) and program.is_leave_pc(30)
    assert not program.is_leave_pc(10)
    ref = _new_ref()
    program.match_ref = ref
    metadata = PEJitCodeMetadata(0, [], [], [], [], [], [])
    metadata.linked_programs.append(program)
    assert metadata.installed_program_for_ref(ref) is program
    assert metadata.installed_program_for_ref(_new_ref()) is None
    assert metadata.installed_program_for_ref(
        lltype.nullptr(llmemory.GCREF.TO)) is None
