from rpython.flowspace.model import checkgraph
from rpython.translator.translator import TranslationContext, graphof
from rpython.rtyper.llinterp import LLInterpreter
from rpython.translator.backendopt.partialeval import PartialEvaluator
from rpython.translator.backendopt.partialeval_template import (
    Continue, Finish)
from rpython.translator.backendopt.generating_extension import (
    GeneratingExtension)


def byte_pair_decoder(code, pc):
    """A decoder for the flat ``<opcode><operand>`` streams used below."""
    opcode = ord(code[pc])
    oparg = ord(code[pc + 1])
    return opcode, {"pc": pc, "oparg": oparg, "code": code}



def get_graph(fn, signature):
    translator = TranslationContext()
    translator.buildannotator().build_types(fn, signature)
    translator.buildrtyper().specialize()
    return graphof(translator, fn), translator


def test_linked_templates_lower_to_one_jitcode_with_real_offsets():
    from rpython.jit.codewriter.codewriter import CodeWriter
    from rpython.jit.codewriter.test.test_codewriter import FakeCPU

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
    graph, translator = get_graph(interpret_one, [int, int, int, int])
    extension = GeneratingExtension.from_step_function(
        translator, interpret_one, [OP_DEC_JUMP, OP_HALT], byte_pair_decoder)

    code = chr(OP_DEC_JUMP) + chr(0) + chr(OP_HALT) + chr(0)
    linked = extension.generate(code)
    codewriter = CodeWriter(FakeCPU(translator.rtyper), [])
    lowered = linked.lower(codewriter, "linked-mini-interpreter")

    checkgraph(lowered.graph)
    result = LLInterpreter(translator.rtyper).eval_graph(
        lowered.graph, [OP_DEC_JUMP, 0, 0, 3])
    assert result.item0 == -1
    assert result.item1 == 0
    assert set(lowered.entry_positions) == set([0, 2])
    assert lowered.entry_positions[0] == 0
    assert lowered.entry_positions[2] > lowered.entry_positions[0]
    assert lowered.jitcode.pe_metadata.entry_positions == [
        lowered.entry_positions[0], lowered.entry_positions[2]]

    dump = lowered.jitcode.dump()
    assert "int_gt" in dump
    assert "int_sub" in dump
    assert "goto" in dump
    assert "strgetitem" not in dump
    assert "int_eq" not in dump
