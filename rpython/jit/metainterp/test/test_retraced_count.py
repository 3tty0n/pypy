from rpython.jit.metainterp.history import JitCellToken


def test_retraced_count_keeps_segmenting_bit_separate():
    token = JitCellToken()
    token.set_retraced_count(1)
    assert token.get_retraced_count() == 1
    assert not (token.retraced_count & token.FORCE_BRIDGE_SEGMENTING)
    token.retraced_count |= token.FORCE_BRIDGE_SEGMENTING
    token.set_retraced_count(3)
    assert token.get_retraced_count() == 3
    assert token.retraced_count & token.FORCE_BRIDGE_SEGMENTING
