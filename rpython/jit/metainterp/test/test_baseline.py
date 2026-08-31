import py
from rpython.jit.metainterp.baseline import BaselineTier


def test_run_from_pyjitpl_not_implemented():
    tier = BaselineTier()
    py.test.raises(NotImplementedError, tier.run_from_pyjitpl,
                    None, 0, None)
