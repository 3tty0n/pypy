from rpython.rtyper.lltypesystem.llmemory import AddressAsInt
from rpython.rlib.rjitlog import rjitlog as jl
from rpython.jit.metainterp.history import ConstInt
from rpython.jit.metainterp.optimizeopt.optimizer import Optimization
from rpython.jit.metainterp.optimizeopt.util import make_dispatcher_method
from rpython.jit.metainterp.opencoder import Trace, TraceIterator
from rpython.jit.metainterp.resoperation import rop, OpHelpers

class OptTraceSplit(Optimization):
    def emit(self, op):
        return Optimization.emit(self, op)

    def optimize_CALL_I(self, op):
        arg0 = op.getarg(0)
        if isinstance(arg0, ConstInt):
            if jl.int_could_be_an_address(arg0.value):
                metainterp = self.optimizer.metainterp_sd
                addr = arg0.getaddr()
                name = metainterp.get_name_from_address(addr)
                if name.find('cut_here') != -1:
                    return None

        return self.emit(op)


dispatch_opt = make_dispatcher_method(OptTraceSplit, 'optimize_',
                                      default=OptTraceSplit.emit)
OptTraceSplit.propagate_forward = dispatch_opt
