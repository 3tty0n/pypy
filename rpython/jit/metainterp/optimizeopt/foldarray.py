from rpython.jit.metainterp.history import Const, TargetToken, JitCellToken
from rpython.jit.metainterp.optimize import InvalidLoop, SpeculativeError
from rpython.jit.metainterp.optimizeopt.shortpreamble import ShortBoxes,\
     ShortPreambleBuilder, ExtendedShortPreambleBuilder, PreambleOp
from rpython.jit.metainterp.optimizeopt import info, intutils
from rpython.jit.metainterp.optimizeopt.optimizer import Optimizer,\
     Optimization, LoopInfo, MININT, MAXINT, BasicLoopInfo, PASS_OP_ON
from rpython.jit.metainterp.optimizeopt.vstring import StrPtrInfo
from rpython.jit.metainterp.optimizeopt.virtualstate import (
    VirtualStateConstructor, VirtualStatesCantMatch)
from .util import get_box_replacement
from rpython.jit.metainterp.optimizeopt.bridgeopt import (
    deserialize_optimizer_knowledge)
from rpython.jit.metainterp.optimizeopt.util import (
    make_dispatcher_method, have_dispatcher_method, get_box_replacement)
from rpython.jit.metainterp.resoperation import rop, ResOperation, GuardResOp
from rpython.jit.metainterp import compile
from rpython.rlib.debug import debug_print, debug_start, debug_stop,\
     have_debug_prints
from rpython.rlib.objectmodel import r_dict


class BasicLoopInfo(LoopInfo):
    def __init__(self, inputargs, quasi_immutable_deps, jump_op):
        self.inputargs = inputargs
        self.jump_op = jump_op
        self.quasi_immutable_deps = quasi_immutable_deps
        self.extra_same_as = []
        self.extra_before_label = []

    def final(self):
        return True

    def post_loop_compilation(self, loop, jitdriver_sd, metainterp, jitcell_token):
        pass

def key_eq(key1, key2):
    return key1.same_constant(key2)

def key_hash(key):
    return key._get_hash_()

class OptFoldAarray(Optimization):
    def __init__(self):
        self.defs = {}
        self.uses = {}

    def propagate_forward(self, op):
        return dispatch_opt(self, op)

    def optimize_GETARRAYITEM_GC_R(self, op):
        arg0 = op.getarg(0)
        arg1 = op.getarg(1)
        if arg0 in self.uses:
            if arg1 in self.uses[arg0]:
                print "can be replaced to", self.uses[arg0][arg1]
                newop = self.uses[arg0][arg1]
                op.set_forwarded(newop)

        self.defs[op] = None
        return self.emit(op)

    def optimize_SETARRAYITEM_GC(self, op):
        arg2 = op.getarg(2)
        if arg2 in self.defs:
            arg0 = op.getarg(0)
            arg1 = op.getarg(1)
            new_dic = r_dict(key_eq, key_hash)
            new_dic[arg1] = arg2
            self.uses[arg0] = new_dic
        return self.emit(op)

    def postprocess_GETARRAYITEM_GC_R(self, op):
        self.emit(op)

    def postprocess_SETARRAYITEM_GC(self, op):
        self.emit(op)

dispatch_opt = make_dispatcher_method(OptFoldAarray, 'optimize_',
                                      default=Optimizer.optimize_default)
OptFoldAarray.propagate_postprocess = make_dispatcher_method(OptFoldAarray, 'postprocess_')
OptFoldAarray.have_postprocess_op = have_dispatcher_method(OptFoldAarray, 'postprocess_')
