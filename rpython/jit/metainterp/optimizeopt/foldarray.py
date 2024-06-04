from rpython.jit.metainterp.history import Const, TargetToken, JitCellToken
from rpython.jit.metainterp.optimize import InvalidLoop, SpeculativeError
from rpython.jit.metainterp.optimizeopt import info, intutils
from rpython.jit.metainterp.optimizeopt.dependency import DependencyGraph
from rpython.jit.metainterp.optimizeopt.optimizer import Optimizer,\
     BasicLoopInfo, Optimization, LoopInfo, MININT, MAXINT, BasicLoopInfo, PASS_OP_ON
from .util import get_box_replacement, make_dispatcher_method, have_dispatcher_method, get_box_replacement
from rpython.jit.metainterp.resoperation import rop, ResOperation, GuardResOp
from rpython.jit.metainterp import compile
from rpython.rlib.debug import debug_print, debug_start, debug_stop,\
     have_debug_prints
from rpython.rlib.objectmodel import r_dict

class FoldArrayOptimizer(Optimizer):
    def __init__(self, metainterp_sd, jitdriver_sd, optimizations):
        Optimizer.__init__(self, metainterp_sd, jitdriver_sd, optimizations)
        self.optfoldarray = OptFoldArray()
        self.optfoldarray.optimizer = self
        self.candidates = {}

    def optimize(self, trace, runtime_boxes, call_pure_results):
        info, newops = self.propagate_all_forward(
            trace.get_iter(), call_pure_results, flush=True)

        return info, newops

    def propagate_all_forward(self, trace, call_pure_results=None, flush=True):
        self.trace = trace
        deadranges = trace.get_dead_ranges()
        self.call_pure_results = call_pure_results
        last_op = None
        i = 0
        candidate = {}
        while not trace.done():
            self._really_emitted_operation = None
            op = trace.next()
            if op.getopnum() in (rop.FINISH, rop.JUMP):
                last_op = op
                break
            self.send_extra_operation(op)
            trace.kill_cache_at(deadranges[i + trace.start_index])
            if op.type != 'v':
                i += 1

        # accumulate counters
        if flush:
            self.flush()
            if last_op:
                self.send_extra_operation(last_op)
        self.resumedata_memo.update_counters(self.metainterp_sd.profiler)

        newoperations = []
        # remove folded setarrayitem
        for op in self._newoperations:
            if not op in self.candidates:
                newoperations.append(op)

        newoperations = self._remove_unreferenced_ops(
            newoperations, trace.inputargs, last_op)
        self._newoperations = newoperations

        return (BasicLoopInfo(trace.inputargs, self.quasi_immutable_deps, last_op),
                self._newoperations)

    def _remove_unreferenced_ops(self, ops, inputargs, last_op):
        # build arg list
        uses = {}
        for inputarg in inputargs:
            uses[inputarg] = None
        for op in ops + [last_op]:
            for arg in op.getarglist():
                if not arg.is_constant():
                    uses[arg] = None
        # remove unused op
        opt_results = []
        for op in ops:
            if op in uses and op not in opt_results:
                opt_results.append(op)
        return opt_results + [last_op]


    def send_extra_operation(self, op, opt=None):
        if opt is None:
            opt = self.first_optimization
        opt_results = None
        while opt is not None:
            if isinstance(opt, OptFoldArray):
                opt_result = opt.propagate_forward(op, self.candidates)
            else:
                opt_result = opt.propagate_forward(op)
            if opt_result is None:
                op = None
                break
            if opt_result is not PASS_OP_ON:
                if opt_results is None:
                    opt_results = [opt_result]
                else:
                    opt_results.append(opt_result)
                op = opt_result.op
            else:
                op = opt.last_emitted_operation
            opt = opt.next_optimization
        if opt_results is not None:
            index = len(opt_results) - 1
            while index >= 0:
                opt_results[index].callback()
                index -= 1

    def _remove_unused_ops(self, ops, allargs):
        opt_results = []
        for op in ops:
            import pdb; pdb.set_trace()
            if op in allargs:
                opt_results.append(op)
        return opt_results

def key_eq(key1, key2):
    return key1.same_constant(key2)

def key_hash(key):
    return key._get_hash_()

class OptFoldArray(Optimization):
    def __init__(self):
        self.defs = {}
        self.uses = {}

    def setup(self):
        Optimization.setup(self)

    def flush(self):
        Optimization.flush(self)

    def get_candidate(self):
        return self.candidates

    def emit(self, op, *args):
        return Optimization.emit(self, op)

    def propagate_forward(self, op, *args):
        return dispatch_opt(self, op, *args)

    def optimize_GETARRAYITEM_GC_R(self, op, *args):
        candidate = None
        if len(args) >= 1:
            candidate = args[-1]
        arg0 = op.getarg(0)
        arg1 = op.getarg(1)
        if arg0 in self.uses:
            if arg1 in self.uses[arg0]:
                newop = self.uses[arg0][arg1]
                op.set_forwarded(newop)
                candidate[self.defs[newop]] = None

        self.defs[op] = None
        return self.emit(op)

    def optimize_SETARRAYITEM_GC(self, op, *arg):
        arg2 = op.getarg(2)
        if arg2 in self.defs:
            arg0 = op.getarg(0)
            arg1 = op.getarg(1)
            new_dic = r_dict(key_eq, key_hash)
            new_dic[arg1] = arg2
            self.uses[arg0] = new_dic
            self.defs[arg2] = op

        return self.emit(op)


dispatch_opt = make_dispatcher_method(OptFoldArray, 'optimize_',
                                      default=OptFoldArray.emit)
