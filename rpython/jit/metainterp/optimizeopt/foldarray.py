from rpython.jit.metainterp.history import Const, ConstPtr, ConstInt, \
    TargetToken, JitCellToken, AbstractValue, AbstractFailDescr
from rpython.jit.metainterp.optimize import InvalidLoop, SpeculativeError
from rpython.jit.metainterp.optimizeopt import info, intutils
from rpython.jit.metainterp.optimizeopt.dependency import DependencyGraph, IntegralForwardModification
from rpython.jit.metainterp.optimizeopt.vector import VectorLoop
from rpython.jit.metainterp.optimizeopt.optimizer import Optimizer,\
     BasicLoopInfo, Optimization, LoopInfo, MININT, MAXINT, BasicLoopInfo, PASS_OP_ON
from .util import get_box_replacement, make_dispatcher_method, have_dispatcher_method, get_box_replacement
from rpython.jit.metainterp.resoperation import rop, ResOperation, GuardResOp
from rpython.jit.metainterp import compile
from rpython.rlib.debug import debug_print, debug_start, debug_stop,\
     have_debug_prints
from rpython.rlib.objectmodel import r_dict

import py
import time

class MemoryRef(object):
    def __init__(self, op, index_var, raw_access=False, saved_item=None):
        self.op = op
        self.array = op.getarg(0)
        self.descr = op.getdescr()
        self.index_var = index_var
        self.raw_access = raw_access
        self.saved_item = saved_item

    def get_saved_item(self):
        if self.is_setarrayitem_gc():
            return self.saved_item
        return None

    def is_setarrayitem_gc(self):
        return self.op.opnum == rop.SETARRAYITEM_GC

    def is_getarrayitem(self):
        return rop.is_getarrayitem(self.op.opnum)

    def stride(self):
        """ the stride in bytes """
        if not self.raw_access:
            return 1
        return self.descr.get_item_size_in_bytes()

    def same_array(self, other):
        return self.array is other.array and self.descr == other.descr

    def alias(self, other):
        """ is this reference an alias to other?
            they can alias iff self.origin != other.origin, or their
            linear combination point to the same element.
        """
        assert other is not None
        if not self.same_array(other):
            return False
        svar = self.index_var
        ovar = other.index_var
        if not svar.same_variable(ovar):
            return True
        if not svar.same_mulfactor(ovar):
            return True
        return abs(svar.constant_diff(ovar)) < self.stride()

    def __repr__(self):
        return "MemoryRef(%s)" % (self.index_var)

class IndexVar(AbstractValue):
    def __init__(self, var, coeff_mul=1, coeff_div=1, constant=0):
        self.var = var
        self.constant = constant
        # saves the next modification that uses a variable
        self.next_nonconst = None
        self.current_end = None

        self.coefficient_mul = coeff_mul
        self.coefficient_div = coeff_div

    def is_const(self):
        return self.constant == 0

    def clone(self):
        c = IndexVar(self.var)
        c.constant = self.constant
        return c

    def add_const(self, number):
        self.var.value += number

    def __eq__(self, other):
        return self.var is other.var

    def __repr__(self):
        return "IndexVar(%s)" % (self.var)

class Node(object):
    def __init__(self, op, opidx):
        self.op = op
        self.opidx = opidx
        self.adjacent = []
        self.adjacent_back = []
        self.memory_ref = None
        self.index_var = None

    def is_memory_ref(self):
        return self.memory_ref is not None

    def is_index_var(self):
        return self.index_var is not None

    def edge_to(self, to, back=False):
        if self is to:
            return
        if back:
            self.adjacent_back.append(to)
        else:
            self.adjacent.append(to)

    def __repr__(self):
        return "Node(%s, %s)" % (self.op, self.memory_ref)

class DefTracker(object):
    def __init__(self, graph):
        self.graph = graph
        self.defs = {}
        self.non_pure = []

    def add_non_pure(self, node):
        self.non_pure.append(node)

    def define(self, arg, node, argcell=None):
        if isinstance(arg, Const):
            return
        if arg in self.defs:
            self.defs[arg].append((node, argcell))
        else:
            self.defs[arg] = [(node, argcell)]

    def definition(self, arg, node=None, argcell=None):
        if arg.is_constant():
            return None
        def_chain = self.defs.get(arg,None)
        if not def_chain:
            return None
        if not argcell:
            return def_chain[-1][0]
        else:
            assert node is not None
            i = len(def_chain) - 1
            try:
                mref = node.memory_ref
                while i >= 0:
                    def_node = def_chain[i][0]
                    oref = def_node.memory_ref
                    if oref is not None and mref.alias(oref):
                        return def_node
                    elif oref is None:
                        return def_node
                    i -= 1
                return None
            except KeyError:
                # when a key error is raised, this means
                # no information is available, safe default
                pass
            return def_chain[-1][0]

    def depends_on_arg(self, arg, to, argcell=None):
        try:
            at = self.definition(arg, to, argcell)
            if at is None:
                return
            at.edge_to(to, arg)
        except KeyError:
            pass

class DependencyGraph(object):
    def __init__(self, oplist):
        self.oplist = oplist
        self.nodes = [ Node(op, 0) for op in self.oplist if not rop.is_jit_debug(op) ]
        for i, node in enumerate(self.nodes):
            node.opidx = i + 1

        self.memory_refs = {}
        self.index_vars = {}
        self.comparison_vars = {}
        self.invariant_vars = {}

        self.build_dependency()

    def build_dependency(self):
        tracker = DefTracker(self)
        foldmod = FoldingModification(self.memory_refs, self.index_vars,
                                      self.comparison_vars, self.invariant_vars)
        for i, node in enumerate(self.nodes):
            op = node.op
            foldmod.inspect_operation(op, node)

            # definition of a new variable
            if op.type != 'v':
                tracker.define(op, node)
            # usege of defined variables
            if rop.is_always_pure(op.opnum) or rop.is_final(op.opnum):
                for arg in op.getarglist():
                    tracker.depends_on_arg(arg, node)
            elif rop.is_guard(op.opnum):
                pass

    def __repr__(self):
        graph = "graph([\n"
        for node in self.nodes:
            graph += "       " + str(node.opidx) + ": "
            # for dep in node.provides():
            #     graph += "=>" + str(dep.to.opidx) + ","
            graph += " | "
            # for dep in node.depends():
            #     graph += "<=" + str(dep.to.opidx) + ","
            graph += "\n"
        return graph + "      ])"

class FoldingModification(object):
    def __init__(self, memory_refs, index_vars, comparison_vars, invariant_vars):
        self.index_vars = index_vars
        self.comparison_vars = comparison_vars
        self.memory_refs = memory_refs
        self.invariant_vars = invariant_vars

    def is_const_integral(self, box):
        if isinstance(box, ConstInt):
            return True
        return False

    def get_or_create(self, arg):
        var = self.index_vars.get(arg, None)
        if not var:
            var = self.index_vars[arg] = IndexVar(arg)
        return var

    additive_func_source = """
    def operation_{name}(self, op, node):
        box_r = op
        box_a0 = op.getarg(0)
        box_a1 = op.getarg(1)
        if self.is_const_integral(box_a0) and self.is_const_integral(box_a1):
            idx_ref = IndexVar(box_r)
            idx_ref.constant = box_a0.getint() {op} box_a1.getint()
            self.index_vars[box_r] = idx_ref
        elif self.is_const_integral(box_a0):
            idx_ref = self.get_or_create(box_a1)
            idx_ref = idx_ref.clone()
            idx_ref.constant {op}= box_a0.getint()
            self.index_vars[box_r] = idx_ref
        elif self.is_const_integral(box_a1):
            idx_ref = self.get_or_create(box_a0)
            idx_ref = idx_ref.clone()
            idx_ref.constant {op}= box_a1.getint()
            self.index_vars[box_r] = idx_ref
    """
    exec(py.code.Source(additive_func_source
            .format(name='INT_ADD', op='+')).compile())
    exec(py.code.Source(additive_func_source
            .format(name='INT_SUB', op='-')).compile())
    del additive_func_source

    multiplicative_func_source = """
    def operation_{name}(self, op, node):
        box_r = op
        if not box_r:
            return
        box_a0 = op.getarg(0)
        box_a1 = op.getarg(1)
        if self.is_const_integral(box_a0) and self.is_const_integral(box_a1):
            idx_ref = IndexVar(box_r)
            idx_ref.constant = box_a0.getint() {cop} box_a1.getint()
            self.index_vars[box_r] = idx_ref
        elif self.is_const_integral(box_a0):
            idx_ref = self.get_or_create(box_a1)
            idx_ref = idx_ref.clone()
            idx_ref.coefficient_{tgt} *= box_a0.getint()
            idx_ref.constant {cop}= box_a0.getint()
            self.index_vars[box_r] = idx_ref
        elif self.is_const_integral(box_a1):
            idx_ref = self.get_or_create(box_a0)
            idx_ref = idx_ref.clone()
            idx_ref.coefficient_{tgt} {op}= box_a1.getint()
            idx_ref.constant {cop}= box_a1.getint()
            self.index_vars[box_r] = idx_ref
    """
    exec(py.code.Source(multiplicative_func_source
            .format(name='INT_MUL', op='*', tgt='mul', cop='*')).compile())
    del multiplicative_func_source

    array_access_source = """
    def operation_{name}(self, op, node):
        descr = op.getdescr()
        idx_ref = self.get_or_create(op.getarg(1))
        if descr and descr.is_array_of_primitives():
            node.memory_ref = MemoryRef(op, idx_ref, {raw_access})
            self.memory_refs[node] = node.memory_ref
    """
    exec(py.code.Source(array_access_source
           .format(name='RAW_LOAD_I',raw_access=True)).compile())
    exec(py.code.Source(array_access_source
           .format(name='RAW_LOAD_F',raw_access=True)).compile())
    exec(py.code.Source(array_access_source
           .format(name='RAW_STORE',raw_access=True)).compile())
    exec(py.code.Source(array_access_source
           .format(name='GETARRAYITEM_RAW_I',raw_access=False)).compile())
    exec(py.code.Source(array_access_source
           .format(name='GETARRAYITEM_RAW_F',raw_access=False)).compile())
    exec(py.code.Source(array_access_source
           .format(name='SETARRAYITEM_RAW',raw_access=False)).compile())
    exec(py.code.Source(array_access_source
           .format(name='GETARRAYITEM_GC_I',raw_access=False)).compile())
    exec(py.code.Source(array_access_source
           .format(name='GETARRAYITEM_GC_F',raw_access=False)).compile())
    exec(py.code.Source(array_access_source
           .format(name='GETARRAYITEM_GC_R',raw_access=False)).compile())
    exec(py.code.Source(array_access_source
           .format(name='SETARRAYITEM_GC',raw_access=False)).compile())
    del array_access_source
foldarray_dispatch_opt = make_dispatcher_method(FoldingModification, 'operation_')
FoldingModification.inspect_operation = foldarray_dispatch_opt
del foldarray_dispatch_opt

def optimize_foldarray(trace, metainterp_sd, jitdriver_sd, optimizations,
                       runtime_boxes=None, call_pure_results=None):
    debug_start("foldarray-opt-loop")
    start = time.clock()
    opt = FoldArrayOptimizer(metainterp_sd, jitdriver_sd, optimizations)
    info, oplist = opt.optimize(trace, runtime_boxes, call_pure_results)
    end = time.clock()

    graph = DependencyGraph(oplist)
    oplist = foldarray(graph)
    import pdb; pdb.set_trace()
    return info, oplist

def foldarray(graph):
    memory_refs = graph.memory_refs
    memory_usage = {}
    ops_foldable = {}
    for i, node in enumerate(memory_refs):
        mem_ref = node.memory_ref
        if mem_ref.is_setarrayitem_gc():
            index_var = mem_ref.index_var
            op = node.op
            a0 = op.getarg(0)
            a2 = op.getarg(2)
            memory_usage[a0] = { index_var: { index_var.constant: a2 } }
        elif mem_ref.is_getarrayitem():
            index_var = mem_ref.index_var
            op = node.op
            a0 = op.getarg(0)
            if a0 in memory_usage:
                if index_var in memory_usage[a0]:
                    # TODO: mark a removable setarray op
                    memref_keys_backward = memory_refs.keys()[:i][:]
                    memref_vals_backward = memory_refs.values()[:i][:]

                    j = 1
                    try:
                        while memref_vals_backward.pop().index_var is not index_var:
                            j += 1
                        foldable_setarrayitem = memref_keys_backward[-j].op
                    except IndexError:
                        foldable_setarrayitem = None
                    ops_foldable[op] = (memory_usage[a0][index_var][index_var.constant],
                                        foldable_setarrayitem)

    if len(ops_foldable) == 0:
        return graph.oplist

    oplist = graph.oplist
    ops_result = []
    for op in oplist:
        if op in ops_foldable:
            op_folded, op_removable = ops_foldable[op]
            if op_removable in ops_result:
                ops_result.remove(op_removable)
            op.set_forwarded(op_folded)
            continue
        for i, arg in enumerate(op.getarglist()):
            if not isinstance(arg, Const) and arg._forwarded:
                arg = arg._forwarded
                op.setarg(i, arg)
        ops_result.append(op)
    return ops_result

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

        # newoperations = []
        # # remove folded setarrayitem
        # for op in self._newoperations:
        #     if not op in self.candidates:
        #         newoperations.append(op)

        # newoperations = self._remove_unreferenced_ops(
        #     newoperations, trace.inputargs, last_op)
        # self._newoperations = newoperations

        return (BasicLoopInfo(trace.inputargs, self.quasi_immutable_deps, last_op),
                self._newoperations)

    def _remove_unreferenced_ops(self, ops, inputargs, last_op):
        # build arg list
        uses = {}
        for inputarg in inputargs:
            uses[inputarg] = None

        if last_op:
            ops.append(last_op)

        for op in ops:
            for arg in op.getarglist():
                if not arg.is_constant():
                    uses[arg] = None
        # remove unused op
        opt_results = []
        for op in ops:
            if op in uses and op not in opt_results:
                opt_results.append(op)

        if last_op:
            opt_results.append(last_op)
        return opt_results


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
    if key1.is_constant():
        return key1.same_constant(key2)
    else:
        return key1 == key2

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
