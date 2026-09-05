from rpython.jit.codewriter.effectinfo import EffectInfo
from rpython.jit.metainterp.history import ConstInt, ConstPtr, CONST_NULL
from rpython.jit.metainterp.optimizeopt.optimizer import REMOVED, Optimization
from rpython.jit.metainterp.optimizeopt.util import (
    make_dispatcher_method, get_box_replacement)
from rpython.jit.metainterp.resoperation import rop, ResOperation
from rpython.jit.metainterp.optimizeopt.info import (
    AbstractVirtualPtrInfo, getptrinfo)
from rpython.rlib import rtensor
from rpython.rlib.objectmodel import specialize
from rpython.rtyper.lltypesystem import lltype, llmemory

def vtensor_info(box):
    info = getptrinfo(box)
    if isinstance(info, VTensorInfo) and info.is_virtual():
        return info
    return None

class VTensorInfo(AbstractVirtualPtrInfo):

    launched_kernel = lltype.nullptr(rtensor.KERNEL)
    launched_box = None
    node_index = -1

    def __init__(self, opcode, args, param, opt=None):
        self.opcode = opcode
        self.args = args
        self.param = param
        self.opt = opt
        self._is_virtual = True

    def is_virtual(self):
        return self._is_virtual

    def force_box(self, op, optforce):
        if not self._is_virtual:
            return op
        self._is_virtual = False
        if self.launched_kernel:
            return self.force_as_extra_output(op, optforce)
        leaves, opcodes, lefts, rights, params, infos = [], [], [], [], [], []
        _collect_indexed(self, leaves, opcodes, lefts, rights, params, infos)
        n = 0
        cols = 0
        if self.opt is not None:
            n = self.opt.static_size(leaves)
            cols = self.opt.static_cols(self.big_leaf())
        kernel = lltype.malloc(rtensor.KERNEL)
        kernel.ninputs = len(leaves)
        kernel.nodes = lltype.malloc(rtensor.NODEARRAY, len(opcodes))
        kernel.fn = kernel.sumroot = kernel.threads = kernel.shared = kernel.nextra = 0
        kernel.rowmode = 0
        kernel.n = n
        kernel.cols = cols
        kernel.outputs = lltype.malloc(rtensor.SHAPEARRAY, 0)
        for i in range(len(opcodes)):
            node = kernel.nodes[i]
            node.opcode = opcodes[i]
            node.a = lefts[i]
            node.b = rights[i]
            node.p = params[i]
        if self.opt is not None:
            self.opt.pending.append(kernel)
        else:
            rtensor.compile_or_reuse(kernel)
        gcref = lltype.cast_opaque_ptr(llmemory.GCREF, kernel)
        cic = optforce.optimizer.metainterp_sd.callinfocollection
        calldescr, func = cic.callinfo_for_oopspec(EffectInfo.OS_TENSOR_LAUNCH)
        args = [ConstInt(func), ConstPtr(gcref)] + leaves
        while len(args) < 2 + rtensor.MAX_INPUTS:
            args.append(CONST_NULL)
        newop = ResOperation(rop.CALL_R, args, descr=calldescr)
        optforce.emit_extra(newop)
        newop = optforce.optimizer.getlastop()
        op = get_box_replacement(op)
        op.set_forwarded(newop)
        for i in range(len(infos)):
            info = infos[i]
            if info is not self and not rtensor.is_reduction(info.opcode):
                info.launched_kernel = kernel
                info.launched_box = newop
                info.node_index = len(leaves) + i
        return newop

    def force_as_extra_output(self, op, optforce):
        k = rtensor.add_output(self.launched_kernel, self.node_index)
        cic = optforce.optimizer.metainterp_sd.callinfocollection
        calldescr, func = cic.callinfo_for_oopspec(EffectInfo.OS_TENSOR_OUTPUT)
        newop = ResOperation(rop.CALL_R, [ConstInt(func), self.launched_box,
                                          ConstInt(k)], descr=calldescr)
        optforce.emit_extra(newop)
        newop = optforce.optimizer.getlastop()
        op = get_box_replacement(op)
        op.set_forwarded(newop)
        return newop

    def big_arg(self):
        if (self.param == rtensor.BC_L_ROW or
                self.param == rtensor.BC_L_SCALAR or
                self.param == rtensor.BC_L_COL):
            return 1
        return 0

    def size_leaf(self):
        if rtensor.is_reduction(self.opcode):
            return None
        box = self.args[self.big_arg()]
        sub = vtensor_info(box)
        if sub is not None:
            return sub.size_leaf()
        return box

    def big_leaf(self):
        if rtensor.is_reduction(self.opcode):
            box = self.args[0]
        else:
            box = self.args[self.big_arg()]
        sub = vtensor_info(box)
        if sub is not None:
            return sub.big_leaf()
        return box

    def is_scalar(self):
        if rtensor.is_reduction(self.opcode):
            return self.param == rtensor.AXIS_ALL
        sub = vtensor_info(self.args[self.big_arg()])
        if sub is not None:
            return sub.is_scalar()
        return False

    def _visitor_walk_recursive(self, instbox, visitor):
        visitor.register_virtual_fields(instbox, self.args)
        for box in self.args:
            sub = vtensor_info(box)
            if sub is not None:
                sub.visitor_walk_recursive(box, visitor)

    @specialize.argtype(1)
    def visitor_dispatch_virtual_type(self, visitor):
        return visitor.visit_vtensor(self.opcode, self.param)

def _collect_indexed(info, leaves, opcodes, lefts, rights, params, infos):
    _collect_leaves(info, leaves)
    _emit_nodes(info, leaves, opcodes, lefts, rights, params, infos)

def _collect_leaves(info, leaves):
    for box in info.args:
        sub = vtensor_info(box)
        if sub is not None:
            _collect_leaves(sub, leaves)
        elif _leaf_index(leaves, box) < 0:
            leaves.append(box)

def _leaf_index(leaves, box):
    for j in range(len(leaves)):
        if leaves[j].same_box(box):
            return j
    return -1

def _emit_nodes(info, leaves, opcodes, lefts, rights, params, infos):
    idx = [-1, -1]
    for i in range(len(info.args)):
        box = info.args[i]
        sub = vtensor_info(box)
        if sub is not None:
            idx[i] = _emit_nodes(sub, leaves, opcodes, lefts, rights, params,
                                 infos)
        else:
            idx[i] = _leaf_index(leaves, box)
    opcodes.append(info.opcode)
    lefts.append(idx[0])
    rights.append(idx[1])
    params.append(info.param)
    infos.append(info)
    return len(leaves) + len(opcodes) - 1

class OptTensor(Optimization):

    def setup(self):
        self.sizes = {}
        self.cols = {}
        self.pending = []

    def propagate_forward(self, op):
        return dispatch_opt(self, op)

    def flush(self):
        pending = self.pending
        self.pending = []
        for kernel in pending:
            rtensor.compile_or_reuse(kernel)
        self.sizes = {}
        self.cols = {}

    def static_size(self, leaves):
        n = -1
        for leaf in leaves:
            sizebox = self.sizes.get(leaf, None)
            if sizebox is None:
                return 0
            sizebox = get_box_replacement(sizebox)
            if not sizebox.is_constant():
                return 0
            value = sizebox.getint()
            if n != -1 and value != n:
                return 0
            n = value
        if n <= 0:
            return 0
        return n

    def static_cols(self, leaf):
        box = self.cols.get(leaf, None)
        if box is None:
            return 0
        box = get_box_replacement(box)
        if not box.is_constant():
            return 0
        value = box.getint()
        if value <= 0:
            return 0
        return value

    def optimize_CALL_R(self, op):
        effectinfo = op.getdescr().get_extra_info()
        idx = effectinfo.oopspecindex
        if EffectInfo.OS_TENSOR_ADD <= idx <= EffectInfo.OS_TENSOR_EQMASK:
            opcode = idx - EffectInfo.OS_TENSOR_ADD
            nargs = rtensor.ARITY[opcode]
            param = 0
            if rtensor.HAS_PARAM[opcode]:
                pbox = get_box_replacement(op.getarg(1 + nargs))
                if not pbox.is_constant():
                    return self.emit(op)
                param = pbox.getint()
            args = [get_box_replacement(op.getarg(1 + i))
                    for i in range(nargs)]
            for box in args:
                sub = vtensor_info(box)
                if sub is not None and sub.launched_kernel:
                    self.optimizer.force_box(box)
            args = [get_box_replacement(box) for box in args]
            if self._nleaves(args) > rtensor.MAX_INPUTS:
                for box in args:
                    self.optimizer.force_box(box)
            info = VTensorInfo(opcode, args, param, self)
            op = self.replace_op_with(op, op.getopnum())
            op.set_forwarded(info)
            self.last_emitted_operation = REMOVED
            return
        return self.emit(op)
    optimize_CALL_PURE_R = optimize_CALL_R

    def optimize_CALL_I(self, op):
        effectinfo = op.getdescr().get_extra_info()
        idx = effectinfo.oopspecindex
        if (idx == EffectInfo.OS_TENSOR_SIZE or
                idx == EffectInfo.OS_TENSOR_SHAPE or
                idx == EffectInfo.OS_TENSOR_NDIM):
            info = vtensor_info(op.getarg(1))
            if info is not None:
                leaf = info.size_leaf()
                if leaf is None:
                    if not info.is_scalar():
                        return self.emit(op)
                    self.make_constant(op, ConstInt(1))
                    self.last_emitted_operation = REMOVED
                    return
                args = [op.getarg(0), leaf]
                if idx == EffectInfo.OS_TENSOR_SHAPE:
                    args.append(op.getarg(2))
                op = self.replace_op_with(op, op.getopnum(), args=args)
            if idx == EffectInfo.OS_TENSOR_SIZE:
                self.sizes[get_box_replacement(op.getarg(1))] = op
            elif idx == EffectInfo.OS_TENSOR_SHAPE:
                axis = get_box_replacement(op.getarg(2))
                if axis.is_constant() and axis.getint() == 1:
                    self.cols[get_box_replacement(op.getarg(1))] = op
        return self.emit(op)
    optimize_CALL_PURE_I = optimize_CALL_I

    def _nleaves(self, args):
        leaves = []
        _collect_leaves(VTensorInfo(0, args, 0), leaves)
        return len(leaves)

    def optimize_GUARD_NO_EXCEPTION(self, op):
        if self.last_emitted_operation is REMOVED:
            return
        return self.emit(op)

dispatch_opt = make_dispatcher_method(OptTensor, 'optimize_',
                                      default=OptTensor.emit)
