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

    def __init__(self, opcode, args):
        self.opcode = opcode
        self.args = args
        self._is_virtual = True

    def is_virtual(self):
        return self._is_virtual

    def force_box(self, op, optforce):
        if not self._is_virtual:
            return op
        self._is_virtual = False
        leaves, opcodes, lefts, rights = [], [], [], []
        _collect_indexed(self, leaves, opcodes, lefts, rights)
        parts = [str(len(leaves))]
        for i in range(len(opcodes)):
            parts.append('%d:%d:%d' % (opcodes[i], lefts[i], rights[i]))
        key = ','.join(parts)
        kernel = rtensor.cached_kernel(key)
        if not kernel:
            kernel = lltype.malloc(rtensor.KERNEL)
            kernel.ninputs = len(leaves)
            kernel.nodes = lltype.malloc(rtensor.NODEARRAY, len(opcodes))
            kernel.fn = kernel.sumroot = kernel.threads = kernel.shared = kernel.nextra = 0
            for i in range(len(opcodes)):
                node = kernel.nodes[i]
                node.opcode = opcodes[i]
                node.a = lefts[i]
                node.b = rights[i]
            rtensor.finish_kernel(kernel)
            rtensor.cache_kernel(key, kernel)
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
        return newop

    def size_leaf(self):
        if self.opcode == rtensor.SUM:
            return None
        box = self.args[0]
        sub = vtensor_info(box)
        if sub is not None:
            return sub.size_leaf()
        return box

    def _visitor_walk_recursive(self, instbox, visitor):
        visitor.register_virtual_fields(instbox, self.args)
        for box in self.args:
            sub = vtensor_info(box)
            if sub is not None:
                sub.visitor_walk_recursive(box, visitor)

    @specialize.argtype(1)
    def visitor_dispatch_virtual_type(self, visitor):
        return visitor.visit_vtensor(self.opcode)

def _collect_indexed(info, leaves, opcodes, lefts, rights):
    _collect_leaves(info, leaves)
    _emit_nodes(info, leaves, opcodes, lefts, rights)

def _collect_leaves(info, leaves):
    for box in info.args:
        sub = vtensor_info(box)
        if sub is not None:
            _collect_leaves(sub, leaves)
        elif not _contains(leaves, box):
            leaves.append(box)

def _contains(leaves, box):
    for leaf in leaves:
        if leaf is box:
            return True
    return False

def _emit_nodes(info, leaves, opcodes, lefts, rights):
    idx = [-1, -1]
    for i in range(len(info.args)):
        box = info.args[i]
        sub = vtensor_info(box)
        if sub is not None:
            idx[i] = _emit_nodes(sub, leaves, opcodes, lefts, rights)
        else:
            for j in range(len(leaves)):
                if leaves[j] is box:
                    idx[i] = j
    opcodes.append(info.opcode)
    lefts.append(idx[0])
    rights.append(idx[1])
    return len(leaves) + len(opcodes) - 1

class OptTensor(Optimization):
    "Keep rtensor ops virtual and fuse them into one kernel launch."

    def propagate_forward(self, op):
        return dispatch_opt(self, op)

    def optimize_CALL_R(self, op):
        effectinfo = op.getdescr().get_extra_info()
        idx = effectinfo.oopspecindex
        if EffectInfo.OS_TENSOR_ADD <= idx <= EffectInfo.OS_TENSOR_SUM:
            opcode = idx - EffectInfo.OS_TENSOR_ADD
            args = [get_box_replacement(op.getarg(i))
                    for i in range(1, op.numargs())]
            if self._nleaves(args) > rtensor.MAX_INPUTS:
                for box in args:
                    self.optimizer.force_box(box)
            info = VTensorInfo(opcode, args)
            op = self.replace_op_with(op, op.getopnum())
            op.set_forwarded(info)
            self.last_emitted_operation = REMOVED
            return
        return self.emit(op)
    optimize_CALL_PURE_R = optimize_CALL_R

    def optimize_CALL_I(self, op):
        effectinfo = op.getdescr().get_extra_info()
        if effectinfo.oopspecindex == EffectInfo.OS_TENSOR_SIZE:
            info = vtensor_info(op.getarg(1))
            if info is not None:
                leaf = info.size_leaf()
                if leaf is None:
                    self.make_constant(op, ConstInt(1))
                    self.last_emitted_operation = REMOVED
                    return
                op = self.replace_op_with(op, op.getopnum(),
                                          args=[op.getarg(0), leaf])
        return self.emit(op)
    optimize_CALL_PURE_I = optimize_CALL_I

    def _nleaves(self, args):
        leaves = []
        _collect_leaves(VTensorInfo(0, args), leaves)
        return len(leaves)

    def optimize_GUARD_NO_EXCEPTION(self, op):
        if self.last_emitted_operation is REMOVED:
            return
        return self.emit(op)

dispatch_opt = make_dispatcher_method(OptTensor, 'optimize_',
                                      default=OptTensor.emit)
