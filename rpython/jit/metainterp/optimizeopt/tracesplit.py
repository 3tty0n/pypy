from rpython.jit.metainterp.optimize import InvalidLoop
from rpython.rlib.debug import debug_print
from rpython.rtyper.lltypesystem.llmemory import AddressAsInt, cast_int_to_adr
from rpython.rlib.rjitlog import rjitlog as jl
from rpython.rlib.rstring import find, startswith, endswith
from rpython.rlib.objectmodel import specialize, we_are_translated, r_dict
from rpython.jit.codewriter.effectinfo import EffectInfo
from rpython.jit.metainterp.history import (
    AbstractFailDescr, ConstInt, ConstFloat, RefFrontendOp, IntFrontendOp, FloatFrontendOp,
    INT, REF, FLOAT, VOID)
from rpython.jit.metainterp import compile, jitprof, history
from rpython.jit.metainterp.history import TargetToken
from rpython.jit.metainterp.optimizeopt.optimizer import (
    Optimizer, Optimization, BasicLoopInfo)
from rpython.jit.metainterp.optimizeopt.intutils import (
    IntBound, ConstIntBound, MININT, MAXINT, IntUnbounded)
from rpython.jit.metainterp.optimizeopt.bridgeopt import (
    deserialize_optimizer_knowledge)
from rpython.jit.metainterp.optimizeopt.util import make_dispatcher_method
from rpython.jit.metainterp.opencoder import Trace, TraceIterator
from rpython.jit.metainterp.resoperation import (
    rop, OpHelpers, ResOperation, InputArgRef, InputArgInt,
    InputArgFloat, InputArgVector, GuardResOp)

class TokenMapError(Exception):
    """Raised when KeyError happens at taking a TargetToken from token_map"""
    def __init__(self, key=None,
                 message="KeyError happens when taking token from token_map"):
        self.key = key
        self.message = message
        if key is not None:
            self.message = "%s, key is %d" % (message, key)

class mark(object):
    CALL_ASSEMBLER = "call_assembler"

class TraceSplitInfo(BasicLoopInfo):
    """ A state after splitting the trace, containing the following:

    * target_token - generated target token for a bridge ("false" branch)
    * label_op - label operations
    * inputargs - input arguments
    * faildescr - used in the case of a bridge trace; for attaching
    """
    def __init__(self, target_token, label_op, inputargs, faildescr=None):
        self.target_token = target_token
        self.label_op = label_op
        self.inputargs = inputargs
        self.faildescr = faildescr

    def final(self):
        return True

    def __copy__(self, target_token, label_op, inputargs, faildescr=None):
        return TraceSplitInfo(target_token, label_op, inputargs, faildescr)

    def set_token(self, target_token):
        self.target_token = target_token

    def set_label(self, label_op):
        self.label_op = label_op

    def set_inputargs(self, inputargs):
        self.inputargs = inputargs

    def set_faildescr(self, faildescr):
        self.faildescr = faildescr

class OptTraceSplit(Optimizer):

    def __init__(self, metainterp_sd, jitdriver_sd,
                 optimizations=None, resumekey=None):
        Optimizer.__init__(self, metainterp_sd, jitdriver_sd)
        self.metainterp_sd = metainterp_sd
        self.jitdriver_sd = jitdriver_sd
        self.trace = None
        self.optimizations = optimizations
        self.resumekey = resumekey

        self.inputargs = None
        self.token = None
        self.token_map = {}

        self.conditions = jitdriver_sd.jitdriver.conditions

        self._already_setup_current_token = False
        self._pseudoops = []
        self._specialguardop = []
        self._newopsandinfo = []
        self._fdescrstack = []

        self._newoperations_slow_path = []

        self._slow_ops = []
        self._in_slow_path = False
        self._slow_path_newopsandinfo = []
        self._slow_path_emit_ptr_eq = None
        self._slow_path_faildescr = None

        self.set_optimizations(optimizations)
        self.setup()

    def split(self, trace, resumestorage, call_pure_results, token):
        traceiter = trace.get_iter()
        self.token = token
        self.propagate_all_forward(traceiter, call_pure_results)
        return self._newopsandinfo

    def propagate_all_forward(self, trace, call_pure_results=None, flush=True):
        self.trace = trace
        deadranges = trace.get_dead_ranges()
        self.inputargs = trace.inputargs
        self.call_pure_results = call_pure_results
        last_op = None
        i = 0

        jd = self.jitdriver_sd
        num_green_args = jd.num_green_args
        num_red_args = jd.num_red_args

        slow_path_jump_op = None
        slow_path_label = None
        while not trace.done():
            self._really_emitted_operation = None
            op = trace.next()
            opnum = op.getopnum()
            numargs = op.numargs()

            # remove op related to pseudo ops
            can_emit = True
            for arg in op.getarglist():
                if arg in self._pseudoops:
                    can_emit = False
                    self.emit_pseudoop(op)
                    break

            if not can_emit:
                continue

            if not self._already_setup_current_token and \
               opnum == rop.DEBUG_MERGE_POINT:
                arglist = op.getarglist()
                # TODO: look up `pc' by name
                greens = arglist[1+num_red_args:1+num_red_args+num_green_args]
                if len(greens) > 0:
                    box = greens[0]
                    assert isinstance(box, ConstInt)
                    token = self._create_token()
                    self.token_map[box.getint()] = token
                    self.emit(ResOperation(rop.LABEL, self.inputargs, token))
                    self._already_setup_current_token = True

            if opnum in (rop.FINISH, rop.JUMP):
                last_op = op
                break

            # shallow tracing: turn on flags
            if rop.is_call(opnum):
                numargs = op.numargs()
                lastarg = op.getarg(numargs - 1)
                if isinstance(lastarg, ConstInt) and lastarg.getint() == 1:
                    op.setarg(numargs - 1, ConstInt(0))

            if rop.is_jit_emit_jump(opnum):
                self._handle_emit_jump(op)
                continue
            elif rop.is_jit_emit_ret(opnum):
                self._handle_emit_ret(op)
                continue
            elif rop.is_begin_slow_path(opnum):
                self._in_slow_path = True
                jitcell_token = compile.make_jitcell_token(self.jitdriver_sd)
                original_jitcell_token = self.token.original_jitcell_token
                token = TargetToken(jitcell_token,
                                    original_jitcell_token=original_jitcell_token)
                label = ResOperation(rop.LABEL, self.inputargs, descr=token)

                self._newoperations_slow_path = self._newoperations
                self._newoperations = self._slow_ops
                self.send_extra_operation(label)

                original_jitcell_token = self.token.original_jitcell_token
                token = TargetToken(jitcell_token,
                                    original_jitcell_token=original_jitcell_token)
                label = ResOperation(rop.LABEL, self.inputargs, descr=token)
                self._slow_ops.append(label)
                continue

            if self._in_slow_path:
                # re-encountering DEBUG_MERGE_POINT when the slow flag is True
                # means the slow path ends just before
                if rop.is_debug_merge_point(opnum):
                    assert slow_path_jump_op is not None
                    self.send_extra_operation(slow_path_jump_op)
                    slow_path_jump_op = None

                    assert self._slow_path_faildescr is not None
                    label = self._slow_ops[0]
                    info = TraceSplitInfo(label.getdescr(), label, self.inputargs,
                                          faildescr=self._slow_path_faildescr)
                    self._slow_path_newopsandinfo.append((info, self._slow_ops[1:]))

                    self._newoperations = self._newoperations_slow_path[:]
                    self._reset_slow_path()

                    self.send_extra_operation(slow_path_label)
                    slow_path_label = None

                    self.send_extra_operation(op)
                    continue

                elif rop.is_end_slow_path(opnum):
                    jitcell_token = compile.make_jitcell_token(self.jitdriver_sd)
                    original_jitcell_token = self.token.original_jitcell_token
                    token_jump_to = TargetToken(jitcell_token,
                                                original_jitcell_token=original_jitcell_token)
                    slow_path_jump_op = ResOperation(rop.JUMP, self.inputargs, descr=token_jump_to)
                    slow_path_label = ResOperation(rop.LABEL, self.inputargs, descr=token_jump_to)
                    continue

                self.send_extra_operation(op)
                continue

            if rop.is_call(opnum):
                name = self._get_name_from_op(op)
                if endswith(name, "in_fast_path"):
                    self._slow_path_emit_ptr_eq = op

            self.send_extra_operation(op)
            trace.kill_cache_at(deadranges[i + trace.start_index])
            if op.type != 'v':
                i += 1

        # accumulate counters
        if flush:
            self.flush()
            if last_op:
                self.send_extra_operation(last_op)

        if self._newoperations[-1].getopnum() in (rop.JUMP, rop.FINISH):
            label = self._newoperations[0]
            info = TraceSplitInfo(label.getdescr(), label, self.inputargs, self.resumekey)
            self._newopsandinfo.append((info, self._newoperations))

        self._newopsandinfo.extend(self._slow_path_newopsandinfo)

        self.resumedata_memo.update_counters(self.metainterp_sd.profiler)
        # XXX: workaround to pass the type checking
        return self._newopsandinfo[0]

    def _reset_slow_path(self):
        self._slow_ops = []
        self._newoperations_slow_path = []
        self._in_slow_path = False

    def emit_pseudoop(self, op):
        self._pseudoops.append(op)

    def optimize_default(self, op):
        self.emit(op)

    def optimize_GUARD_VALUE(self, op):
        self.emit(op)
        if self._check_if_guard_marked(op):
            newfailargs = []
            for farg in op.getfailargs():
                if not farg in self._specialguardop:
                    newfailargs.append(farg)

            op.setfailargs(newfailargs)
            self._fdescrstack.append(op.getdescr())
        elif op.getarg(0) is self._slow_path_emit_ptr_eq:
            self._slow_path_faildescr = op.getdescr()
            op.setfailargs(self.inputargs)


    optimize_GUARD_TRUE = optimize_GUARD_VALUE
    optimize_GUARD_FALSE = optimize_GUARD_VALUE

    def optimize_CALL_N(self, op):
        descr = op.getdescr()
        effectinfo = descr.get_extra_info()
        name = self._get_name_from_op(op)
        if effectinfo.oopspecindex == EffectInfo.OS_JIT_CALL_ASSEMBLER:
            self._handle_call_assembler(op)
        elif self._check_if_cond_marked(op):
            self._specialguardop.append(op)
            self.emit(op)
        elif startswith(name, "handler_"):
            self._handle_dummy_flag(op)
        else:
            self.emit(op)

    def optimize_CALL_MAY_FORCE_R(self, op):
        descr = op.getdescr()
        effectinfo = descr.get_extra_info()
        name = self._get_name_from_op(op)
        if effectinfo.oopspecindex == EffectInfo.OS_JIT_CALL_ASSEMBLER:
            self._handle_call_assembler(op)
        elif startswith(name, "handler_"):
            self._handle_dummy_flag(op)
        else:
            self.emit(op)

    optimize_CALL_MAY_FORCE_I = optimize_CALL_MAY_FORCE_R
    optimize_CALL_MAY_FORCE_F = optimize_CALL_MAY_FORCE_R
    optimize_CALL_MAY_FORCE_N = optimize_CALL_MAY_FORCE_R

    optimize_CALL_I = optimize_CALL_N
    optimize_CALL_F = optimize_CALL_N
    optimize_CALL_R = optimize_CALL_N

    def _handle_emit_ret(self, op):
        inputargs = self.inputargs
        jd_no = self.jitdriver_sd.index
        result_type = self.jitdriver_sd.result_type
        sd = self.metainterp_sd
        numargs = op.numargs()
        assert numargs > 1, "emit_ret must have at least one argument"
        if result_type == history.VOID:
            exits = []
            finishtoken = sd.done_with_this_frame_descr_void
        elif result_type == history.INT:
            exits = [op.getarg(numargs - 1)]
            finishtoken = sd.done_with_this_frame_descr_int
        elif result_type == history.REF:
            exits = [op.getarg(numargs - 1)]
            finishtoken = sd.done_with_this_frame_descr_ref
        elif result_type == history.FLOAT:
            exits = [op.getarg(numargs - 1)]
            finishtoken = sd.done_with_this_frame_descr_float
        else:
            assert False

        # host-stack style
        ret_ops = [
            ResOperation(rop.LEAVE_PORTAL_FRAME, [ConstInt(jd_no)], None),
            ResOperation(rop.FINISH, exits, finishtoken)
        ]

        label_op = self._newoperations[0]
        info = TraceSplitInfo(label_op.getdescr(), label_op, inputargs, self.resumekey)
        self._newopsandinfo.append((info, self._newoperations[1:] + ret_ops))
        self._newoperations = []

        self._already_setup_current_token = False

        if len(self._fdescrstack) > 0:
            self.resumekey = self._fdescrstack.pop()

    def _handle_emit_jump(self, op, emit_label=False):
        jd = self.jitdriver_sd
        inputargs = self.inputargs
        numargs = op.numargs()

        # create token
        targetbox = op.getarg(numargs - 1)
        assert isinstance(targetbox, ConstInt)
        target = targetbox.getvalue()
        if target in self.token_map.keys():
            target_token = self._get_token(target)
        else:
            # TODO: should get target_token from jitcelltoken.target_tokens
            target_token = self._create_token()
            self._invest_label_jump_dest(targetbox, target_token)

        # TODO: should add target_token to jitcelltoken.target_tokens
        self.token_map[target] = target_token

        jump_op = ResOperation(rop.JUMP, inputargs, descr=target_token)
        info = TraceSplitInfo(target_token, self._newoperations[0], inputargs, self.resumekey)

        self._newopsandinfo.append((info, self._newoperations[1:] + [jump_op]))
        self._newoperations = []

        self._already_setup_current_token = False

        if len(self._fdescrstack) > 0:
            self.resumekey = self._fdescrstack.pop()

    def _handle_call_assembler(self, op):
        "convert recursive calls to an op using `call_assembler_x'"
        jd = self.jitdriver_sd

        arglist = op.getarglist()
        num_green_args = jd.num_green_args
        num_red_args = jd.num_red_args
        greenargs = arglist[1+num_red_args:1+num_red_args+num_green_args]
        args = arglist[1:num_red_args+1]
        assert len(args) == jd.num_red_args
        warmrunnerstate = jd.warmstate
        new_token = warmrunnerstate.get_assembler_token(greenargs)
        opnum = OpHelpers.call_assembler_for_descr(op.getdescr())
        newop = op.copy_and_change(opnum, args, new_token)
        op.set_forwarded(newop)
        self.emit(newop)

    def _handle_dummy_flag(self, op):
        numargs = op.numargs()
        opnum = op.getopnum()
        arglist = op.getarglist()

        newfunc = arglist[-2]
        offset = numargs - 2
        assert offset >= 0
        newargs = arglist[:offset]
        newargs[0] = newfunc

        descr = op.getdescr()
        newdescr = descr.get_calldescr_without_flag()

        newop = op.copy_and_change(opnum, newargs, descr=newdescr)
        op.set_forwarded(newop)
        self.emit(newop)

    def _check_and_insert_label(self, ops, targetbox, token):
        for i, op in enumerate(ops):
            if op.getopnum() == rop.DEBUG_MERGE_POINT:
                if self._insert_label(op, i, ops, targetbox, token):
                    return

    def _invest_label_jump_dest(self, targetbox, token):
        for _, ops in self._newopsandinfo:
            self._check_and_insert_label(ops, targetbox, token)

        self._check_and_insert_label(self._newoperations, targetbox, token)

    def _insert_label(self, op, i, ops, targetbox, token):
        jd = self.jitdriver_sd
        num_green_args = jd.num_green_args
        num_red_args = jd.num_red_args
        arglist = op.getarglist()
        greenargs = arglist[1+num_red_args:1+num_red_args+num_green_args]
        posbox = greenargs[0]
        if posbox.same_constant(targetbox):
            label_op = ResOperation(rop.LABEL, self.inputargs, token)
            ops.insert(i, label_op)
            return True
        return False

    def _get_name_from_op(self, op):
        arg0 = op.getarg(0)
        assert isinstance(arg0, ConstInt)
        adr = cast_int_to_adr(arg0.getint())
        return self.metainterp_sd.get_name_from_address(adr)

    def _get_token(self, key):
        if self.token_map is None:
            raise Exception("token_map is None")

        try:
            return self.token_map[key]
        except KeyError:
            raise TokenMapError(key=key)

    def _create_token(self):
        if len(self._newopsandinfo) > 0:
            jitcell_token = compile.make_jitcell_token(self.jitdriver_sd)
            original_jitcell_token = self.token.original_jitcell_token
            return TargetToken(jitcell_token,
                               original_jitcell_token=original_jitcell_token)
        else:
            return self.token

    def _is_guard_marked(self, op, mark):
        "Check if the guard_op is marked"
        assert op.is_guard()
        failargs = op.getarglist()
        for op in self._newoperations:
            opnum = op.getopnum()
            if rop.is_plain_call(opnum) or rop.is_call_may_force(opnum):
                if op in failargs:
                    name = self._get_name_from_op(op)
                    return name.find(mark) != -1
        return False

    def _check_if_guard_marked(self, op):
        conditions = self.conditions
        for cond in conditions:
            if not self._is_guard_marked(op, cond):
                continue
            return True
        return False

    def _check_if_cond_marked(self, op):
        conditions = self.conditions
        name = self._get_name_from_op(op)
        for cond in conditions:
            if not endswith(name, cond):
                continue
            return True
        return False

dispatch_opt = make_dispatcher_method(OptTraceSplit, 'optimize_',
                                      default=OptTraceSplit.optimize_default)
OptTraceSplit.propagate_forward = dispatch_opt
