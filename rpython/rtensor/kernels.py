from rpython.rtyper.lltypesystem import lltype
from rpython.rtyper.lltypesystem import rffi
import os
from rpython.rtensor.core import (ADD, ARITY, AXIS_ALL, BC_L_COL, BC_L_ROW, BC_L_SCALAR, BC_R_COL, BC_R_ROW, BC_R_SCALAR, COMP_NEG_INF, COMP_TYPE, DIV, EQMASK, EXP, F64, GA_COL2CHW, GA_HEADMERGE, GA_HEADSPLIT, GA_IM2COL, KERNEL, MAXR, MUL, NDTYPES, NODEARRAY, NOPCODES, NPARAMS, RELU, RELUGRAD, SHAPEARRAY, SQRT, STORE_TYPE, SUB, SUM, config, is_reduction, param_slot, slot_param, slot_used)
from rpython.rtensor.device import (_env, _here, gpu_enabled, profile, rt_cuda_load, rt_cuda_set_budget)

class SingleKernels(object):
    def __init__(self):
        self.kernels = [_empty_kernel()
                        for i in range(NOPCODES * NPARAMS * NDTYPES)]
        self.done = [False] * NDTYPES

def _empty_kernel():
    k = lltype.malloc(KERNEL)
    k.ninputs = 0
    k.nodes = lltype.malloc(NODEARRAY, 0)
    k.fn = k.sumroot = k.threads = k.shared = k.nextra = 0
    k.rowmode = 0
    k.n = 0
    k.cols = 0
    k.dtype = F64
    k.outputs = lltype.malloc(SHAPEARRAY, 0)
    return k
single_kernels = SingleKernels()

def single_kernel(opcode, p, dtype):
    return single_kernels.kernels[(dtype * NOPCODES + opcode) * NPARAMS +
                                  param_slot(opcode, p)]

def init_device():
    try:
        config.block = int(_env('RTENSOR_BLOCK', '4096'))
        config.num_warps = int(_env('RTENSOR_WARPS', '8'))
        profile.enabled = os.environ.get('RTENSOR_PROFILE') is not None
        rt_cuda_set_budget(int(_env('RTENSOR_BUDGET_MB', '8')) << 20)
    except ValueError:
        pass
    init_dtype(F64)

def init_dtype(dtype):
    if single_kernels.done[dtype]:
        return
    single_kernels.done[dtype] = True
    for opcode in range(NOPCODES):
        for slot in range(NPARAMS):
            if not slot_used(opcode, slot):
                continue
            opcodes = []
            opcodes.append(opcode)
            lefts = []
            lefts.append(0)
            rights = []
            rights.append(1 if ARITY[opcode] == 2 else -1)
            params = []
            params.append(slot_param(opcode, slot))
            single_kernels.kernels[(dtype * NOPCODES + opcode) * NPARAMS +
                                   slot] = build_kernel(
                ARITY[opcode], opcodes, lefts, rights, params, dtype)
class KernelCache(object):
    def __init__(self):
        self.kernels = {}
kernel_cache = KernelCache()

def cached_kernel(key):
    return kernel_cache.kernels.get(key, lltype.nullptr(KERNEL))

def cache_kernel(key, kernel):
    kernel_cache.kernels[key] = kernel

def new_kernel(ninputs, nnodes, dtype=F64):
    kernel = lltype.malloc(KERNEL)
    kernel.ninputs = ninputs
    kernel.nodes = lltype.malloc(NODEARRAY, nnodes)
    kernel.fn = kernel.sumroot = kernel.threads = kernel.shared = kernel.nextra = 0
    kernel.rowmode = 0
    kernel.n = 0
    kernel.cols = 0
    kernel.dtype = dtype
    kernel.outputs = lltype.malloc(SHAPEARRAY, 0)
    return kernel

def add_output(kernel, node):
    old = kernel.outputs
    new = lltype.malloc(SHAPEARRAY, len(old) + 1)
    for i in range(len(old)):
        new[i] = old[i]
    new[len(old)] = node
    kernel.outputs = new
    return len(old)

def next_pow2(c):
    t = 32
    while t < c:
        t *= 2
    return t

def row_tile(kernel):
    if kernel.cols > 0:
        t = next_pow2(kernel.cols)
        if t <= config.block:
            return t
    return config.block

def row_warps(tile):
    w = tile // 128
    if w < 1:
        return 1
    if w > 8:
        return 8
    return w

def kernel_key(kernel):
    rowmode = kernel_row_mode(kernel)
    parts = [str(kernel.ninputs), '0' if rowmode else str(kernel.n)]
    if rowmode:
        parts.append('r%d' % row_tile(kernel))
    for i in range(len(kernel.nodes)):
        node = kernel.nodes[i]
        parts.append('%d:%d:%d:%d' % (node.opcode, node.a, node.b, node.p))
    for i in range(len(kernel.outputs)):
        parts.append('o%d' % kernel.outputs[i])
    parts.append('d%d' % kernel.dtype)
    return ','.join(parts)

def compile_or_reuse(kernel):
    key = kernel_key(kernel)
    cached = cached_kernel(key)
    if cached:
        kernel.fn = cached.fn
        kernel.threads = cached.threads
        kernel.shared = cached.shared
        kernel.nextra = cached.nextra
        kernel.sumroot = cached.sumroot
        kernel.rowmode = cached.rowmode
        return kernel
    finish_kernel(kernel)
    cache_kernel(key, kernel)
    return kernel

def set_node(kernel, i, opcode, a, b, p):
    node = kernel.nodes[i]
    node.opcode = opcode
    node.a = a
    node.b = b
    node.p = p

def finish_kernel(kernel):
    n = len(kernel.nodes)
    kernel.sumroot = int(n > 0 and is_reduction(kernel.nodes[n - 1].opcode))
    kernel.rowmode = int(kernel_row_mode(kernel))
    kernel.fn = compile_gpu(kernel)
    return kernel

def build_kernel(ninputs, opcodes, lefts, rights, params, dtype=F64):
    kernel = new_kernel(ninputs, len(opcodes), dtype)
    for i in range(len(opcodes)):
        set_node(kernel, i, opcodes[i], lefts[i], rights[i], params[i])
    return finish_kernel(kernel)

def to_tile_ir(kernel, name, n):
    ty = STORE_TYPE[kernel.dtype]
    tile = 'tile<%dx%s>' % (n, ty)
    params = ', '.join(['%%in%d: !cuda_tile.ptr<%s>' % (i, ty)
                        for i in range(kernel.ninputs)] +
                       ['%%out: !cuda_tile.ptr<%s>' % ty])
    lines = ['cuda_tile.module {',
             '  cuda_tile.entry @%s(%s) {' % (name, params)]
    for i in range(kernel.ninputs):
        lines.append('    %%v%d = cuda_tile.load_ptr_tko %%in%d : %s'
                     % (i, i, tile))
    result = tile
    nodes = kernel.nodes
    for i in range(len(nodes)):
        node = nodes[i]
        v = kernel.ninputs + i
        if node.opcode == ADD:
            lines.append('    %%v%d = arith.addf %%v%d, %%v%d : %s'
                         % (v, node.a, node.b, tile))
        elif node.opcode == MUL:
            lines.append('    %%v%d = arith.mulf %%v%d, %%v%d : %s'
                         % (v, node.a, node.b, tile))
        elif node.opcode == RELU:
            lines.append('    %%v%d = arith.maximumf %%v%d, %%zero : %s'
                         % (v, node.a, tile))
        elif node.opcode == RELUGRAD:
            lines.append('    %%v%d = cuda_tile.select %%v%d, %%v%d, %%zero : %s'
                         % (v, node.a, node.b, tile))
        else:
            result = 'tile<1x%s>' % ty
            lines.append('    %%v%d = cuda_tile.reduce add %%v%d : %s -> %s'
                         % (v, node.a, tile, result))
    lines.append('    cuda_tile.store_ptr_tko %%out, %%v%d : %s'
                 % (kernel.ninputs + len(nodes) - 1, result))
    lines.append('    cuda_tile.return')
    lines.append('  }')
    lines.append('}')
    return '\n'.join(lines)

def set_mode(modes, v, m):
    if modes[v] < 0:
        modes[v] = m
        return True
    return modes[v] == m

def all_modes(kernel):
    nin = kernel.ninputs
    nodes = kernel.nodes
    modes = [-1] * (nin + len(nodes))
    for k in range(len(nodes) - 1, -1, -1):
        node = nodes[k]
        m = modes[nin + k]
        if m < 0:
            m = 0
        ma = m
        mb = m
        if is_reduction(node.opcode):
            ma = 0
        elif ARITY[node.opcode] == 2:
            if node.p == BC_R_ROW:
                mb = 1
            elif node.p == BC_R_SCALAR:
                mb = 2
            elif node.p == BC_R_COL:
                mb = 3
            elif node.p == BC_L_ROW:
                ma = 1
            elif node.p == BC_L_SCALAR:
                ma = 2
            elif node.p == BC_L_COL:
                ma = 3
        if not set_mode(modes, node.a, ma):
            return []
        if node.b >= 0 and not set_mode(modes, node.b, mb):
            return []
    for i in range(len(modes)):
        if modes[i] < 0:
            modes[i] = 0
    return modes

def input_modes(kernel):
    modes = all_modes(kernel)
    nin = kernel.ninputs
    if len(modes) != nin + len(kernel.nodes):
        return []
    result = []
    for i in range(nin):
        result.append(modes[i])
    return result

def row_mode(kernel, modes):
    nin = kernel.ninputs
    nodes = kernel.nodes
    if len(modes) != nin + len(nodes):
        return False
    for k in range(len(nodes)):
        node = nodes[k]
        if is_reduction(node.opcode) and node.p == 1:
            if k == len(nodes) - 1 or modes[nin + k] == 3:
                return True
    for i in range(nin):
        if modes[i] == 3:
            return True
    return False

def kernel_row_mode(kernel):
    return row_mode(kernel, all_modes(kernel))

def to_ttir(kernel, name):
    modes = all_modes(kernel)
    if len(modes) != kernel.ninputs + len(kernel.nodes):
        return ''
    if row_mode(kernel, modes):
        return to_ttir_row(kernel, name, modes)
    return to_ttir_flat(kernel, name)

def _elementwise(lines, node, v, T, I1):
    if node.opcode == ADD:
        lines.append('    %%v%d = arith.addf %%v%d, %%v%d : %s'
                     % (v, node.a, node.b, T))
    elif node.opcode == MUL:
        lines.append('    %%v%d = arith.mulf %%v%d, %%v%d : %s'
                     % (v, node.a, node.b, T))
    elif node.opcode == RELU:
        lines.append('    %%c%d = arith.cmpf ogt, %%v%d, %%zero : %s'
                     % (v, node.a, T))
        lines.append('    %%v%d = arith.select %%c%d, %%v%d, %%zero : %s, %s'
                     % (v, v, node.a, I1, T))
    elif node.opcode == RELUGRAD:
        lines.append('    %%c%d = arith.cmpf ogt, %%v%d, %%zero : %s'
                     % (v, node.a, T))
        lines.append('    %%v%d = arith.select %%c%d, %%v%d, %%zero : %s, %s'
                     % (v, v, node.b, I1, T))
    elif node.opcode == SUB:
        lines.append('    %%v%d = arith.subf %%v%d, %%v%d : %s'
                     % (v, node.a, node.b, T))
    elif node.opcode == DIV:
        lines.append('    %%v%d = arith.divf %%v%d, %%v%d : %s'
                     % (v, node.a, node.b, T))
    elif node.opcode == EXP:
        lines.append('    %%v%d = math.exp %%v%d : %s' % (v, node.a, T))
    elif node.opcode == SQRT:
        lines.append('    %%v%d = math.sqrt %%v%d : %s' % (v, node.a, T))
    elif node.opcode == EQMASK:
        lines.append('    %%c%d = arith.cmpf oeq, %%v%d, %%v%d : %s'
                     % (v, node.a, node.b, T))
        lines.append('    %%v%d = arith.select %%c%d, %%one, %%zero : %s, %s'
                     % (v, v, I1, T))
    else:
        return False
    return True

def _trunc(lines, tag, v, half, T, TS):
    if not half:
        return '%%v%d' % v
    lines.append('    %%%s = arith.truncf %%v%d : %s to %s' % (tag, v, T, TS))
    return '%%%s' % tag


def to_ttir_row(kernel, name, modes):
    nodes = kernel.nodes
    nin = kernel.ninputs
    last = len(nodes) - 1
    if last < 0:
        return ''
    BLOCK = row_tile(kernel)
    dt = kernel.dtype
    S = STORE_TYPE[dt]
    C = COMP_TYPE[dt]
    half = S != C
    T = 'tensor<%dx%s>' % (BLOCK, C)
    TS = 'tensor<%dx%s>' % (BLOCK, S)
    P = 'tensor<%dx!tt.ptr<%s>>' % (BLOCK, S)
    I32 = 'tensor<%dxi32>' % BLOCK
    I64 = 'tensor<%dxi64>' % BLOCK
    I1 = 'tensor<%dxi1>' % BLOCK
    isred = [False] * (nin + len(nodes))
    for k in range(len(nodes)):
        node = nodes[k]
        if not is_reduction(node.opcode):
            continue
        if node.p == 1:
            if k != last and modes[nin + k] != 3:
                return ''
        elif node.p == AXIS_ALL:
            if k != last or node.opcode != SUM or half:
                return ''
        else:
            return ''
        isred[nin + k] = True
    for k in range(len(kernel.outputs)):
        if isred[kernel.outputs[k]]:
            return ''
    params = ['%%in%d: !tt.ptr<%s>' % (i, S) for i in range(nin)]
    params.append('%%out: !tt.ptr<%s>' % S)
    for k in range(len(kernel.outputs)):
        params.append('%%out%d: !tt.ptr<%s>' % (k, S))
    lines = ['module {',
             '  tt.func public @%s(%s, %%n: i64, %%c: i64) '
             'attributes {noinline = false} {' % (name, ', '.join(params)),
             '    %%zero = arith.constant dense<0.0> : %s' % T,
             '    %%one = arith.constant dense<1.0> : %s' % T,
             '    %pid = tt.get_program_id x : i32',
             '    %rowi = arith.extsi %pid : i32 to i64',
             '    %%range = tt.make_range {end = %d : i32, start = 0 : i32} '
             ': %s' % (BLOCK, I32),
             '    %%ar = arith.extsi %%range : %s to %s' % (I32, I64),
             '    %%cs = tt.splat %%c : i64 -> %s' % I64,
             '    %%mask = arith.cmpi slt, %%ar, %%cs : %s' % I64,
             '    %base = arith.muli %rowi, %c : i64',
             '    %%bases = tt.splat %%base : i64 -> %s' % I64,
             '    %%offs = arith.addi %%bases, %%ar : %s' % I64]
    if half:
        lines.append('    %%zeros = arith.constant dense<0.0> : %s' % TS)
    for i in range(nin):
        if modes[i] == 2 or modes[i] == 3:
            src = '%%in%d' % i
            if modes[i] == 3:
                lines.append('    %%sp%d = tt.addptr %%in%d, %%pid : '
                             '!tt.ptr<%s>, i32' % (i, i, S))
                src = '%%sp%d' % i
            lines.append('    %%sv%d = tt.load %s : !tt.ptr<%s>' % (i, src, S))
            sv = '%%sv%d' % i
            if half:
                lines.append('    %%se%d = arith.extf %s : %s to %s'
                             % (i, sv, S, C))
                sv = '%%se%d' % i
            lines.append('    %%v%d = tt.splat %s : %s -> %s' % (i, sv, C, T))
            continue
        offs = '%offs'
        if modes[i] == 1:
            offs = '%ar'
        lines.append('    %%p%d = tt.splat %%in%d : !tt.ptr<%s> -> %s'
                     % (i, i, S, P))
        lines.append('    %%q%d = tt.addptr %%p%d, %s : %s, %s'
                     % (i, i, offs, P, I64))
        if half:
            lines.append('    %%rv%d = tt.load %%q%d, %%mask, %%zeros : %s'
                         % (i, i, P))
            lines.append('    %%v%d = arith.extf %%rv%d : %s to %s'
                         % (i, i, TS, T))
        else:
            lines.append('    %%v%d = tt.load %%q%d, %%mask, %%zero : %s'
                         % (i, i, P))
    for k in range(len(nodes)):
        node = nodes[k]
        v = nin + k
        if _elementwise(lines, node, v, T, I1):
            continue
        if node.opcode == MAXR:
            lines.append('    %%ninf%d = arith.constant dense<%s> : %s'
                         % (v, COMP_NEG_INF[dt], T))
            init = '%%ninf%d' % v
            combine = 'maximumf'
        else:
            init = '%zero'
            combine = 'addf'
        lines.append('    %%rm%d = arith.select %%mask, %%v%d, %s : %s, %s'
                     % (v, node.a, init, I1, T))
        lines.append('    %%rs%d = "tt.reduce"(%%rm%d) <{axis = 0 : i32}> ({'
                     % (v, v))
        lines.append('    ^bb0(%%x%d: %s, %%y%d: %s):' % (v, C, v, C))
        lines.append('      %%rr%d = arith.%s %%x%d, %%y%d : %s'
                     % (v, combine, v, v, C))
        lines.append('      tt.reduce.return %%rr%d : %s' % (v, C))
        lines.append('    }) : (%s) -> %s' % (T, C))
        lines.append('    %%v%d = tt.splat %%rs%d : %s -> %s' % (v, v, C, T))
        if k != last:
            continue
        if node.p == 1:
            lines.append('    %%po = tt.addptr %%out, %%pid : !tt.ptr<%s>, i32'
                         % S)
            sv = '%%rs%d' % v
            if half:
                lines.append('    %%rt%d = arith.truncf %s : %s to %s'
                             % (v, sv, C, S))
                sv = '%%rt%d' % v
            lines.append('    tt.store %%po, %s : !tt.ptr<%s>' % (sv, S))
        else:
            lines.append('    %true = arith.constant true')
            lines.append('    %%o = tt.atomic_rmw fadd, acq_rel, gpu, %%out, '
                         '%%rs%d, %%true : (!tt.ptr<f64>, f64, i1) -> f64' % v)
    if not isred[nin + last]:
        lines.append('    %%po = tt.splat %%out : !tt.ptr<%s> -> %s' % (S, P))
        lines.append('    %%qo = tt.addptr %%po, %%offs : %s, %s' % (P, I64))
        lines.append('    tt.store %%qo, %s, %%mask : %s'
                     % (_trunc(lines, 'so', nin + last, half, T, TS), P))
    for k in range(len(kernel.outputs)):
        lines.append('    %%po%d = tt.splat %%out%d : !tt.ptr<%s> -> %s'
                     % (k, k, S, P))
        lines.append('    %%qo%d = tt.addptr %%po%d, %%offs : %s, %s'
                     % (k, k, P, I64))
        lines.append('    tt.store %%qo%d, %s, %%mask : %s'
                     % (k, _trunc(lines, 'sx%d' % k, kernel.outputs[k], half,
                                  T, TS), P))
    lines.append('    tt.return')
    lines.append('  }')
    lines.append('}')
    return '\n'.join(lines) + '\n'

def to_ttir_flat(kernel, name):
    nodes = kernel.nodes
    nin = kernel.ninputs
    BLOCK = config.block
    masked = kernel.n == 0 or kernel.n % BLOCK != 0
    dt = kernel.dtype
    S = STORE_TYPE[dt]
    C = COMP_TYPE[dt]
    half = S != C
    T = 'tensor<%dx%s>' % (BLOCK, C)
    TS = 'tensor<%dx%s>' % (BLOCK, S)
    P = 'tensor<%dx!tt.ptr<%s>>' % (BLOCK, S)
    I32 = 'tensor<%dxi32>' % BLOCK
    I64 = 'tensor<%dxi64>' % BLOCK
    I1 = 'tensor<%dxi1>' % BLOCK
    modes = input_modes(kernel)
    if len(modes) != nin:
        return ''
    axis = AXIS_ALL
    if kernel.sumroot:
        axis = nodes[len(nodes) - 1].p
    need_mod = axis == 0
    need_div = axis == 1
    need_zero_off = False
    for i in range(nin):
        if modes[i] == 1:
            need_mod = True
        elif modes[i] == 2:
            need_zero_off = True
        elif modes[i] == 3:
            need_div = True
    if half and kernel.sumroot and nodes[len(nodes) - 1].opcode == SUM:
        return ''
    params = ['%%in%d: !tt.ptr<%s>' % (i, S) for i in range(nin)]
    params.append('%%out: !tt.ptr<%s>' % S)
    for k in range(len(kernel.outputs)):
        params.append('%%out%d: !tt.ptr<%s>' % (k, S))
    lines = ['module {',
             '  tt.func public @%s(%s, %%n: i64, %%c: i64) '
             'attributes {noinline = false} {' % (name, ', '.join(params)),
             '    %%zero = arith.constant dense<0.0> : %s' % T,
             '    %%one = arith.constant dense<1.0> : %s' % T,
             '    %%bs = arith.constant %d : i32' % BLOCK,
             '    %pid = tt.get_program_id x : i32',
             '    %start = arith.muli %pid, %bs : i32',
             '    %%range = tt.make_range {end = %d : i32, start = 0 : i32} '
             ': %s' % (BLOCK, I32),
             '    %%starts = tt.splat %%start : i32 -> %s' % I32,
             '    %%offs = arith.addi %%starts, %%range : %s' % I32,
             '    %%offs64 = arith.extsi %%offs : %s to %s' % (I32, I64),
             '    %%ns = tt.splat %%n : i64 -> %s' % I64,
             '    %%mask = arith.cmpi slt, %%offs64, %%ns : %s' % I64]
    if need_mod or need_div:
        lines.append('    %%cs = tt.splat %%c : i64 -> %s' % I64)
    if need_mod:
        lines.append('    %%offsm = arith.remsi %%offs64, %%cs : %s' % I64)
    if need_div:
        lines.append('    %%offsd = arith.divsi %%offs64, %%cs : %s' % I64)
    if need_zero_off:
        lines.append('    %%zoffs = arith.constant dense<0> : %s' % I32)
    if masked:
        tmask = '%mask'
    else:
        tmask = '%tmask'
        lines.append('    %%tmask = arith.constant dense<true> : %s' % I1)
    if half:
        lines.append('    %%zeros = arith.constant dense<0.0> : %s' % TS)
    for i in range(nin):
        lines.append('    %%p%d = tt.splat %%in%d : !tt.ptr<%s> -> %s'
                     % (i, i, S, P))
        if modes[i] == 1:
            lines.append('    %%q%d = tt.addptr %%p%d, %%offsm : %s, %s'
                         % (i, i, P, I64))
        elif modes[i] == 2:
            lines.append('    %%q%d = tt.addptr %%p%d, %%zoffs : %s, %s'
                         % (i, i, P, I32))
        elif modes[i] == 3:
            lines.append('    %%q%d = tt.addptr %%p%d, %%offsd : %s, %s'
                         % (i, i, P, I64))
        else:
            lines.append('    %%q%d = tt.addptr %%p%d, %%offs : %s, %s'
                         % (i, i, P, I32))
        dst = '%%v%d' % i
        if half:
            dst = '%%rv%d' % i
        if masked:
            lines.append('    %s = tt.load %%q%d, %%mask, %s : %s'
                         % (dst, i, '%zeros' if half else '%zero', P))
        else:
            lines.append('    %s = tt.load %%q%d : %s' % (dst, i, P))
        if half:
            lines.append('    %%v%d = arith.extf %%rv%d : %s to %s'
                         % (i, i, TS, T))
    last = nin + len(nodes) - 1
    for k in range(len(nodes)):
        node = nodes[k]
        v = nin + k
        if _elementwise(lines, node, v, T, I1):
            continue
        if k != len(nodes) - 1 or not is_reduction(node.opcode):
            return ''
        if node.opcode == MAXR:
            if axis != AXIS_ALL or kernel.n == 0 or kernel.n > BLOCK:
                return ''
            src = '%%v%d' % node.a
            if masked:
                lines.append('    %%ninf = arith.constant dense<%s> : %s'
                             % (COMP_NEG_INF[dt], T))
                lines.append('    %%m%d = arith.select %%mask, %%v%d, %%ninf '
                             ': %s, %s' % (v, node.a, I1, T))
                src = '%%m%d' % v
            lines.append('    %%v%d = "tt.reduce"(%s) <{axis = 0 : i32}> ({'
                         % (v, src))
            lines.append('    ^bb0(%%x: %s, %%y: %s):' % (C, C))
            lines.append('      %%r = arith.maximumf %%x, %%y : %s' % C)
            lines.append('      tt.reduce.return %%r : %s' % C)
            lines.append('    }) : (%s) -> %s' % (T, C))
            out = '%%v%d' % v
            if half:
                lines.append('    %%mt%d = arith.truncf %s : %s to %s'
                             % (v, out, C, S))
                out = '%%mt%d' % v
            lines.append('    tt.store %%out, %s : !tt.ptr<%s>' % (out, S))
        else:
            if axis == 0 or axis == 1:
                offs = '%offsm'
                if axis == 1:
                    offs = '%offsd'
                lines.append('    %%po = tt.splat %%out : !tt.ptr<%s> -> %s'
                             % (S, P))
                lines.append('    %%qo = tt.addptr %%po, %s : %s, %s'
                             % (offs, P, I64))
                lines.append('    %%v%d = tt.atomic_rmw fadd, acq_rel, gpu, '
                             '%%qo, %%v%d, %s : (%s, %s, %s) -> %s'
                             % (v, node.a, tmask, P, T, I1, T))
            else:
                lines.append('    %%v%d = "tt.reduce"(%%v%d) <{axis = 0 : i32}> ({'
                             % (v, node.a))
                lines.append('    ^bb0(%%x: %s, %%y: %s):' % (C, C))
                lines.append('      %%r = arith.addf %%x, %%y : %s' % C)
                lines.append('      tt.reduce.return %%r : %s' % C)
                lines.append('    }) : (%s) -> %s' % (T, C))
                lines.append('    %true = arith.constant true')
                lines.append('    %%o = tt.atomic_rmw fadd, acq_rel, gpu, '
                             '%%out, %%v%d, %%true : (!tt.ptr<%s>, %s, i1) '
                             '-> %s' % (v, S, C, C))
    if not kernel.sumroot:
        lines.append('    %%po = tt.splat %%out : !tt.ptr<%s> -> %s' % (S, P))
        lines.append('    %%qo = tt.addptr %%po, %%offs : %s, %s' % (P, I32))
        val = _trunc(lines, 'so', last, half, T, TS)
        if masked:
            lines.append('    tt.store %%qo, %s, %%mask : %s' % (val, P))
        else:
            lines.append('    tt.store %%qo, %s : %s' % (val, P))
    for k in range(len(kernel.outputs)):
        lines.append('    %%po%d = tt.splat %%out%d : !tt.ptr<%s> -> %s'
                     % (k, k, S, P))
        lines.append('    %%qo%d = tt.addptr %%po%d, %%offs : %s, %s'
                     % (k, k, P, I32))
        val = _trunc(lines, 'sx%d' % k, kernel.outputs[k], half, T, TS)
        if masked:
            lines.append('    tt.store %%qo%d, %s, %%mask : %s' % (k, val, P))
        else:
            lines.append('    tt.store %%qo%d, %s : %s' % (k, val, P))
    lines.append('    tt.return')
    lines.append('  }')
    lines.append('}')
    return '\n'.join(lines) + '\n'

class Counter(object):
    n = 0
counter = Counter()

def _write(path, data):
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0644)
    os.write(fd, data)
    os.close(fd)

def _read(path):
    fd = os.open(path, os.O_RDONLY, 0)
    chunks = []
    while True:
        chunk = os.read(fd, 65536)
        if not chunk:
            break
        chunks.append(chunk)
    os.close(fd)
    return ''.join(chunks)

def compile_gpu(kernel):
    if not gpu_enabled():
        return 0
    try:
        return _compile_gpu(kernel)
    except (OSError, ValueError, IndexError):
        return 0

def _compile_gpu(kernel):
    name = 'rtensor_k%d' % counter.n
    counter.n += 1
    src = to_ttir(kernel, name)
    if not src:
        return 0
    warps = config.num_warps
    if kernel.rowmode and kernel.cols > 0:
        warps = row_warps(row_tile(kernel))
    base = _env('TMPDIR', '/tmp') + '/' + name
    _write(base + '.ttir', src)
    cmd = '%s -P %s %s.ttir %s.ptx %s.meta %s %d' % (
        _env('RTENSOR_PYTHON', 'python3'), _here + '/triton_compile.py',
        base, base, base, _env('RTENSOR_CC', '86'), warps)
    if os.system(cmd) != 0:
        return 0
    words = _read(base + '.meta').strip().split(' ')
    kernel.threads = int(words[0])
    kernel.shared = int(words[1])
    kernel.nextra = int(words[2])
    ptx = _read(base + '.ptx')
    p_ptx = rffi.str2charp(ptx)
    p_name = rffi.str2charp(name)
    fn = rt_cuda_load(p_ptx, p_name)
    rffi.free_charp(p_ptx)
    rffi.free_charp(p_name)
    return fn

def needs_zero(kernel):
    if not kernel.sumroot:
        return 0
    root = kernel.nodes[len(kernel.nodes) - 1]
    if root.opcode != SUM:
        return 0
    if kernel.rowmode and root.p != AXIS_ALL:
        return 0
    return 1


class GatherKernel(object):
    def __init__(self, fn, threads, shared, nextra):
        self.fn = fn
        self.threads = threads
        self.shared = shared
        self.nextra = nextra


class GatherCache(object):
    def __init__(self):
        self.kernels = {}
gather_cache = GatherCache()


class Emitter(object):
    def __init__(self):
        self.lines = []
        self.k = 0

    def tmp(self):
        self.k += 1
        return '%%t%d' % self.k

    def add(self, line):
        self.lines.append(line)


def _gconst(e, v, ty):
    r = e.tmp()
    e.add('    %s = arith.constant dense<%d> : %s' % (r, v, ty))
    return r


def _gbin(e, op, a, b, ty):
    r = e.tmp()
    e.add('    %s = arith.%s %s, %s : %s' % (r, op, a, b, ty))
    return r


def _gcmp(e, pred, a, b, ty):
    r = e.tmp()
    e.add('    %s = arith.cmpi %s, %s, %s : %s' % (r, pred, a, b, ty))
    return r


def _gdiv(e, a, v, ty):
    return _gbin(e, 'divsi', a, _gconst(e, v, ty), ty)


def _gmod(e, a, v, ty):
    return _gbin(e, 'remsi', a, _gconst(e, v, ty), ty)


def _gmul(e, a, v, ty):
    return _gbin(e, 'muli', a, _gconst(e, v, ty), ty)


def _gaddc(e, a, v, ty):
    return _gbin(e, 'addi', a, _gconst(e, v, ty), ty)


def _gather_index(e, op, params, I64, I1):
    off = '%offs64'
    srcs = []
    if op == GA_IM2COL:
        n, c, h, w, k, pad = (params[0], params[1], params[2], params[3],
                              params[4], params[5])
        hw = h * w
        kk = k * k
        ckk = c * kk
        row = _gdiv(e, off, ckk, I64)
        col = _gmod(e, off, ckk, I64)
        img = _gdiv(e, row, hw, I64)
        pos = _gmod(e, row, hw, I64)
        ph = _gdiv(e, pos, w, I64)
        pw = _gmod(e, pos, w, I64)
        ch = _gdiv(e, col, kk, I64)
        rk = _gmod(e, col, kk, I64)
        ih = _gaddc(e, _gbin(e, 'addi', ph, _gdiv(e, rk, k, I64), I64),
                    -pad, I64)
        iw = _gaddc(e, _gbin(e, 'addi', pw, _gmod(e, rk, k, I64), I64),
                    -pad, I64)
        zero = _gconst(e, 0, I64)
        ok = _gbin(e, 'andi',
                   _gbin(e, 'andi', _gcmp(e, 'sge', ih, zero, I64),
                         _gcmp(e, 'slt', ih, _gconst(e, h, I64), I64), I1),
                   _gbin(e, 'andi', _gcmp(e, 'sge', iw, zero, I64),
                         _gcmp(e, 'slt', iw, _gconst(e, w, I64), I64), I1), I1)
        base = _gbin(e, 'addi', _gmul(e, img, c * hw, I64),
                     _gmul(e, ch, hw, I64), I64)
        src = _gbin(e, 'addi', base,
                    _gbin(e, 'addi', _gmul(e, ih, w, I64), iw, I64), I64)
        srcs.append(src)
        return srcs, ok
    if op == GA_COL2CHW:
        n, hw, o = params[0], params[1], params[2]
        img = _gdiv(e, off, o * hw, I64)
        rem = _gmod(e, off, o * hw, I64)
        ch = _gdiv(e, rem, hw, I64)
        pos = _gmod(e, rem, hw, I64)
        row = _gbin(e, 'addi', _gmul(e, img, hw, I64), pos, I64)
        srcs.append(_gbin(e, 'addi', _gmul(e, row, o, I64), ch, I64))
        return srcs, ''
    if op == GA_HEADSPLIT or op == GA_HEADMERGE:
        rows, dh, heads = params[0], params[1], params[2]
        if op == GA_HEADSPLIT:
            ostride, sstride = rows * dh, heads * dh
        else:
            ostride, sstride = heads * dh, rows * dh
        oi = _gdiv(e, off, ostride, I64)
        rem = _gmod(e, off, ostride, I64)
        ii = _gdiv(e, rem, dh, I64)
        ci = _gmod(e, rem, dh, I64)
        srcs.append(_gbin(e, 'addi',
                          _gbin(e, 'addi', _gmul(e, ii, sstride, I64),
                                _gmul(e, oi, dh, I64), I64), ci, I64))
        return srcs, ''
    n, c, h, w = params[0], params[1], params[2], params[3]
    oh = h // 2
    ow = w // 2
    pw = _gmod(e, off, ow, I64)
    q = _gdiv(e, off, ow, I64)
    ph = _gmod(e, q, oh, I64)
    q2 = _gdiv(e, q, oh, I64)
    ch = _gmod(e, q2, c, I64)
    img = _gdiv(e, q2, c, I64)
    base = _gbin(e, 'addi', _gmul(e, img, c * h * w, I64),
                 _gmul(e, ch, h * w, I64), I64)
    base = _gbin(e, 'addi', base, _gmul(e, ph, 2 * w, I64), I64)
    base = _gbin(e, 'addi', base, _gmul(e, pw, 2, I64), I64)
    srcs.append(base)
    srcs.append(_gaddc(e, base, 1, I64))
    srcs.append(_gaddc(e, base, w, I64))
    srcs.append(_gaddc(e, base, w + 1, I64))
    return srcs, ''


def to_ttir_gather(op, params, name, dtype):
    BLOCK = config.block
    S = STORE_TYPE[dtype]
    C = COMP_TYPE[dtype]
    half = S != C
    T = 'tensor<%dx%s>' % (BLOCK, S)
    P = 'tensor<%dx!tt.ptr<%s>>' % (BLOCK, S)
    I32 = 'tensor<%dxi32>' % BLOCK
    I64 = 'tensor<%dxi64>' % BLOCK
    I1 = 'tensor<%dxi1>' % BLOCK
    e = Emitter()
    e.add('module {')
    e.add('  tt.func public @%s(%%in: !tt.ptr<%s>, %%out: !tt.ptr<%s>, '
          '%%n: i64, %%c: i64) attributes {noinline = false} {' % (name, S, S))
    e.add('    %%zero = arith.constant dense<0.0> : %s' % T)
    e.add('    %%bs = arith.constant %d : i32' % BLOCK)
    e.add('    %pid = tt.get_program_id x : i32')
    e.add('    %start = arith.muli %pid, %bs : i32')
    e.add('    %%range = tt.make_range {end = %d : i32, start = 0 : i32} : %s'
          % (BLOCK, I32))
    e.add('    %%starts = tt.splat %%start : i32 -> %s' % I32)
    e.add('    %%offs = arith.addi %%starts, %%range : %s' % I32)
    e.add('    %%offs64 = arith.extsi %%offs : %s to %s' % (I32, I64))
    e.add('    %%ns = tt.splat %%n : i64 -> %s' % I64)
    e.add('    %%mask = arith.cmpi slt, %%offs64, %%ns : %s' % I64)
    srcs, ok = _gather_index(e, op, params, I64, I1)
    mask = '%mask'
    if ok:
        mask = _gbin(e, 'andi', mask, ok, I1)
    e.add('    %%pin = tt.splat %%in : !tt.ptr<%s> -> %s' % (S, P))
    vals = []
    for i in range(len(srcs)):
        q = e.tmp()
        e.add('    %s = tt.addptr %%pin, %s : %s, %s' % (q, srcs[i], P, I64))
        v = e.tmp()
        e.add('    %s = tt.load %s, %s, %%zero : %s' % (v, q, mask, P))
        vals.append(v)
    acc = vals[0]
    if len(vals) > 1 and half:
        TC = 'tensor<%dx%s>' % (BLOCK, C)
        for i in range(len(vals)):
            r = e.tmp()
            e.add('    %s = arith.extf %s : %s to %s' % (r, vals[i], T, TC))
            vals[i] = r
        acc = vals[0]
        for i in range(1, len(vals)):
            acc = _gbin(e, 'maximumf', acc, vals[i], TC)
        r = e.tmp()
        e.add('    %s = arith.truncf %s : %s to %s' % (r, acc, TC, T))
        acc = r
    else:
        for i in range(1, len(vals)):
            acc = _gbin(e, 'maximumf', acc, vals[i], T)
    e.add('    %%pout = tt.splat %%out : !tt.ptr<%s> -> %s' % (S, P))
    e.add('    %%qout = tt.addptr %%pout, %%offs : %s, %s' % (P, I32))
    e.add('    tt.store %%qout, %s, %%mask : %s' % (acc, P))
    e.add('    tt.return')
    e.add('  }')
    e.add('}')
    return '\n'.join(e.lines) + '\n'


def gather_key(op, params, dtype):
    parts = ['g%d' % op]
    for i in range(len(params)):
        parts.append(str(params[i]))
    parts.append('d%d' % dtype)
    return ','.join(parts)


def gather_kernel(op, params, dtype):
    key = gather_key(op, params, dtype)
    k = gather_cache.kernels.get(key, None)
    if k is None:
        k = _gather_compile(op, params, dtype)
        gather_cache.kernels[key] = k
    return k


def _gather_compile(op, params, dtype):
    if not gpu_enabled():
        return GatherKernel(0, 0, 0, 0)
    try:
        return _gather_compile_gpu(op, params, dtype)
    except (OSError, ValueError, IndexError):
        return GatherKernel(0, 0, 0, 0)


def _gather_compile_gpu(op, params, dtype):
    name = 'rtensor_g%d' % counter.n
    counter.n += 1
    src = to_ttir_gather(op, params, name, dtype)
    base = _env('TMPDIR', '/tmp') + '/' + name
    _write(base + '.ttir', src)
    cmd = '%s -P %s %s.ttir %s.ptx %s.meta %s %d' % (
        _env('RTENSOR_PYTHON', 'python3'), _here + '/triton_compile.py',
        base, base, base, _env('RTENSOR_CC', '86'), config.num_warps)
    if os.system(cmd) != 0:
        return GatherKernel(0, 0, 0, 0)
    words = _read(base + '.meta').strip().split(' ')
    threads = int(words[0])
    shared = int(words[1])
    nextra = int(words[2])
    ptx = _read(base + '.ptx')
    p_ptx = rffi.str2charp(ptx)
    p_name = rffi.str2charp(name)
    fn = rt_cuda_load(p_ptx, p_name)
    rffi.free_charp(p_ptx)
    rffi.free_charp(p_name)
    return GatherKernel(fn, threads, shared, nextra)

