from rpython.rlib.rfloat import INFINITY, formatd
from rpython.metatensor.core import (ADD, ARITY, B_KEEP_NZ, B_KEEP_Z, B_LT, B_MAX, B_MIN, B_NE, B_POW, BINARY, U_NEG, U_SIGMOID, U_TANH, UNARY, bc_mode, binary_fn, GA_ROWS, AXIS_ALL, BC_L_COL, BC_L_ROW, BC_L_SCALAR, BC_R_COL, BC_R_ROW, BC_R_SCALAR, COMP_NEG_INF, COMP_TYPE, DIV, EQMASK, EXP, GATHER, GA_COL2CHW, GA_HEADMERGE, GA_HEADSPLIT, GA_IM2COL, GA_IM2COL_NHWC, GA_IM2COL2, GA_IM2COL_T, GA_MAXPOOL_NHWC, GA_POOL, GA_ROTHALF, GA_STRIDED, GA_TAKE, POOL_ADAPTIVE, POOL_AVG, POOL_DEPTHWISE, conv_out, MAXR, MUL, RELU, RELUGRAD, SQRT, STORE_TYPE, SUB, SUM, config, gather_changes_shape, gather_dh, gather_heads, gather_kind, is_reduction, nvals)


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

def set_mode(modes, v, m):
    if modes[v] < 0:
        modes[v] = m
        return True
    return modes[v] == m

def all_modes(kernel):
    nin = nvals(kernel)
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
            bc = bc_mode(node.opcode, node.p)
            if bc == BC_R_ROW:
                mb = 1
            elif bc == BC_R_SCALAR:
                mb = 2
            elif bc == BC_R_COL:
                mb = 3
            elif bc == BC_L_ROW:
                ma = 1
            elif bc == BC_L_SCALAR:
                ma = 2
            elif bc == BC_L_COL:
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
    nin = nvals(kernel)
    if len(modes) != nin + len(kernel.nodes):
        return []
    result = []
    for i in range(kernel.ninputs):
        result.append(modes[i])
    return result

def row_mode(kernel, modes):
    nin = nvals(kernel)
    nodes = kernel.nodes
    if len(modes) != nin + len(nodes):
        return False
    if has_gather(kernel):
        return False
    for k in range(len(nodes)):
        node = nodes[k]
        if is_reduction(node.opcode) and node.p == 1:
            if k == len(nodes) - 1 or modes[nin + k] == 3:
                return True
    if kernel.cols <= 0 or kernel.cols > config.block:
        return False
    for i in range(nin):
        if modes[i] == 3:
            return True
    return False

def kernel_row_mode(kernel):
    return row_mode(kernel, all_modes(kernel))

def out_modes(kernel):
    modes = all_modes(kernel)
    if len(modes) != nvals(kernel) + len(kernel.nodes):
        return []
    result = []
    for k in range(len(kernel.outputs)):
        result.append(modes[kernel.outputs[k]])
    return result

def _fliteral(v):
    # MLIR wants a decimal point before any exponent, so 1e-05 is rejected;
    # %e always has one, and 16 digits round-trips an f64.
    t = formatd(v, 'e', 16)
    if 'inf' in t or 'nan' in t:
        return ''
    return t


def _tile_types(block, S, C):
    return ('tensor<%dx%s>' % (block, C), 'tensor<%dx%s>' % (block, S),
            'tensor<%dx!tt.ptr<%s>>' % (block, S), 'tensor<%dxi32>' % block,
            'tensor<%dxi64>' % block, 'tensor<%dxi1>' % block)

def _flat_prologue(lines, block, I32, I64):
    lines.append('    %%bs = arith.constant %d : i32' % block)
    lines.append('    %pid = tt.get_program_id x : i32')
    lines.append('    %start = arith.muli %pid, %bs : i32')
    lines.append('    %%range = tt.make_range {end = %d : i32, start = 0 : i32} '
                 ': %s' % (block, I32))
    lines.append('    %%starts = tt.splat %%start : i32 -> %s' % I32)
    lines.append('    %%offs = arith.addi %%starts, %%range : %s' % I32)
    lines.append('    %%offs64 = arith.extsi %%offs : %s to %s' % (I32, I64))
    lines.append('    %%ns = tt.splat %%n : i64 -> %s' % I64)
    lines.append('    %%mask = arith.cmpi slt, %%offs64, %%ns : %s' % I64)


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

def has_gather(kernel):
    for k in range(len(kernel.nodes)):
        if kernel.nodes[k].opcode == GATHER:
            return True
    return False

def to_ttir(kernel, name):
    modes = all_modes(kernel)
    if len(modes) != nvals(kernel) + len(kernel.nodes):
        return ''
    if has_gather(kernel):
        return to_ttir_gathered(kernel, name, modes)
    if row_mode(kernel, modes):
        return to_ttir_row(kernel, name, modes)
    return to_ttir_flat(kernel, name)

def _elementwise(lines, node, v, T, I1):
    return _ew(lines, node.opcode, node.p, '%%v%d' % v, '%%c%d' % v,
               '%%v%d' % node.a, '%%v%d' % node.b, T, I1)

def _ew(lines, opcode, p, dst, cmp, a, b, T, I1):
    if opcode == ADD:
        lines.append('    %s = arith.addf %s, %s : %s' % (dst, a, b, T))
    elif opcode == MUL:
        lines.append('    %s = arith.mulf %s, %s : %s' % (dst, a, b, T))
    elif opcode == RELU:
        lines.append('    %s = arith.cmpf ogt, %s, %%zero : %s' % (cmp, a, T))
        lines.append('    %s = arith.select %s, %s, %%zero : %s, %s'
                     % (dst, cmp, a, I1, T))
    elif opcode == RELUGRAD:
        lines.append('    %s = arith.cmpf ogt, %s, %%zero : %s' % (cmp, a, T))
        lines.append('    %s = arith.select %s, %s, %%zero : %s, %s'
                     % (dst, cmp, b, I1, T))
    elif opcode == SUB:
        lines.append('    %s = arith.subf %s, %s : %s' % (dst, a, b, T))
    elif opcode == DIV:
        lines.append('    %s = arith.divf %s, %s : %s' % (dst, a, b, T))
    elif opcode == EXP:
        lines.append('    %s = math.exp %s : %s' % (dst, a, T))
    elif opcode == SQRT:
        lines.append('    %s = math.sqrt %s : %s' % (dst, a, T))
    elif opcode == EQMASK:
        lines.append('    %s = arith.cmpf oeq, %s, %s : %s' % (cmp, a, b, T))
        lines.append('    %s = arith.select %s, %%one, %%zero : %s, %s'
                     % (dst, cmp, I1, T))
    elif opcode == UNARY:
        return _unary(lines, p, dst, a, T)
    elif opcode == BINARY:
        return _binary(lines, binary_fn(p), dst, cmp, a, b, T, I1)
    else:
        return False
    return True

def _libdevice(T, name):
    # f16 computes in f32, so the compute type picks the variant
    if 'xf64>' in T:
        return '__nv_' + name
    return '__nv_' + name + 'f'

_UNARY_OPS = ['', '', 'math.log', 'math.absf', 'math.sin', 'math.cos',
              'math.erf', 'math.floor', 'arith.negf']

def _unary(lines, fn, dst, a, T):
    if fn == U_TANH:
        # Triton lowers no math.tanh
        lines.append('    %s = tt.extern_elementwise %s {libname = "", '
                     'libpath = "", pure = true, symbol = "%s"} : (%s) -> %s'
                     % (dst, a, _libdevice(T, 'tanh'), T, T))
    elif fn == U_SIGMOID:
        lines.append('    %s_n = arith.negf %s : %s' % (dst, a, T))
        lines.append('    %s_e = math.exp %s_n : %s' % (dst, dst, T))
        lines.append('    %s_d = arith.addf %s_e, %%one : %s' % (dst, dst, T))
        lines.append('    %s = arith.divf %%one, %s_d : %s' % (dst, dst, T))
    elif fn > U_SIGMOID and fn <= U_NEG:
        lines.append('    %s = %s %s : %s' % (dst, _UNARY_OPS[fn], a, T))
    else:
        return False
    return True

_CMP = ['', '', '', 'olt', 'ole', 'ogt', 'oge', 'oeq', 'une']

def _binary(lines, fn, dst, cmp, a, b, T, I1):
    if fn == B_MAX:
        lines.append('    %s = arith.maximumf %s, %s : %s' % (dst, a, b, T))
    elif fn == B_MIN:
        lines.append('    %s = arith.minimumf %s, %s : %s' % (dst, a, b, T))
    elif fn == B_POW:
        lines.append('    %s = tt.extern_elementwise %s, %s {libname = "", '
                     'libpath = "", pure = true, symbol = "%s"} : '
                     '(%s, %s) -> %s' % (dst, a, b, _libdevice(T, 'pow'),
                                         T, T, T))
    elif fn >= B_LT and fn <= B_NE:
        lines.append('    %s = arith.cmpf %s, %s, %s : %s'
                     % (cmp, _CMP[fn], a, b, T))
        lines.append('    %s = arith.select %s, %%one, %%zero : %s, %s'
                     % (dst, cmp, I1, T))
    elif fn == B_KEEP_NZ or fn == B_KEEP_Z:
        pred = 'une' if fn == B_KEEP_NZ else 'oeq'
        lines.append('    %s = arith.cmpf %s, %s, %%zero : %s'
                     % (cmp, pred, a, T))
        lines.append('    %s = arith.select %s, %s, %%zero : %s, %s'
                     % (dst, cmp, b, I1, T))
    else:
        return False
    return True

def _trunc(lines, tag, v, half, T, TS):
    if not half:
        return '%%v%d' % v
    lines.append('    %%%s = arith.truncf %%v%d : %s to %s' % (tag, v, T, TS))
    return '%%%s' % tag

def _emit_consts(lines, kernel, nreal, T):
    for j in range(len(kernel.consts)):
        text = _fliteral(kernel.consts[j])
        if not text:
            return False
        lines.append('    %%v%d = arith.constant dense<%s> : %s'
                     % (nreal + j, text, T))
    return True


def to_ttir_row(kernel, name, modes):
    nodes = kernel.nodes
    nreal = kernel.ninputs
    nin = nvals(kernel)
    last = len(nodes) - 1
    if last < 0:
        return ''
    BLOCK = row_tile(kernel)
    dt = kernel.dtype
    S = STORE_TYPE[dt]
    C = COMP_TYPE[dt]
    half = S != C
    T, TS, P, I32, I64, I1 = _tile_types(BLOCK, S, C)
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
    omodes = out_modes(kernel)
    if len(omodes) != len(kernel.outputs):
        return ''
    params = ['%%in%d: !tt.ptr<%s>' % (i, S) for i in range(nreal)]
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
    if not _emit_consts(lines, kernel, nreal, T):
        return ''
    for i in range(nreal):
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
            # The accumulator has the kernel's storage type, not always f64,
            # and for f16 the reduction runs in f32 and has to be narrowed
            # first, exactly as the tt.store branch above does.
            sv = '%%rs%d' % v
            if half:
                lines.append('    %%rt%d = arith.truncf %s : %s to %s'
                             % (v, sv, C, S))
                sv = '%%rt%d' % v
            lines.append('    %%o = tt.atomic_rmw fadd, acq_rel, gpu, %%out, '
                         '%s, %%true : (!tt.ptr<%s>, %s, i1) -> %s'
                         % (sv, S, S, S))
    if not isred[nin + last]:
        lines.append('    %%po = tt.splat %%out : !tt.ptr<%s> -> %s' % (S, P))
        lines.append('    %%qo = tt.addptr %%po, %%offs : %s, %s' % (P, I64))
        lines.append('    tt.store %%qo, %s, %%mask : %s'
                     % (_trunc(lines, 'so', nin + last, half, T, TS), P))
    for k in range(len(kernel.outputs)):
        ooffs = '%offs'
        if omodes[k] == 1:
            ooffs = '%ar'
        elif omodes[k] == 2:
            lines.append('    %%zo%d = arith.constant dense<0> : %s' % (k, I64))
            ooffs = '%%zo%d' % k
        elif omodes[k] == 3:
            lines.append('    %%ro%d = tt.splat %%rowi : i64 -> %s' % (k, I64))
            ooffs = '%%ro%d' % k
        lines.append('    %%po%d = tt.splat %%out%d : !tt.ptr<%s> -> %s'
                     % (k, k, S, P))
        lines.append('    %%qo%d = tt.addptr %%po%d, %s : %s, %s'
                     % (k, k, ooffs, P, I64))
        lines.append('    tt.store %%qo%d, %s, %%mask : %s'
                     % (k, _trunc(lines, 'sx%d' % k, kernel.outputs[k], half,
                                  T, TS), P))
    lines.append('    tt.return')
    lines.append('  }')
    lines.append('}')
    return '\n'.join(lines) + '\n'

def to_ttir_flat(kernel, name):
    nodes = kernel.nodes
    nreal = kernel.ninputs
    nin = nvals(kernel)
    BLOCK = config.flat
    masked = kernel.n == 0 or kernel.n % BLOCK != 0
    dt = kernel.dtype
    S = STORE_TYPE[dt]
    C = COMP_TYPE[dt]
    half = S != C
    T, TS, P, I32, I64, I1 = _tile_types(BLOCK, S, C)
    modes = input_modes(kernel)
    if len(modes) != nreal:
        return ''
    omodes = out_modes(kernel)
    if len(omodes) != len(kernel.outputs):
        return ''
    axis = AXIS_ALL
    if kernel.sumroot:
        axis = nodes[len(nodes) - 1].p
    need_mod = axis == 0
    need_div = axis == 1
    need_zero_off = False
    for i in range(nreal):
        if modes[i] == 1:
            need_mod = True
        elif modes[i] == 2:
            need_zero_off = True
        elif modes[i] == 3:
            need_div = True
    for k in range(len(omodes)):
        if omodes[k] == 1:
            need_mod = True
        elif omodes[k] == 2:
            need_zero_off = True
        elif omodes[k] == 3:
            need_div = True
    if half and kernel.sumroot and nodes[len(nodes) - 1].opcode == SUM:
        return ''
    params = ['%%in%d: !tt.ptr<%s>' % (i, S) for i in range(nreal)]
    params.append('%%out: !tt.ptr<%s>' % S)
    for k in range(len(kernel.outputs)):
        params.append('%%out%d: !tt.ptr<%s>' % (k, S))
    lines = ['module {',
             '  tt.func public @%s(%s, %%n: i64, %%c: i64) '
             'attributes {noinline = false} {' % (name, ', '.join(params)),
             '    %%zero = arith.constant dense<0.0> : %s' % T,
             '    %%one = arith.constant dense<1.0> : %s' % T]
    if not _emit_consts(lines, kernel, nreal, T):
        return ''
    _flat_prologue(lines, BLOCK, I32, I64)
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
    for i in range(nreal):
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
                # The tail lanes hold whatever the chain makes of a masked
                # load, which is only zero when every op in it maps 0 to 0 -
                # not true once a constant is folded into the body.  Zero them
                # before the reduction, the way the MAXR branch does.
                src = '%%v%d' % node.a
                if masked:
                    lines.append('    %%rm%d = arith.select %%mask, %%v%d, '
                                 '%%zero : %s, %s' % (v, node.a, I1, T))
                    src = '%%rm%d' % v
                lines.append('    %%v%d = "tt.reduce"(%s) <{axis = 0 : i32}> ({'
                             % (v, src))
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
        ooffs = '%offs'
        OI = I32
        if omodes[k] == 1:
            ooffs = '%offsm'
            OI = I64
        elif omodes[k] == 2:
            ooffs = '%zoffs'
        elif omodes[k] == 3:
            ooffs = '%offsd'
            OI = I64
        lines.append('    %%po%d = tt.splat %%out%d : !tt.ptr<%s> -> %s'
                     % (k, k, S, P))
        lines.append('    %%qo%d = tt.addptr %%po%d, %s : %s, %s'
                     % (k, k, ooffs, P, OI))
        val = _trunc(lines, 'sx%d' % k, kernel.outputs[k], half, T, TS)
        if masked:
            lines.append('    tt.store %%qo%d, %s, %%mask : %s' % (k, val, P))
        else:
            lines.append('    tt.store %%qo%d, %s : %s' % (k, val, P))
    lines.append('    tt.return')
    lines.append('  }')
    lines.append('}')
    return '\n'.join(lines) + '\n'

def _gdivv(e, a, v, ty):
    if v > 0:
        return _gdiv(e, a, v, ty)
    return _gbin(e, 'divsi', a, '%rd', ty)


def _gmodv(e, a, v, ty):
    if v > 0:
        return _gmod(e, a, v, ty)
    return _gbin(e, 'remsi', a, '%rd', ty)


def _gmulv(e, a, v, ty):
    if v > 0:
        return _gmul(e, a, v, ty)
    return _gbin(e, 'muli', a, '%rd', ty)


def _gather_src(e, p, idx, rd, I64):
    """Where output lane `idx` of a GATHER node reads its operand.  `rd` is
    rows * dh: a constant when the kernel's size is, else 0 for the %rd
    splat of n / heads.  Dividing by a constant is a multiply and a shift;
    dividing by %rd is a real division in every lane."""
    kind = gather_kind(p)
    dh = gather_dh(p)
    if kind == GA_ROTHALF:
        blk = _gdiv(e, idx, dh, I64)
        c = _gmod(e, idx, dh, I64)
        rc = _gmod(e, _gaddc(e, c, dh // 2, I64), dh, I64)
        return _gbin(e, 'addi', _gmul(e, blk, dh, I64), rc, I64)
    hd = gather_heads(p) * dh
    if kind == GA_HEADSPLIT:
        oi = _gdivv(e, idx, rd, I64)
        rem = _gmodv(e, idx, rd, I64)
        return _gbin(e, 'addi', _gbin(e, 'addi',
                                      _gmul(e, _gdiv(e, rem, dh, I64), hd, I64),
                                      _gmul(e, oi, dh, I64), I64),
                     _gmod(e, rem, dh, I64), I64)
    oi = _gdiv(e, idx, hd, I64)
    rem = _gmod(e, idx, hd, I64)
    return _gbin(e, 'addi', _gbin(e, 'addi',
                                  _gmulv(e, _gdiv(e, rem, dh, I64), rd, I64),
                                  _gmul(e, oi, dh, I64), I64),
                 _gmod(e, rem, dh, I64), I64)


class Gathered(object):
    """Emits a flat kernel whose nodes may include GATHERs.  A value is
    emitted once per index it is read at: a gather reads its operand at a
    permuted index, and everything under it - leaves included - is evaluated
    there, so the permutation costs addressing and nothing else."""

    def __init__(self, kernel, e, modes, T, P, IX, I1, S, half, rd):
        self.rd = rd
        self.kernel = kernel
        self.e = e
        self.modes = modes
        self.T = T
        self.P = P
        self.IX = IX
        self.I1 = I1
        self.S = S
        self.half = half
        self.memo = {}

    def val(self, v, idx):
        key = '%d@%s' % (v, idx)
        r = self.memo.get(key, '')
        if r:
            return r
        r = self._val(v, idx)
        self.memo[key] = r
        return r

    def _val(self, v, idx):
        kernel = self.kernel
        e = self.e
        nreal = kernel.ninputs
        nin = nvals(kernel)
        if v >= nreal and v < nin:
            return '%%v%d' % v
        if v < nreal:
            mode = self.modes[v]
            if mode == 1:
                off = _gbin(e, 'remsi', idx, '%cs', self.IX)
            elif mode == 3:
                off = _gbin(e, 'divsi', idx, '%cs', self.IX)
            elif mode == 2:
                off = _gconst(e, 0, self.IX)
            else:
                off = idx
            q = e.tmp()
            e.add('    %s = tt.addptr %%p%d, %s : %s, %s'
                  % (q, v, off, self.P, self.IX))
            r = e.tmp()
            if self.half:
                e.add('    %s = tt.load %s, %%mask, %%zeros : %s' % (r, q, self.P))
                x = e.tmp()
                e.add('    %s = arith.extf %s : %s to %s'
                      % (x, r, 'tensor<%dx%s>' % (config.flat, self.S), self.T))
                return x
            e.add('    %s = tt.load %s, %%mask, %%zero : %s' % (r, q, self.P))
            return r
        node = kernel.nodes[v - nin]
        if node.opcode == GATHER:
            src = _gather_src(e, node.p, idx, self.rd, self.IX)
            return self.val(node.a, src)
        a = self.val(node.a, idx)
        b = ''
        if node.b >= 0:
            b = self.val(node.b, idx)
        r = e.tmp()
        if not _ew(e.lines, node.opcode, node.p, r, e.tmp(), a, b, self.T,
                   self.I1):
            return ''
        return r


def to_ttir_gathered(kernel, name, modes):
    nodes = kernel.nodes
    nreal = kernel.ninputs
    nin = nvals(kernel)
    last = len(nodes) - 1
    if last < 0:
        return ''
    omodes = out_modes(kernel)
    if len(omodes) != len(kernel.outputs):
        return ''
    for k in range(len(omodes)):
        if omodes[k] != 0:
            return ''
    BLOCK = config.flat
    dt = kernel.dtype
    S = STORE_TYPE[dt]
    C = COMP_TYPE[dt]
    half = S != C
    T, TS, P, I32, I64, I1 = _tile_types(BLOCK, S, C)
    reshaped = False
    heads = 1
    for k in range(len(nodes)):
        node = nodes[k]
        if node.opcode == GATHER:
            if gather_changes_shape(node.p):
                if reshaped and heads != gather_heads(node.p):
                    return ''
                reshaped = True
                heads = gather_heads(node.p)
        elif is_reduction(node.opcode):
            if k != last or node.p != AXIS_ALL:
                return ''
            if node.opcode == MAXR and (kernel.n == 0 or kernel.n > BLOCK):
                return ''
            if half and node.opcode == SUM:
                return ''
    for i in range(nreal):
        # A row- or column-broadcast operand is addressed by the column count
        # of the big input, which a reshaping gather makes ambiguous.
        if reshaped and (modes[i] == 1 or modes[i] == 3):
            return ''
    e = Emitter()
    e.add('module {')
    params = ['%%in%d: !tt.ptr<%s>' % (i, S) for i in range(nreal)]
    params.append('%%out: !tt.ptr<%s>' % S)
    for k in range(len(kernel.outputs)):
        params.append('%%out%d: !tt.ptr<%s>' % (k, S))
    e.add('  tt.func public @%s(%s, %%n: i64, %%c: i64) '
          'attributes {noinline = false} {' % (name, ', '.join(params)))
    e.add('    %%zero = arith.constant dense<0.0> : %s' % T)
    e.add('    %%one = arith.constant dense<1.0> : %s' % T)
    if not _emit_consts(e.lines, kernel, nreal, T):
        return ''
    _flat_prologue(e.lines, BLOCK, I32, I64)
    # Lane indices are i32: a flat kernel's size is below 2**31, and 64-bit
    # division and remainder cost several times the 32-bit ones.
    e.add('    %c32 = arith.trunci %c : i64 to i32')
    e.add('    %%cs = tt.splat %%c32 : i32 -> %s' % I32)
    rd = 0
    if kernel.n > 0 and kernel.n % heads == 0:
        rd = kernel.n // heads
    else:
        e.add('    %%hc = arith.constant %d : i64' % heads)
        e.add('    %nh = arith.divsi %n, %hc : i64')
        e.add('    %nh32 = arith.trunci %nh : i64 to i32')
        e.add('    %%rd = tt.splat %%nh32 : i32 -> %s' % I32)
    if half:
        e.add('    %%zeros = arith.constant dense<0.0> : %s' % TS)
    for i in range(nreal):
        e.add('    %%p%d = tt.splat %%in%d : !tt.ptr<%s> -> %s' % (i, i, S, P))
    g = Gathered(kernel, e, modes, T, P, I32, I1, S, half, rd)
    root = nodes[last]
    if is_reduction(root.opcode):
        x = g.val(root.a, '%offs')
        if not x:
            return ''
        init = '%zero'
        combine = 'addf'
        if root.opcode == MAXR:
            e.add('    %%ninf = arith.constant dense<%s> : %s'
                  % (COMP_NEG_INF[dt], T))
            init = '%ninf'
            combine = 'maximumf'
        m = e.tmp()
        e.add('    %s = arith.select %%mask, %s, %s : %s, %s'
              % (m, x, init, I1, T))
        r = e.tmp()
        e.add('    %s = "tt.reduce"(%s) <{axis = 0 : i32}> ({' % (r, m))
        e.add('    ^bb0(%%x: %s, %%y: %s):' % (C, C))
        e.add('      %%r = arith.%s %%x, %%y : %s' % (combine, C))
        e.add('      tt.reduce.return %%r : %s' % C)
        e.add('    }) : (%s) -> %s' % (T, C))
        if root.opcode == MAXR:
            if half:
                t = e.tmp()
                e.add('    %s = arith.truncf %s : %s to %s' % (t, r, C, S))
                r = t
            e.add('    tt.store %%out, %s : !tt.ptr<%s>' % (r, S))
        else:
            e.add('    %true = arith.constant true')
            e.add('    %%o = tt.atomic_rmw fadd, acq_rel, gpu, %%out, %s, '
                  '%%true : (!tt.ptr<%s>, %s, i1) -> %s' % (r, S, C, C))
    else:
        x = g.val(nin + last, '%offs')
        if not x:
            return ''
        if half:
            t = e.tmp()
            e.add('    %s = arith.truncf %s : %s to %s' % (t, x, T, TS))
            x = t
        e.add('    %%po = tt.splat %%out : !tt.ptr<%s> -> %s' % (S, P))
        e.add('    %%qo = tt.addptr %%po, %%offs : %s, %s' % (P, I32))
        e.add('    tt.store %%qo, %s, %%mask : %s' % (x, P))
    # An extra output is a node's own value, so it is read at the lane's own
    # index even when the root reads it through a gather.
    for k in range(len(kernel.outputs)):
        x = g.val(kernel.outputs[k], '%offs')
        if not x:
            return ''
        if half:
            t = e.tmp()
            e.add('    %s = arith.truncf %s : %s to %s' % (t, x, T, TS))
            x = t
        po = e.tmp()
        e.add('    %s = tt.splat %%out%d : !tt.ptr<%s> -> %s' % (po, k, S, P))
        qo = e.tmp()
        e.add('    %s = tt.addptr %s, %%offs : %s, %s' % (qo, po, P, I32))
        e.add('    tt.store %s, %s, %%mask : %s' % (qo, x, P))
    e.add('    tt.return')
    e.add('  }')
    e.add('}')
    return '\n'.join(e.lines) + '\n'


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


def _gclamp(e, a, hi, ty):
    a = _gbin(e, 'maxsi', a, _gconst(e, 0, ty), ty)
    return _gbin(e, 'minsi', a, _gconst(e, hi, ty), ty)

def _gather_index(e, op, params, I64, I1):
    off = '%offs64'
    srcs = []
    if op == GA_IM2COL:
        n, c, h, w, k, pad, stride = (params[0], params[1], params[2],
                                      params[3], params[4], params[5],
                                      params[6])
        hw = h * w
        kk = k * k
        ckk = c * kk
        oh = (h + 2 * pad - k) // stride + 1
        ow = (w + 2 * pad - k) // stride + 1
        ohw = oh * ow
        row = _gdiv(e, off, ckk, I64)
        col = _gmod(e, off, ckk, I64)
        img = _gdiv(e, row, ohw, I64)
        pos = _gmod(e, row, ohw, I64)
        ph = _gmul(e, _gdiv(e, pos, ow, I64), stride, I64)
        pw = _gmul(e, _gmod(e, pos, ow, I64), stride, I64)
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
    if op == GA_IM2COL_NHWC:
        n, c, h, w, k, pad, stride = (params[0], params[1], params[2],
                                      params[3], params[4], params[5],
                                      params[6])
        kk = k * k
        kkc = kk * c
        oh = (h + 2 * pad - k) // stride + 1
        ow = (w + 2 * pad - k) // stride + 1
        ohw = oh * ow
        row = _gdiv(e, off, kkc, I64)
        col = _gmod(e, off, kkc, I64)
        ci = _gmod(e, col, c, I64)
        rk = _gdiv(e, col, c, I64)
        img = _gdiv(e, row, ohw, I64)
        pos = _gmod(e, row, ohw, I64)
        ph = _gmul(e, _gdiv(e, pos, ow, I64), stride, I64)
        pw = _gmul(e, _gmod(e, pos, ow, I64), stride, I64)
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
        pix = _gbin(e, 'addi',
                    _gbin(e, 'addi', _gmul(e, img, h * w, I64),
                          _gmul(e, ih, w, I64), I64), iw, I64)
        srcs.append(_gbin(e, 'addi', _gmul(e, pix, c, I64), ci, I64))
        return srcs, ok
    if op == GA_IM2COL2:
        c, h, w, kh, kw = params[1], params[2], params[3], params[4], params[5]
        sh, sw, ph, pw = params[6], params[7], params[8], params[9]
        dh, dw = params[10], params[11]
        hw = h * w
        kk = kh * kw
        ckk = c * kk
        ow = conv_out(w, kw, sw, pw, dw)
        ohw = conv_out(h, kh, sh, ph, dh) * ow
        row = _gdiv(e, off, ckk, I64)
        col = _gmod(e, off, ckk, I64)
        img = _gdiv(e, row, ohw, I64)
        pos = _gmod(e, row, ohw, I64)
        ch = _gdiv(e, col, kk, I64)
        rk = _gmod(e, col, kk, I64)
        ih = _gaddc(e, _gbin(e, 'addi', _gmul(e, _gdiv(e, pos, ow, I64), sh,
                                              I64),
                             _gmul(e, _gdiv(e, rk, kw, I64), dh, I64), I64),
                    -ph, I64)
        iw = _gaddc(e, _gbin(e, 'addi', _gmul(e, _gmod(e, pos, ow, I64), sw,
                                              I64),
                             _gmul(e, _gmod(e, rk, kw, I64), dw, I64), I64),
                    -pw, I64)
        zero = _gconst(e, 0, I64)
        ok = _gbin(e, 'andi',
                   _gbin(e, 'andi', _gcmp(e, 'sge', ih, zero, I64),
                         _gcmp(e, 'slt', ih, _gconst(e, h, I64), I64), I1),
                   _gbin(e, 'andi', _gcmp(e, 'sge', iw, zero, I64),
                         _gcmp(e, 'slt', iw, _gconst(e, w, I64), I64), I1), I1)
        base = _gbin(e, 'addi', _gmul(e, img, c * hw, I64),
                     _gmul(e, ch, hw, I64), I64)
        srcs.append(_gbin(e, 'addi', base,
                          _gbin(e, 'addi', _gmul(e, ih, w, I64), iw, I64),
                          I64))
        return srcs, ok
    if op == GA_IM2COL_T:
        c, h, w, kh, kw = params[1], params[2], params[3], params[4], params[5]
        sh, sw, ph, pw = params[6], params[7], params[8], params[9]
        dh, dw, oh, ow = params[10], params[11], params[12], params[13]
        hw = h * w
        kk = kh * kw
        ckk = c * kk
        row = _gdiv(e, off, ckk, I64)
        col = _gmod(e, off, ckk, I64)
        img = _gdiv(e, row, oh * ow, I64)
        pos = _gmod(e, row, oh * ow, I64)
        ch = _gdiv(e, col, kk, I64)
        rk = _gmod(e, col, kk, I64)
        ty = _gaddc(e, _gbin(e, 'subi', _gdiv(e, pos, ow, I64),
                             _gmul(e, _gdiv(e, rk, kw, I64), dh, I64), I64),
                    ph, I64)
        tx = _gaddc(e, _gbin(e, 'subi', _gmod(e, pos, ow, I64),
                             _gmul(e, _gmod(e, rk, kw, I64), dw, I64), I64),
                    pw, I64)
        iy = _gdiv(e, ty, sh, I64)
        ix = _gdiv(e, tx, sw, I64)
        zero = _gconst(e, 0, I64)
        ok = _gbin(e, 'andi',
                   _gbin(e, 'andi', _gcmp(e, 'sge', ty, zero, I64),
                         _gcmp(e, 'slt', iy, _gconst(e, h, I64), I64), I1),
                   _gbin(e, 'andi', _gcmp(e, 'sge', tx, zero, I64),
                         _gcmp(e, 'slt', ix, _gconst(e, w, I64), I64), I1), I1)
        ok = _gbin(e, 'andi', ok,
                   _gbin(e, 'andi',
                         _gcmp(e, 'eq', _gmod(e, ty, sh, I64), zero, I64),
                         _gcmp(e, 'eq', _gmod(e, tx, sw, I64), zero, I64),
                         I1), I1)
        base = _gbin(e, 'addi', _gmul(e, img, c * hw, I64),
                     _gmul(e, ch, hw, I64), I64)
        srcs.append(_gbin(e, 'addi', base,
                          _gbin(e, 'addi', _gmul(e, iy, w, I64), ix, I64),
                          I64))
        return srcs, ok
    if op == GA_MAXPOOL_NHWC:
        n, c, h, w, k, stride, pad = (params[0], params[1], params[2],
                                      params[3], params[4], params[5],
                                      params[6])
        oh = (h + 2 * pad - k) // stride + 1
        ow = (w + 2 * pad - k) // stride + 1
        ohw = oh * ow
        ci = _gmod(e, off, c, I64)
        row = _gdiv(e, off, c, I64)
        img = _gdiv(e, row, ohw, I64)
        pos = _gmod(e, row, ohw, I64)
        ph = _gmul(e, _gdiv(e, pos, ow, I64), stride, I64)
        pw = _gmul(e, _gmod(e, pos, ow, I64), stride, I64)
        base = _gmul(e, img, h * w, I64)
        for a in range(k):
            ih = _gclamp(e, _gaddc(e, ph, a - pad, I64), h - 1, I64)
            rh = _gbin(e, 'addi', base, _gmul(e, ih, w, I64), I64)
            for b in range(k):
                iw = _gclamp(e, _gaddc(e, pw, b - pad, I64), w - 1, I64)
                pix = _gbin(e, 'addi', rh, iw, I64)
                srcs.append(_gbin(e, 'addi', _gmul(e, pix, c, I64), ci, I64))
        return srcs, ''
    if op == GA_COL2CHW:
        n, hw, o = params[0], params[1], params[2]
        img = _gdiv(e, off, o * hw, I64)
        rem = _gmod(e, off, o * hw, I64)
        ch = _gdiv(e, rem, hw, I64)
        pos = _gmod(e, rem, hw, I64)
        row = _gbin(e, 'addi', _gmul(e, img, hw, I64), pos, I64)
        srcs.append(_gbin(e, 'addi', _gmul(e, row, o, I64), ch, I64))
        return srcs, ''
    if op == GA_ROTHALF:
        dh = params[0]
        half = dh // 2
        blk = _gdiv(e, off, dh, I64)
        c = _gmod(e, off, dh, I64)
        rc = _gmod(e, _gaddc(e, c, half, I64), dh, I64)
        srcs.append(_gbin(e, 'addi', _gmul(e, blk, dh, I64), rc, I64))
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
    n, c, h, w, k, stride, pad = (params[0], params[1], params[2], params[3],
                                  params[4], params[5], params[6])
    oh = (h + 2 * pad - k) // stride + 1
    ow = (w + 2 * pad - k) // stride + 1
    pw = _gmul(e, _gmod(e, off, ow, I64), stride, I64)
    q = _gdiv(e, off, ow, I64)
    ph = _gmul(e, _gmod(e, q, oh, I64), stride, I64)
    plane = _gdiv(e, q, oh, I64)
    base = _gmul(e, plane, h * w, I64)
    for a in range(k):
        ih = _gclamp(e, _gaddc(e, ph, a - pad, I64), h - 1, I64)
        for b in range(k):
            iw = _gclamp(e, _gaddc(e, pw, b - pad, I64), w - 1, I64)
            srcs.append(_gbin(e, 'addi', base,
                              _gbin(e, 'addi', _gmul(e, ih, w, I64), iw,
                                    I64), I64))
    return srcs, ''

def to_ttir_rowgather(params, name, dtype):
    BLOCK = config.flat
    S = STORE_TYPE[dtype]
    cols = params[0]
    T, TS, P, I32, I64, I1 = _tile_types(BLOCK, S, S)
    e = Emitter()
    e.add('module {')
    e.add('  tt.func public @%s(%%in: !tt.ptr<%s>, %%idx: !tt.ptr<%s>, '
          '%%out: !tt.ptr<%s>, %%n: i64, %%c: i64) '
          'attributes {noinline = false} {' % (name, S, S, S))
    e.add('    %%zero = arith.constant dense<0.0> : %s' % T)
    _flat_prologue(e.lines, BLOCK, I32, I64)
    row = _gdiv(e, '%offs64', cols, I64)
    col = _gmod(e, '%offs64', cols, I64)
    e.add('    %%pidx = tt.splat %%idx : !tt.ptr<%s> -> %s' % (S, P))
    q = e.tmp()
    e.add('    %s = tt.addptr %%pidx, %s : %s, %s' % (q, row, P, I64))
    iv = e.tmp()
    e.add('    %s = tt.load %s, %%mask, %%zero : %s' % (iv, q, P))
    ii = e.tmp()
    e.add('    %s = arith.fptosi %s : %s to %s' % (ii, iv, T, I64))
    src = _gbin(e, 'addi', _gbin(e, 'muli', ii, _gconst(e, cols, I64), I64),
                col, I64)
    e.add('    %%pin = tt.splat %%in : !tt.ptr<%s> -> %s' % (S, P))
    qs = e.tmp()
    e.add('    %s = tt.addptr %%pin, %s : %s, %s' % (qs, src, P, I64))
    v = e.tmp()
    e.add('    %s = tt.load %s, %%mask, %%zero : %s' % (v, qs, P))
    e.add('    %%pout = tt.splat %%out : !tt.ptr<%s> -> %s' % (S, P))
    e.add('    %%qout = tt.addptr %%pout, %%offs : %s, %s' % (P, I32))
    e.add('    tt.store %%qout, %s, %%mask : %s' % (v, P))
    e.add('    tt.return')
    e.add('  }')
    e.add('}')
    return '\n'.join(e.lines) + '\n'

def to_ttir_gather(op, params, name, dtype, fill=0.0):
    if op == GA_ROWS:
        return to_ttir_rowgather(params, name, dtype)
    if op == GA_STRIDED:
        return to_ttir_strided(params, fill, name, dtype)
    if op == GA_TAKE:
        return to_ttir_take(params, name, dtype)
    if op == GA_POOL:
        return to_ttir_pool(params, name, dtype)
    BLOCK = config.flat
    S = STORE_TYPE[dtype]
    C = COMP_TYPE[dtype]
    half = S != C
    T, TS, P, I32, I64, I1 = _tile_types(BLOCK, S, S)
    e = Emitter()
    e.add('module {')
    e.add('  tt.func public @%s(%%in: !tt.ptr<%s>, %%out: !tt.ptr<%s>, '
          '%%n: i64, %%c: i64) attributes {noinline = false} {' % (name, S, S))
    e.add('    %%zero = arith.constant dense<0.0> : %s' % T)
    _flat_prologue(e.lines, BLOCK, I32, I64)
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


def _float_text(v, C):
    wide = C == 'f64'
    if v != v:
        return '0x7FF8000000000000' if wide else '0x7FC00000'
    if v == INFINITY:
        return '0x7FF0000000000000' if wide else '0x7F800000'
    if v == -INFINITY:
        return '0xFFF0000000000000' if wide else '0xFF800000'
    return _fliteral(v)


def _sconst(e, v, ty):
    r = e.tmp()
    e.add('    %s = arith.constant %d : %s' % (r, v, ty))
    return r


def _splat(e, v, ty, tty):
    r = e.tmp()
    e.add('    %s = tt.splat %s : %s -> %s' % (r, v, ty, tty))
    return r


def _load(e, ptr, idx, mask, S, P, IX):
    p = _splat(e, ptr, '!tt.ptr<%s>' % S, P)
    q = e.tmp()
    e.add('    %s = tt.addptr %s, %s : %s, %s' % (q, p, idx, P, IX))
    v = e.tmp()
    e.add('    %s = tt.load %s, %s, %%zero : %s' % (v, q, mask, P))
    return v


def _ext(e, v, T, TC, half):
    if not half:
        return v
    r = e.tmp()
    e.add('    %s = arith.extf %s : %s to %s' % (r, v, T, TC))
    return r


def _and(e, a, b, I1):
    if not a:
        return b
    return _gbin(e, 'andi', a, b, I1)


def _store(e, v, S, P, I32):
    e.add('    %%pout = tt.splat %%out : !tt.ptr<%s> -> %s' % (S, P))
    e.add('    %%qout = tt.addptr %%pout, %%offs : %s, %s' % (P, I32))
    e.add('    tt.store %%qout, %s, %%mask : %s' % (v, P))
    e.add('    tt.return')
    e.add('  }')
    e.add('}')
    return '\n'.join(e.lines) + '\n'


def _header(e, name, ins, S):
    args = []
    for i in range(len(ins)):
        args.append('%%%s: !tt.ptr<%s>' % (ins[i], S))
    e.add('module {')
    e.add('  tt.func public @%s(%s, %%out: !tt.ptr<%s>, %%n: i64, %%c: i64) '
          'attributes {noinline = false} {' % (name, ', '.join(args), S))


def strided_fits_i32(params):
    nsrc = params[0]
    rank = params[1]
    n = 1
    for j in range(rank):
        n *= params[2 + j]
    if n >= 1 << 31:
        return False
    for s in range(nsrc):
        base = 2 + rank + 3 * rank * s
        ext = 0
        for j in range(rank):
            ext += (params[2 + j] - 1) * abs(params[base + j])
        if ext >= 1 << 31:
            return False
    return True


def _coord(e, index, shape, j, off, IX):
    if not index[j]:
        inner = 1
        for q in range(j + 1, len(shape)):
            inner *= shape[q]
        v = off
        if inner != 1:
            v = _gdiv(e, v, inner, IX)
        if j > 0:
            v = _gmod(e, v, shape[j], IX)
        index[j] = v
    return index[j]


def to_ttir_strided(params, fill, name, dtype):
    """params = [nsrc, rank, shape..., then per source strides..., lo...,
    hi...].  Element i of the output, at coordinates x, is source s at
    sum(x[j] * stride[j]) for the first s with lo <= x < hi in every
    dimension, fill if there is none.  Each source's base offset is in its
    pointer."""
    BLOCK = config.flat
    S = STORE_TYPE[dtype]
    C = COMP_TYPE[dtype]
    T, TS, P, I32, I64, I1 = _tile_types(BLOCK, S, S)
    nsrc = params[0]
    rank = params[1]
    shape = []
    for j in range(rank):
        shape.append(params[2 + j])
    IX, off = I64, '%offs64'
    if strided_fits_i32(params):
        IX, off = I32, '%offs'
    e = Emitter()
    ins = []
    for s in range(nsrc):
        ins.append('in%d' % s)
    _header(e, name, ins, S)
    e.add('    %%zero = arith.constant dense<0.0> : %s' % T)
    _flat_prologue(e.lines, BLOCK, I32, I64)
    index = [''] * rank
    vals = []
    oks = []
    for s in range(nsrc):
        base = 2 + rank + 3 * rank * s
        addr = ''
        ok = ''
        for j in range(rank):
            st = params[base + j]
            lo = params[base + rank + j]
            hi = params[base + 2 * rank + j]
            if st != 0:
                x = _coord(e, index, shape, j, off, IX)
                if st != 1:
                    x = _gmul(e, x, st, IX)
                addr = _gbin(e, 'addi', addr, x, IX) if addr else x
            if lo > 0:
                ok = _and(e, ok, _gcmp(e, 'sge', _coord(e, index, shape, j,
                                                        off, IX),
                                       _gconst(e, lo, IX), IX), I1)
            if hi < shape[j]:
                ok = _and(e, ok, _gcmp(e, 'slt', _coord(e, index, shape, j,
                                                        off, IX),
                                       _gconst(e, hi, IX), IX), I1)
        if not addr:
            addr = _gconst(e, 0, IX)
        mask = _and(e, ok, '%mask', I1)
        vals.append(_load(e, '%%in%d' % s, addr, mask, S, P, IX))
        oks.append(ok)
    acc = '%zero'
    if fill != 0.0:
        TC = 'tensor<%dx%s>' % (BLOCK, C)
        e.add('    %%fillc = arith.constant dense<%s> : %s'
              % (_float_text(fill, C), TC))
        acc = '%fillc'
        if S != C:
            e.add('    %%fill = arith.truncf %%fillc : %s to %s' % (TC, T))
            acc = '%fill'
    for s in range(nsrc - 1, -1, -1):
        if oks[s]:
            r = e.tmp()
            e.add('    %s = arith.select %s, %s, %s : %s, %s'
                  % (r, oks[s], vals[s], acc, I1, T))
            acc = r
        else:
            acc = vals[s]
    return _store(e, acc, S, P, I32)


def to_ttir_take(params, name, dtype):
    """out[o, j, i] = in[o, idx[o*so + j*sj + i*si], i] over in as
    [outer, d, inner], zero where the index is out of range."""
    d, k, inner, so, sj, si = (params[0], params[1], params[2], params[3],
                               params[4], params[5])
    BLOCK = config.flat
    S = STORE_TYPE[dtype]
    C = COMP_TYPE[dtype]
    T, TS, P, I32, I64, I1 = _tile_types(BLOCK, S, S)
    TC = 'tensor<%dx%s>' % (BLOCK, C)
    e = Emitter()
    _header(e, name, ['in', 'idx'], S)
    e.add('    %%zero = arith.constant dense<0.0> : %s' % T)
    _flat_prologue(e.lines, BLOCK, I32, I64)
    off = '%offs64'
    xi = _gmod(e, off, inner, I64)
    q = _gdiv(e, off, inner, I64)
    xj = _gmod(e, q, k, I64)
    xo = _gdiv(e, q, k, I64)
    pos = _gconst(e, 0, I64)
    if so:
        pos = _gbin(e, 'addi', pos, _gmul(e, xo, so, I64), I64)
    if sj:
        pos = _gbin(e, 'addi', pos, _gmul(e, xj, sj, I64), I64)
    if si:
        pos = _gbin(e, 'addi', pos, _gmul(e, xi, si, I64), I64)
    iv = _ext(e, _load(e, '%idx', pos, '%mask', S, P, I64), T, TC, S != C)
    ii = e.tmp()
    e.add('    %s = arith.fptosi %s : %s to %s' % (ii, iv, TC, I64))
    ok = _gbin(e, 'andi', _gcmp(e, 'sge', ii, _gconst(e, 0, I64), I64),
               _gcmp(e, 'slt', ii, _gconst(e, d, I64), I64), I1)
    src = _gbin(e, 'addi', _gmul(e, _gbin(e, 'addi', _gmul(e, xo, d, I64), ii,
                                          I64), inner, I64), xi, I64)
    v = _load(e, '%in', src, _gbin(e, 'andi', '%mask', ok, I1), S, P, I64)
    return _store(e, v, S, P, I32)


def to_ttir_pool(params, name, dtype):
    """Average, adaptive average and depthwise convolution over NCHW planes:
    each output element sums a window of its plane in an scf.for.

    params = [mode, h, w, oh, ow, kh, kw, sh, sw, ph, pw, dh, dw, cout,
    mult, flags]: cout and mult (output channels, outputs per input channel)
    are for depthwise, whose weight is [cin, kh*kw, mult] and whose flags
    bit 0 says there is a bias; for average pooling that bit is
    count_include_pad.  Adaptive windows come from oh/ow alone, kh and kw are
    the largest of them."""
    mode, h, w, oh, ow = params[0], params[1], params[2], params[3], params[4]
    kh, kw, sh, sw = params[5], params[6], params[7], params[8]
    ph, pw, dh, dw = params[9], params[10], params[11], params[12]
    cout, mult, flags = params[13], params[14], params[15]
    BLOCK = config.flat
    S = STORE_TYPE[dtype]
    C = COMP_TYPE[dtype]
    half = S != C
    T, TS, P, I32, I64, I1 = _tile_types(BLOCK, S, S)
    TC = 'tensor<%dx%s>' % (BLOCK, C)
    X = I64
    conv = mode == POOL_DEPTHWISE
    bias = conv and (flags & 1) != 0
    ins = ['in']
    if conv:
        ins.append('wt')
        if bias:
            ins.append('bias')
    e = Emitter()
    _header(e, name, ins, S)
    e.add('    %%zero = arith.constant dense<0.0> : %s' % T)
    e.add('    %%zc = arith.constant dense<0.0> : %s' % TC)
    _flat_prologue(e.lines, BLOCK, I32, I64)
    off = '%offs64'
    owi = _gmod(e, off, ow, X)
    t = _gdiv(e, off, ow, X)
    ohi = _gmod(e, t, oh, X)
    plane = _gdiv(e, t, oh, X)
    oc = ''
    wbase = ''
    if conv:
        oc = _gmod(e, plane, cout, X)
        ci = oc
        if mult > 1:
            ci = _gdiv(e, oc, mult, X)
        plane = _gbin(e, 'addi', _gmul(e, _gdiv(e, plane, cout, X),
                                       cout // mult, X), ci, X)
        wbase = _gmul(e, ci, kh * kw * mult, X)
        if mult > 1:
            wbase = _gbin(e, 'addi', wbase, _gmod(e, oc, mult, X), X)
    base = _gmul(e, plane, h * w, X)
    cnt = ''
    if mode == POOL_ADAPTIVE:
        hs = _gdiv(e, _gmul(e, ohi, h, X), oh, X)
        he = _gdiv(e, _gaddc(e, _gmul(e, ohi, h, X), h + oh - 1, X), oh, X)
        ws = _gdiv(e, _gmul(e, owi, w, X), ow, X)
        we = _gdiv(e, _gaddc(e, _gmul(e, owi, w, X), w + ow - 1, X), ow, X)
        cnt = _gbin(e, 'muli', _gbin(e, 'subi', he, hs, X),
                    _gbin(e, 'subi', we, ws, X), X)
    else:
        hs = _gaddc(e, _gmul(e, ohi, sh, X), -ph, X)
        ws = _gaddc(e, _gmul(e, owi, sw, X), -pw, X)
        he = we = ''
        if mode == POOL_AVG:
            he = _gbin(e, 'minsi', _gaddc(e, hs, kh, X),
                       _gconst(e, h + ph, X), X)
            we = _gbin(e, 'minsi', _gaddc(e, ws, kw, X),
                       _gconst(e, w + pw, X), X)
            if flags & 1:
                lh, lw, uh, uw = hs, ws, he, we
            else:
                zero = _gconst(e, 0, X)
                lh = _gbin(e, 'maxsi', hs, zero, X)
                lw = _gbin(e, 'maxsi', ws, zero, X)
                uh = _gbin(e, 'minsi', he, _gconst(e, h, X), X)
                uw = _gbin(e, 'minsi', we, _gconst(e, w, X), X)
            cnt = _gbin(e, 'muli', _gbin(e, 'subi', uh, lh, X),
                        _gbin(e, 'subi', uw, lw, X), X)
    lb = _sconst(e, 0, 'i32')
    ub = _sconst(e, kh * kw, 'i32')
    step = _sconst(e, 1, 'i32')
    res = e.tmp()
    iv = e.tmp()
    acc = e.tmp()
    e.add('    %s = scf.for %s = %s to %s step %s iter_args(%s = %%zc) -> '
          '(%s) : i32 {' % (res, iv, lb, ub, step, acc, TC))
    a = _gbin(e, 'divsi', iv, _sconst(e, kw, 'i32'), 'i32')
    b = _gbin(e, 'remsi', iv, _sconst(e, kw, 'i32'), 'i32')
    a = _gbin(e, 'muli', a, _sconst(e, dh, 'i32'), 'i32')
    b = _gbin(e, 'muli', b, _sconst(e, dw, 'i32'), 'i32')
    a64 = e.tmp()
    e.add('    %s = arith.extsi %s : i32 to i64' % (a64, a))
    b64 = e.tmp()
    e.add('    %s = arith.extsi %s : i32 to i64' % (b64, b))
    ih = _gbin(e, 'addi', hs, _splat(e, a64, 'i64', X), X)
    iw = _gbin(e, 'addi', ws, _splat(e, b64, 'i64', X), X)
    if mode == POOL_ADAPTIVE:
        ok = _gbin(e, 'andi', _gcmp(e, 'slt', ih, he, X),
                   _gcmp(e, 'slt', iw, we, X), I1)
    else:
        zero = _gconst(e, 0, X)
        ok = _gbin(e, 'andi',
                   _gbin(e, 'andi', _gcmp(e, 'sge', ih, zero, X),
                         _gcmp(e, 'slt', ih, _gconst(e, h, X), X), I1),
                   _gbin(e, 'andi', _gcmp(e, 'sge', iw, zero, X),
                         _gcmp(e, 'slt', iw, _gconst(e, w, X), X), I1), I1)
    src = _gbin(e, 'addi', base, _gbin(e, 'addi', _gmul(e, ih, w, X), iw, X),
                X)
    v = _ext(e, _load(e, '%in', src, _gbin(e, 'andi', '%mask', ok, I1), S, P,
                      X), T, TC, half)
    if conv:
        r = _gbin(e, 'muli', iv, _sconst(e, mult, 'i32'), 'i32')
        r64 = e.tmp()
        e.add('    %s = arith.extsi %s : i32 to i64' % (r64, r))
        wi = _gbin(e, 'addi', wbase, _splat(e, r64, 'i64', X), X)
        wv = _ext(e, _load(e, '%wt', wi, '%mask', S, P, X), T, TC, half)
        v = _gbin(e, 'mulf', v, wv, TC)
    nxt = _gbin(e, 'addf', acc, v, TC)
    e.add('      scf.yield %s : %s' % (nxt, TC))
    e.add('    }')
    out = res
    if cnt:
        cf = e.tmp()
        e.add('    %s = arith.sitofp %s : %s to %s' % (cf, cnt, X, TC))
        out = _gbin(e, 'divf', out, cf, TC)
    if bias:
        bv = _ext(e, _load(e, '%bias', oc, '%mask', S, P, X), T, TC, half)
        out = _gbin(e, 'addf', out, bv, TC)
    if half:
        r = e.tmp()
        e.add('    %s = arith.truncf %s : %s to %s' % (r, out, TC, T))
        out = r
    return _store(e, out, S, P, I32)
