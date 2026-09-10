from rpython.rlib.rfloat import formatd
from rpython.metatensor.core import (ADD, ARITY, GA_ROWS, AXIS_ALL, BC_L_COL, BC_L_ROW, BC_L_SCALAR, BC_R_COL, BC_R_ROW, BC_R_SCALAR, COMP_NEG_INF, COMP_TYPE, DIV, EQMASK, EXP, GA_COL2CHW, GA_HEADMERGE, GA_HEADSPLIT, GA_IM2COL, GA_IM2COL_NHWC, GA_MAXPOOL_NHWC, GA_ROTHALF, MAXR, MUL, RELU, RELUGRAD, SQRT, STORE_TYPE, SUB, SUM, config, is_reduction, nvals)


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

def to_ttir(kernel, name):
    modes = all_modes(kernel)
    if len(modes) != nvals(kernel) + len(kernel.nodes):
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

def to_ttir_gather(op, params, name, dtype):
    if op == GA_ROWS:
        return to_ttir_rowgather(params, name, dtype)
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
