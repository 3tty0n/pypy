from rpython.rtyper.lltypesystem import lltype
from rpython.rtyper.lltypesystem import rffi
import os
from rpython.metatensor.core import (ARITY, AXIS_ALL, F64, KERNEL, NDTYPES, NODEARRAY, NOPCODES, NPARAMS, SHAPEARRAY, SUM, config, is_reduction, param_slot, slot_param, slot_used)
from rpython.metatensor.device import (_env, _here, gpu_enabled, profile, rt_cuda_load, rt_cuda_set_budget)
from rpython.metatensor.ttir import (input_modes, kernel_row_mode, out_modes, row_tile, row_warps, to_tile_ir, to_ttir, to_ttir_gather)

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
    k.modes = 0
    k.outmodes = 0
    k.outputs = lltype.malloc(SHAPEARRAY, 0)
    return k
single_kernels = SingleKernels()

def single_kernel(opcode, p, dtype):
    return single_kernels.kernels[(dtype * NOPCODES + opcode) * NPARAMS +
                                  param_slot(opcode, p)]

def init_device():
    try:
        config.block = int(_env('RTENSOR_BLOCK', '4096'))
        config.flat = int(_env('RTENSOR_FLAT_BLOCK', '4096'))
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
    kernel.modes = 0
    kernel.outmodes = 0
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
        kernel.modes = cached.modes
        kernel.outmodes = cached.outmodes
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
def packed_out_modes(kernel):
    modes = out_modes(kernel)
    packed = 0
    for i in range(len(modes)):
        packed |= modes[i] << (2 * i)
    return packed

def packed_modes(kernel):
    modes = input_modes(kernel)
    packed = 0
    for i in range(len(modes)):
        packed |= modes[i] << (2 * i)
    return packed

def finish_kernel(kernel):
    n = len(kernel.nodes)
    kernel.sumroot = int(n > 0 and is_reduction(kernel.nodes[n - 1].opcode))
    kernel.rowmode = int(kernel_row_mode(kernel))
    kernel.modes = packed_modes(kernel)
    kernel.outmodes = packed_out_modes(kernel)
    kernel.fn = compile_gpu(kernel)
    return kernel

def build_kernel(ninputs, opcodes, lefts, rights, params, dtype=F64):
    kernel = new_kernel(ninputs, len(opcodes), dtype)
    for i in range(len(opcodes)):
        set_node(kernel, i, opcodes[i], lefts[i], rights[i], params[i])
    return finish_kernel(kernel)

class Counter(object):
    n = 0
counter = Counter()

def get_cc():
    return _env('RTENSOR_CC', 'auto')

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
def compile_ttir(src, name, warps):
    base = _env('TMPDIR', '/tmp') + '/' + name
    _write(base + '.ttir', src)
    cmd = '%s -P %s %s.ttir %s.ptx %s.meta %s %d' % (
        _env('RTENSOR_PYTHON', 'python3'), _here + '/triton_compile.py',
        base, base, base, get_cc(), warps)
    if os.system(cmd) != 0:
        return 0, 0, 0, 0
    words = _read(base + '.meta').strip().split(' ')
    ptx = _read(base + '.ptx')
    p_ptx = rffi.str2charp(ptx)
    p_name = rffi.str2charp(name)
    fn = rt_cuda_load(p_ptx, p_name)
    rffi.free_charp(p_ptx)
    rffi.free_charp(p_name)
    return fn, int(words[0]), int(words[1]), int(words[2])

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
    fn, threads, shared, nextra = compile_ttir(src, name, warps)
    kernel.threads = threads
    kernel.shared = shared
    kernel.nextra = nextra
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
    fn, threads, shared, nextra = compile_ttir(src, name, config.num_warps)
    return GatherKernel(fn, threads, shared, nextra)
