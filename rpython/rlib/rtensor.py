import os
from rpython.rlib import jit
from rpython.rtyper.lltypesystem import lltype, rffi
from rpython.translator.tool.cbuild import ExternalCompilationInfo

HOSTARRAY = lltype.GcArray(lltype.Float)
TENSOR = lltype.GcStruct('TENSOR', ('size', lltype.Signed),
                         ('dptr', lltype.Signed),
                         ('host', lltype.Ptr(HOSTARRAY)))
TENSORPTR = lltype.Ptr(TENSOR)
NULLTENSOR = lltype.nullptr(TENSOR)

ADD, MUL, RELU, SUM = 0, 1, 2, 3
ARITY = [2, 2, 1, 1]
NAMES = ['add', 'mul', 'relu', 'sum']
MAX_INPUTS = 3

NODE = lltype.Struct('TENSOR_NODE', ('opcode', lltype.Signed),
                     ('a', lltype.Signed), ('b', lltype.Signed))
NODEARRAY = lltype.GcArray(NODE)
KERNEL = lltype.GcStruct('TENSOR_KERNEL', ('ninputs', lltype.Signed),
                         ('nodes', lltype.Ptr(NODEARRAY)),
                         ('fn', lltype.Signed),
                         ('sumroot', lltype.Signed),
                         ('threads', lltype.Signed),
                         ('shared', lltype.Signed),
                         ('nextra', lltype.Signed))
KERNELPTR = lltype.Ptr(KERNEL)

def new_tensor(n):
    t = lltype.malloc(TENSOR)
    t.size = n
    t.dptr = 0
    t.host = lltype.malloc(HOSTARRAY, n)
    return t

def device_tensor(n, dptr):
    t = lltype.malloc(TENSOR)
    t.size = n
    t.dptr = dptr
    t.host = lltype.nullptr(HOSTARRAY)
    return t

def from_list(values):
    t = new_tensor(len(values))
    for i in range(len(values)):
        t.host[i] = values[i]
    return t

def host(t):
    if not t.host:
        t.host = lltype.malloc(HOSTARRAY, t.size)
        download(t.dptr, t.host)
    return t.host

def dev(t):
    if t.dptr == 0:
        t.dptr = upload(t.host)
    return t.dptr

def eval_op(opcode, a, b):
    kernel = single_kernel(opcode)
    if kernel.fn != 0:
        inputs = [a]
        if ARITY[opcode] == 2:
            inputs.append(b)
        r = launch_gpu(kernel, inputs)
        if r:
            return r
    return eval_op_cpu(opcode, a, b)

def eval_op_cpu(opcode, a, b):
    ha = host(a)
    n = a.size
    if opcode == SUM:
        r = new_tensor(1)
        s = 0.0
        for i in range(n):
            s += ha[i]
        r.host[0] = s
        return r
    r = new_tensor(n)
    hr = r.host
    if opcode == RELU:
        for i in range(n):
            v = ha[i]
            hr[i] = v if v > 0.0 else 0.0
    else:
        hb = host(b)
        if opcode == ADD:
            for i in range(n):
                hr[i] = ha[i] + hb[i]
        else:
            for i in range(n):
                hr[i] = ha[i] * hb[i]
    return r

class SingleKernels(object):
    def __init__(self):
        self.kernels = [_empty_kernel() for i in range(4)]

def _empty_kernel():
    k = lltype.malloc(KERNEL)
    k.ninputs = 0
    k.nodes = lltype.malloc(NODEARRAY, 0)
    k.fn = k.sumroot = k.threads = k.shared = k.nextra = 0
    return k
single_kernels = SingleKernels()

def single_kernel(opcode):
    return single_kernels.kernels[opcode]

def init_device():
    try:
        config.block = int(_env('RTENSOR_BLOCK', '4096'))
        config.num_warps = int(_env('RTENSOR_WARPS', '8'))
    except ValueError:
        pass
    for opcode in range(4):
        opcodes = []
        opcodes.append(opcode)
        lefts = []
        lefts.append(0)
        rights = []
        rights.append(1 if ARITY[opcode] == 2 else -1)
        single_kernels.kernels[opcode] = build_kernel(ARITY[opcode], opcodes,
                                                     lefts, rights)

@jit.oopspec("tensor.add(a, b)")
def tensor_add(a, b):
    return eval_op(ADD, a, b)

@jit.oopspec("tensor.mul(a, b)")
def tensor_mul(a, b):
    return eval_op(MUL, a, b)

@jit.oopspec("tensor.relu(a)")
def tensor_relu(a):
    return eval_op(RELU, a, NULLTENSOR)

@jit.oopspec("tensor.sum(a)")
def tensor_sum(a):
    return eval_op(SUM, a, NULLTENSOR)

@jit.oopspec("tensor.size(a)")
def tensor_size(a):
    return a.size

@jit.dont_look_inside
def tensor_force(a):
    return a

@jit.elidable
def tensor_item(a):
    return host(a)[0]

class KernelCache(object):
    def __init__(self):
        self.kernels = {}
kernel_cache = KernelCache()

def cached_kernel(key):
    return kernel_cache.kernels.get(key, lltype.nullptr(KERNEL))

def cache_kernel(key, kernel):
    kernel_cache.kernels[key] = kernel

def new_kernel(ninputs, nnodes):
    kernel = lltype.malloc(KERNEL)
    kernel.ninputs = ninputs
    kernel.nodes = lltype.malloc(NODEARRAY, nnodes)
    kernel.fn = kernel.sumroot = kernel.threads = kernel.shared = kernel.nextra = 0
    return kernel

def set_node(kernel, i, opcode, a, b):
    node = kernel.nodes[i]
    node.opcode = opcode
    node.a = a
    node.b = b

def finish_kernel(kernel):
    n = len(kernel.nodes)
    kernel.sumroot = int(n > 0 and kernel.nodes[n - 1].opcode == SUM)
    kernel.fn = compile_gpu(kernel)
    return kernel

def build_kernel(ninputs, opcodes, lefts, rights):
    kernel = new_kernel(ninputs, len(opcodes))
    for i in range(len(opcodes)):
        set_node(kernel, i, opcodes[i], lefts[i], rights[i])
    return finish_kernel(kernel)

def launch(kernel, a, b, c):
    values = [a]
    if kernel.ninputs > 1:
        values.append(b)
    if kernel.ninputs > 2:
        values.append(c)
    if kernel.fn != 0:
        r = launch_gpu(kernel, values)
        if r:
            return r
    nodes = kernel.nodes
    for i in range(len(nodes)):
        node = nodes[i]
        opcode = node.opcode
        assert opcode >= 0
        right = values[node.b] if node.b >= 0 else NULLTENSOR
        values.append(eval_op(opcode, values[node.a], right))
    return values[len(values) - 1]

def to_tile_ir(kernel, name, n):
    tile = 'tile<%dxf64>' % n
    params = ', '.join(['%%in%d: !cuda_tile.ptr<f64>' % i
                        for i in range(kernel.ninputs)] +
                       ['%out: !cuda_tile.ptr<f64>'])
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
        else:
            result = 'tile<1xf64>'
            lines.append('    %%v%d = cuda_tile.reduce add %%v%d : %s -> %s'
                         % (v, node.a, tile, result))
    lines.append('    cuda_tile.store_ptr_tko %%out, %%v%d : %s'
                 % (kernel.ninputs + len(nodes) - 1, result))
    lines.append('    cuda_tile.return')
    lines.append('  }')
    lines.append('}')
    return '\n'.join(lines)

class Config(object):
    block = 4096
    num_warps = 8
config = Config()
DOUBLEARRAY = rffi.CArray(rffi.DOUBLE)
SIGNEDARRAY = rffi.CArray(lltype.Signed)
_here = os.path.dirname(os.path.abspath(__file__))
CUDA_HOME = os.environ.get('CUDA_HOME', '/usr/local/cuda')
eci = ExternalCompilationInfo(
    include_dirs=[os.path.join(CUDA_HOME, 'include')],
    separate_module_files=[os.path.join(_here, 'rtensor_cuda.c')],
    post_include_bits=["""
RPY_EXTERN int rt_cuda_available(void);
RPY_EXTERN long rt_cuda_load(const char *ptx, const char *name);
RPY_EXTERN long rt_cuda_alloc(long n);
RPY_EXTERN long rt_cuda_upload(double *host, long n);
RPY_EXTERN int rt_cuda_download(long dptr, double *host, long n);
RPY_EXTERN void rt_cuda_reset(void);
RPY_EXTERN long rt_cuda_mark(void);
RPY_EXTERN void rt_cuda_release_since(long mark);
RPY_EXTERN void rt_cuda_release_range(long from, long to);
RPY_EXTERN void rt_cuda_sync(void);
RPY_EXTERN int rt_cuda_launch(long fn, long *inputs, int ninputs, long n,
                              long out, int threads, long elems_per_block,
                              int shared, int nextra);
"""],
    libraries=['cuda'])
rt_cuda_load = rffi.llexternal('rt_cuda_load', [rffi.CCHARP, rffi.CCHARP],
                               lltype.Signed, compilation_info=eci,
                                releasegil=False)
rt_cuda_launch = rffi.llexternal(
    'rt_cuda_launch',
    [lltype.Signed, rffi.CArrayPtr(lltype.Signed), rffi.INT, lltype.Signed,
     lltype.Signed, rffi.INT, lltype.Signed, rffi.INT, rffi.INT],
    rffi.INT, compilation_info=eci,
                                releasegil=False)
rt_cuda_available = rffi.llexternal('rt_cuda_available', [], rffi.INT,
                                    compilation_info=eci,
                                releasegil=False)
rt_cuda_alloc = rffi.llexternal('rt_cuda_alloc', [lltype.Signed],
                                lltype.Signed, compilation_info=eci,
                                releasegil=False)
rt_cuda_upload = rffi.llexternal('rt_cuda_upload',
                                 [rffi.DOUBLEP, lltype.Signed],
                                 lltype.Signed, compilation_info=eci,
                                releasegil=False)
rt_cuda_download = rffi.llexternal('rt_cuda_download',
                                   [lltype.Signed, rffi.DOUBLEP, lltype.Signed],
                                   rffi.INT, compilation_info=eci,
                                releasegil=False)
rt_cuda_reset = rffi.llexternal('rt_cuda_reset', [], lltype.Void,
                                compilation_info=eci,
                                releasegil=False)
rt_cuda_sync = rffi.llexternal('rt_cuda_sync', [], lltype.Void,
                               compilation_info=eci,
                                releasegil=False)

def _env(name, default):
    value = os.environ.get(name)
    if value is None:
        return default
    return value

def gpu_enabled():
    if os.environ.get('RTENSOR_CPU') is not None:
        return False
    return rffi.cast(lltype.Signed, rt_cuda_available()) != 0

def upload(hostarray):
    if not gpu_enabled():
        return 0
    n = len(hostarray)
    buf = lltype.malloc(DOUBLEARRAY, n, flavor='raw')
    for i in range(n):
        buf[i] = hostarray[i]
    dptr = rt_cuda_upload(buf, n)
    lltype.free(buf, flavor='raw')
    return dptr

def download(dptr, hostarray):
    n = len(hostarray)
    buf = lltype.malloc(DOUBLEARRAY, n, flavor='raw')
    rt_cuda_download(dptr, buf, n)
    for i in range(n):
        hostarray[i] = buf[i]
    lltype.free(buf, flavor='raw')

rt_cuda_mark = rffi.llexternal('rt_cuda_mark', [], lltype.Signed,
                               compilation_info=eci, releasegil=False)
rt_cuda_release_since = rffi.llexternal('rt_cuda_release_since',
                                        [lltype.Signed], lltype.Void,
                                        compilation_info=eci, releasegil=False)

rt_cuda_release_range = rffi.llexternal('rt_cuda_release_range',
                                        [lltype.Signed, lltype.Signed], lltype.Void,
                                        compilation_info=eci, releasegil=False)

def reset_device():
    rt_cuda_reset()

def release_range(start, stop):
    rt_cuda_release_range(start, stop)

def device_mark():
    return rt_cuda_mark()

def release_since(mark):
    rt_cuda_release_since(mark)

def sync_device():
    rt_cuda_sync()

def to_ttir(kernel, name):
    nodes = kernel.nodes
    nin = kernel.ninputs
    BLOCK = config.block
    T = 'tensor<%dxf64>' % BLOCK
    P = 'tensor<%dx!tt.ptr<f64>>' % BLOCK
    params = ['%%in%d: !tt.ptr<f64>' % i for i in range(nin)]
    lines = ['module {',
             '  tt.func public @%s(%s, %%out: !tt.ptr<f64>, %%n: i64) '
             'attributes {noinline = false} {' % (name, ', '.join(params)),
             '    %%zero = arith.constant dense<0.0> : %s' % T,
             '    %%bs = arith.constant %d : i32' % BLOCK,
             '    %pid = tt.get_program_id x : i32',
             '    %start = arith.muli %pid, %bs : i32',
             '    %%range = tt.make_range {end = %d : i32, start = 0 : i32} '
             ': tensor<%dxi32>' % (BLOCK, BLOCK),
             '    %%starts = tt.splat %%start : i32 -> tensor<%dxi32>' % BLOCK,
             '    %%offs = arith.addi %%starts, %%range : tensor<%dxi32>' % BLOCK,
             '    %%offs64 = arith.extsi %%offs : tensor<%dxi32> to tensor<%dxi64>'
             % (BLOCK, BLOCK),
             '    %%ns = tt.splat %%n : i64 -> tensor<%dxi64>' % BLOCK,
             '    %%mask = arith.cmpi slt, %%offs64, %%ns : tensor<%dxi64>' % BLOCK]
    for i in range(nin):
        lines.append('    %%p%d = tt.splat %%in%d : !tt.ptr<f64> -> %s' % (i, i, P))
        lines.append('    %%q%d = tt.addptr %%p%d, %%offs : %s, tensor<%dxi32>'
                     % (i, i, P, BLOCK))
        lines.append('    %%v%d = tt.load %%q%d, %%mask, %%zero : %s' % (i, i, P))
    last = nin + len(nodes) - 1
    for k in range(len(nodes)):
        node = nodes[k]
        v = nin + k
        if node.opcode == ADD:
            lines.append('    %%v%d = arith.addf %%v%d, %%v%d : %s'
                         % (v, node.a, node.b, T))
        elif node.opcode == MUL:
            lines.append('    %%v%d = arith.mulf %%v%d, %%v%d : %s'
                         % (v, node.a, node.b, T))
        elif node.opcode == RELU:
            lines.append('    %%c%d = arith.cmpf ogt, %%v%d, %%zero : %s'
                         % (v, node.a, T))
            lines.append('    %%v%d = arith.select %%c%d, %%v%d, %%zero : '
                         'tensor<%dxi1>, %s' % (v, v, node.a, BLOCK, T))
        elif k == len(nodes) - 1:
            lines.append('    %%v%d = "tt.reduce"(%%v%d) <{axis = 0 : i32}> ({'
                         % (v, node.a))
            lines.append('    ^bb0(%x: f64, %y: f64):')
            lines.append('      %r = arith.addf %x, %y : f64')
            lines.append('      tt.reduce.return %r : f64')
            lines.append('    }) : (%s) -> f64' % T)
            lines.append('    %true = arith.constant true')
            lines.append('    %%o = tt.atomic_rmw fadd, acq_rel, gpu, %%out, %%v%d, '
                         '%%true : (!tt.ptr<f64>, f64, i1) -> f64' % v)
        else:
            return ''
    if not kernel.sumroot:
        lines.append('    %%po = tt.splat %%out : !tt.ptr<f64> -> %s' % P)
        lines.append('    %%qo = tt.addptr %%po, %%offs : %s, tensor<%dxi32>'
                     % (P, BLOCK))
        lines.append('    tt.store %%qo, %%v%d, %%mask : %s' % (last, P))
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
    base = _env('TMPDIR', '/tmp') + '/' + name
    _write(base + '.ttir', src)
    cmd = '%s -P %s %s.ttir %s.ptx %s.meta %s %d' % (
        _env('RTENSOR_PYTHON', 'python3'), _here + '/rtensor_triton.py',
        base, base, base, _env('RTENSOR_CC', '86'), config.num_warps)
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

def launch_gpu(kernel, inputs):
    nin = len(inputs)
    n = inputs[0].size
    outlen = 1 if kernel.sumroot else n
    dptrs = lltype.malloc(SIGNEDARRAY, nin, flavor='raw')
    ok = True
    for k in range(nin):
        dptrs[k] = dev(inputs[k])
        if dptrs[k] == 0:
            ok = False
    out = rt_cuda_alloc(outlen) if ok else 0
    if out != 0:
        ok = rffi.cast(lltype.Signed, rt_cuda_launch(
            kernel.fn, dptrs, rffi.cast(rffi.INT, nin), n, out,
            rffi.cast(rffi.INT, kernel.threads), config.block,
            rffi.cast(rffi.INT, kernel.shared),
            rffi.cast(rffi.INT, kernel.nextra))) != 0
    else:
        ok = False
    lltype.free(dptrs, flavor='raw')
    if not ok:
        return NULLTENSOR
    return device_tensor(outlen, out)
