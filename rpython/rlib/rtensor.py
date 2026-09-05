import os
from rpython.rlib import jit, rgc
from rpython.rtyper.annlowlevel import cast_instance_to_base_ptr
from rpython.rtyper.rclass import OBJECTPTR
from rpython.rtyper.lltypesystem import lltype, rffi
from rpython.translator.tool.cbuild import ExternalCompilationInfo

HOSTARRAY = lltype.GcArray(lltype.Float)
SHAPEARRAY = lltype.GcArray(lltype.Signed)
TENSOR = lltype.GcForwardReference()
TENSORPTR = lltype.Ptr(TENSOR)
TENSORARRAY = lltype.GcArray(TENSORPTR)
TENSOR.become(lltype.GcStruct('TENSOR', ('size', lltype.Signed),
                              ('shape', lltype.Ptr(SHAPEARRAY)),
                              ('dptr', lltype.Signed),
                              ('host', lltype.Ptr(HOSTARRAY)),
                              ('extra', lltype.Ptr(TENSORARRAY)),
                              ('buf', OBJECTPTR)))
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
                         ('nextra', lltype.Signed),
                         ('n', lltype.Signed),
                         ('outputs', lltype.Ptr(SHAPEARRAY)))
KERNELPTR = lltype.Ptr(KERNEL)

class DeviceBuffer(object):
    def __init__(self, dptr, n):
        self.dptr = dptr
        self.n = n

    @rgc.must_be_light_finalizer
    def __del__(self):
        rt_cuda_free(self.dptr, self.n)

def attach_buffer(t, dptr, n):
    t.dptr = dptr
    t.buf = cast_instance_to_base_ptr(DeviceBuffer(dptr, n))

def _shape1(n):
    shape = lltype.malloc(SHAPEARRAY, 1)
    shape[0] = n
    return shape

def new_tensor(n, shape=lltype.nullptr(SHAPEARRAY)):
    t = lltype.malloc(TENSOR)
    t.size = n
    t.shape = shape if shape else _shape1(n)
    t.dptr = 0
    t.host = lltype.malloc(HOSTARRAY, n)
    t.extra = lltype.nullptr(TENSORARRAY)
    t.buf = lltype.nullptr(OBJECTPTR.TO)
    return t

def device_tensor(n, dptr, shape=lltype.nullptr(SHAPEARRAY)):
    t = lltype.malloc(TENSOR)
    t.size = n
    t.shape = shape if shape else _shape1(n)
    t.host = lltype.nullptr(HOSTARRAY)
    t.extra = lltype.nullptr(TENSORARRAY)
    attach_buffer(t, dptr, n)
    return t

def zeros(shape_list):
    n = 1
    for d in shape_list:
        n *= d
    shape = lltype.malloc(SHAPEARRAY, len(shape_list))
    for i in range(len(shape_list)):
        shape[i] = shape_list[i]
    return new_tensor(n, shape)

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
        dptr = upload(t.host)
        if dptr != 0:
            attach_buffer(t, dptr, t.size)
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
    r = new_tensor(n, a.shape)
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
    k.n = 0
    k.outputs = lltype.malloc(SHAPEARRAY, 0)
    return k
single_kernels = SingleKernels()

def single_kernel(opcode):
    return single_kernels.kernels[opcode]

def init_device():
    try:
        config.block = int(_env('RTENSOR_BLOCK', '4096'))
        config.num_warps = int(_env('RTENSOR_WARPS', '8'))
        rt_cuda_set_budget(int(_env('RTENSOR_BUDGET_MB', '256')) << 20)
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

@jit.oopspec("tensor.shape(a, axis)")
def tensor_shape(a, axis):
    return a.shape[axis]


class SizePolicy(object):
    _immutable_fields_ = ['static?']
    def __init__(self):
        self.static = True
        self.seen = []
policy = SizePolicy()
MAX_STATIC_SIZES = 3

@jit.elidable
def note_size(n):
    if n not in policy.seen:
        policy.seen.append(n)
        if len(policy.seen) > MAX_STATIC_SIZES:
            policy.static = False
    return n

def size(t):
    n = tensor_size(t)
    if policy.static:
        n = jit.promote(n)
        note_size(n)
    return n

def same_size(a, b):
    if size(a) != size(b):
        raise ValueError("shape mismatch")

def add(a, b):
    same_size(a, b)
    return tensor_add(a, b)

def mul(a, b):
    same_size(a, b)
    return tensor_mul(a, b)

def relu(a):
    return tensor_relu(a)

def sum(a):
    return tensor_sum(a)

def item(a):
    return tensor_item(a)

def tensor_output(t, k):
    return t.extra[k]

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
    kernel.n = 0
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
    parts = [str(kernel.ninputs), str(kernel.n)]
    for i in range(len(kernel.nodes)):
        node = kernel.nodes[i]
        parts.append('%d:%d:%d' % (node.opcode, node.a, node.b))
    for i in range(len(kernel.outputs)):
        parts.append('o%d' % kernel.outputs[i])
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
        return kernel
    finish_kernel(kernel)
    cache_kernel(key, kernel)
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
    if kernel.fn != 0 and (kernel.n == 0 or kernel.n == a.size):
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
    result = values[len(values) - 1]
    nout = len(kernel.outputs)
    if nout > 0:
        result.extra = lltype.malloc(TENSORARRAY, nout)
        for k in range(nout):
            result.extra[k] = values[kernel.outputs[k]]
    return result

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
RPY_EXTERN void rt_cuda_free(long dptr, long n);
RPY_EXTERN void rt_cuda_set_budget(long bytes);
RPY_EXTERN long rt_cuda_launch_count(void);
RPY_EXTERN int rt_cuda_over_budget(void);
RPY_EXTERN void rt_cuda_sync(void);
RPY_EXTERN int rt_cuda_launch(long fn, long *inputs, int ninputs, long n,
                              long *outs, int nouts, int threads,
                              long elems_per_block, int shared, int nextra);
"""],
    libraries=['cuda'])
rt_cuda_load = rffi.llexternal('rt_cuda_load', [rffi.CCHARP, rffi.CCHARP],
                               lltype.Signed, compilation_info=eci,
                                releasegil=False)
rt_cuda_launch = rffi.llexternal(
    'rt_cuda_launch',
    [lltype.Signed, rffi.CArrayPtr(lltype.Signed), rffi.INT, lltype.Signed,
     rffi.CArrayPtr(lltype.Signed), rffi.INT, rffi.INT, lltype.Signed,
     rffi.INT, rffi.INT],
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



rt_cuda_free = rffi.llexternal('rt_cuda_free', [lltype.Signed, lltype.Signed],
                               lltype.Void, compilation_info=eci,
                               releasegil=False)

rt_cuda_set_budget = rffi.llexternal('rt_cuda_set_budget', [lltype.Signed],
                                     lltype.Void, compilation_info=eci,
                                     releasegil=False)
rt_cuda_over_budget = rffi.llexternal('rt_cuda_over_budget', [], rffi.INT,
                                      compilation_info=eci, releasegil=False)

rt_cuda_launch_count = rffi.llexternal('rt_cuda_launch_count', [],
                                       lltype.Signed, compilation_info=eci,
                                       releasegil=False)

def launch_count():
    return rt_cuda_launch_count()

def reset_device():
    rt_cuda_reset()

def collect_if_over_budget():
    if rffi.cast(lltype.Signed, rt_cuda_over_budget()) != 0:
        rgc.collect(0)
        if rffi.cast(lltype.Signed, rt_cuda_over_budget()) != 0:
            rgc.collect()

def sync_device():
    rt_cuda_sync()

def to_ttir(kernel, name):
    nodes = kernel.nodes
    nin = kernel.ninputs
    BLOCK = config.block
    masked = kernel.n == 0 or kernel.n % BLOCK != 0
    T = 'tensor<%dxf64>' % BLOCK
    P = 'tensor<%dx!tt.ptr<f64>>' % BLOCK
    params = ['%%in%d: !tt.ptr<f64>' % i for i in range(nin)]
    params.append('%out: !tt.ptr<f64>')
    for k in range(len(kernel.outputs)):
        params.append('%%out%d: !tt.ptr<f64>' % k)
    lines = ['module {',
             '  tt.func public @%s(%s, %%n: i64) '
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
        if masked:
            lines.append('    %%v%d = tt.load %%q%d, %%mask, %%zero : %s' % (i, i, P))
        else:
            lines.append('    %%v%d = tt.load %%q%d : %s' % (i, i, P))
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
        if masked:
            lines.append('    tt.store %%qo, %%v%d, %%mask : %s' % (last, P))
        else:
            lines.append('    tt.store %%qo, %%v%d : %s' % (last, P))
    for k in range(len(kernel.outputs)):
        lines.append('    %%po%d = tt.splat %%out%d : !tt.ptr<f64> -> %s' % (k, k, P))
        lines.append('    %%qo%d = tt.addptr %%po%d, %%offs : %s, tensor<%dxi32>'
                     % (k, k, P, BLOCK))
        if masked:
            lines.append('    tt.store %%qo%d, %%v%d, %%mask : %s'
                         % (k, kernel.outputs[k], P))
        else:
            lines.append('    tt.store %%qo%d, %%v%d : %s' % (k, kernel.outputs[k], P))
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
    shape = lltype.nullptr(SHAPEARRAY) if kernel.sumroot else inputs[0].shape
    nout = 1 + len(kernel.outputs)
    collect_if_over_budget()
    dptrs = lltype.malloc(SIGNEDARRAY, nin, flavor='raw')
    outs = lltype.malloc(SIGNEDARRAY, nout, flavor='raw')
    ok = True
    for k in range(nin):
        dptrs[k] = dev(inputs[k])
        if dptrs[k] == 0:
            ok = False
    outs[0] = rt_cuda_alloc(outlen) if ok else 0
    if outs[0] == 0:
        ok = False
    for k in range(1, nout):
        outs[k] = rt_cuda_alloc(n) if ok else 0
        if outs[k] == 0:
            ok = False
    if ok:
        ok = rffi.cast(lltype.Signed, rt_cuda_launch(
            kernel.fn, dptrs, rffi.cast(rffi.INT, nin), n,
            outs, rffi.cast(rffi.INT, nout),
            rffi.cast(rffi.INT, kernel.threads), config.block,
            rffi.cast(rffi.INT, kernel.shared),
            rffi.cast(rffi.INT, kernel.nextra))) != 0
    result = NULLTENSOR
    if ok:
        result = device_tensor(outlen, outs[0], shape)
        if nout > 1:
            result.extra = lltype.malloc(TENSORARRAY, nout - 1)
            for k in range(1, nout):
                result.extra[k - 1] = device_tensor(n, outs[k], inputs[0].shape)
    lltype.free(dptrs, flavor='raw')
    lltype.free(outs, flavor='raw')
    return result
