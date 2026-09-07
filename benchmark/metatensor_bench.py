import sys, os, time
from rpython.rlib import jit
from rpython.metatensor import core, device, kernels, runtime
from bench import cnn, chain, mlp, reduce, transformer
from bench.common import bench_env, cfg, emit, sink

def run_model(variant, n, iters):
    if variant == 13:
        return transformer.run_attn(n, iters)
    if variant == 12:
        return reduce.run_matmul(n, iters)
    if variant == 11:
        return reduce.run_reduction(n, iters)
    if variant == 10:
        return transformer.run_transformer_train(n, iters)
    if variant == 9:
        return cnn.run_cnn(n, iters)
    if variant == 7:
        return mlp.run_mlp_train(n, iters)
    if variant == 8:
        return transformer.run_block(n, iters)
    return mlp.run_mlp(n, iters)

def entry_point(argv):
    if len(argv) != 6:
        print 'usage: metatensor-bench MODE VARIANT K N ITERS  (MODE: fused|eager|nojit, VARIANT: 0..13)'
        return 1
    mode = argv[1]
    variant = int(argv[2])
    k = int(argv[3])
    n = int(argv[4])
    iters = int(argv[5])
    jit.set_user_param(None, 'threshold=3,function_threshold=3,trace_eagerness=2')
    jit.set_user_param(None, 'trace_limit=60000')
    extra = os.environ.get('RTENSOR_JIT')
    if extra is not None:
        jit.set_user_param(None, extra)
    if mode == 'eager':
        jit.set_user_param(None, 'enable_opts=intbounds:rewrite:virtualize:'
                                 'string:pure:earlyforce:heap:unroll')
    elif mode == 'nojit':
        jit.set_user_param(None, 'off')
    kernels.init_device()
    try:
        cfg.dtype = core.dtype_of_name(bench_env('RTENSOR_DTYPE', 'float64'))
    except ValueError:
        print 'unknown RTENSOR_DTYPE'
        return 1
    kernels.init_dtype(cfg.dtype)
    dtname = core.DTYPE_NAMES[cfg.dtype]
    sink.fd = os.open('/dev/null', os.O_WRONLY, 0)
    if variant >= 6:
        run_model(variant, n, 20)
        t0 = time.time()
        run_model(variant, n, 20)
        warm = time.time() - t0
        run_model(variant, n, 30)
        run_model(variant, n, 30)
        before = kernels.counter.n
        launches_before = device.launch_count()
        t0 = time.time()
        acc = run_model(variant, n, iters)
    else:
        w, b = chain.make_inputs(n)
        t0 = time.time()
        chain.run(variant, k, w, b, 20)
        warm = time.time() - t0
        chain.run(variant, k, w, b, 30)
        chain.run(variant, k, w, b, 30)
        before = kernels.counter.n
        launches_before = device.launch_count()
        t0 = time.time()
        acc = chain.run(variant, k, w, b, iters)
    device.sync_device()
    steady = (time.time() - t0) / iters * 1e6
    launches = float(device.launch_count() - launches_before) / iters
    runtime.reset_device()
    emit(mode, variant, k, n, iters, warm, steady, kernels.counter.n, acc,
         kernels.counter.n - before, launches, dtname)
    return 0

def target(*args):
    return entry_point, None

if __name__ == '__main__':
    entry_point(sys.argv)
