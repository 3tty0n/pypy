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
    if len(argv) != 6 and len(argv) != 7:
        print ('usage: metatensor-bench MODE VARIANT K N ITERS [WARMUP]  '
               '(MODE: fused|eager|nojit, VARIANT: 0..13)')
        return 1
    mode = argv[1]
    variant = int(argv[2])
    k = int(argv[3])
    n = int(argv[4])
    iters = int(argv[5])
    # Warm-up iterations, same argv position and same default as every other
    # micro driver.  The first call is separate because it is the one that
    # traces and compiles kernels; the remaining warmup-1 are plain
    # iterations, so exactly `warmup` iterations run before the timed loop.
    warmup = int(argv[6]) if len(argv) == 7 else 30
    if warmup < 1:
        warmup = 1
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
    if bench_env('RTENSOR_DEBUG_KNOBS', '') != '':
        os.write(2, 'max_inputs=%d\n' % core.max_inputs())
    try:
        cfg.dtype = core.dtype_of_name(bench_env('RTENSOR_DTYPE', 'float64'))
    except ValueError:
        print 'unknown RTENSOR_DTYPE'
        return 1
    kernels.init_dtype(cfg.dtype)
    dtname = core.DTYPE_NAMES[cfg.dtype]
    sink.fd = os.open('/dev/null', os.O_WRONLY, 0)
    cfg.eff_n = n
    if variant >= 6:
        run_model(variant, n, 1)
        t0 = time.time()
        run_model(variant, n, warmup - 1)
        warm = time.time() - t0
        before = kernels.counter.n
        launches_before = device.launch_count()
        t0 = time.time()
        acc = run_model(variant, n, iters)
    else:
        w, b = chain.make_inputs(n)
        chain.run(variant, k, w, b, 1)
        t0 = time.time()
        chain.run(variant, k, w, b, warmup - 1)
        warm = time.time() - t0
        before = kernels.counter.n
        launches_before = device.launch_count()
        t0 = time.time()
        acc = chain.run(variant, k, w, b, iters)
    device.sync_device()
    steady = (time.time() - t0) / iters * 1e6
    launches = float(device.launch_count() - launches_before) / iters
    if device.alloc_failed():
        os.write(2, 'metatensor-bench: variant %d fell back to CPU mid-run '
                    '(device allocation failed), discarding this '
                    'measurement\n' % variant)
        return 1
    runtime.reset_device()
    # Only report a different n when the working set actually had to shrink;
    # the plain floor of n // d is left alone so rows stay comparable with
    # earlier results.
    report_n = n
    if cfg.capped:
        report_n = cfg.eff_n
        os.write(2, 'metatensor-bench: variant %d n %d -> %d '
                    '(fitted to GPU memory)\n' % (variant, n, report_n))
    emit(mode, variant, k, report_n, iters, warm, steady, kernels.counter.n,
         acc, kernels.counter.n - before, launches, dtname)
    return 0

def target(*args):
    return entry_point, None

if __name__ == '__main__':
    entry_point(sys.argv)
