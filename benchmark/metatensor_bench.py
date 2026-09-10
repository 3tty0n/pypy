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

# Warmup schedule.  The JIT thresholds are 3, and steady_us on variant 13 at
# n=256000 was flat within 0.4% between 21 and 101 total warmup iterations, so
# the old 20/30 schedule was about five times longer than it needed to be.
WARMUP_ITERS, PRESTEADY_ITERS = 10, 10
MIN_WARMUP = 5

def warmup_budget_s():
    try:
        return int(bench_env('RTENSOR_WARMUP_BUDGET_S', '30'))
    except ValueError:
        return 30

def warmup_plan(per_iter):
    """Warmup iteration counts for one measurement point.

    Cheap points keep the fixed 20/30 schedule.  For a point where a single
    iteration already costs a noticeable fraction of the budget - the quadratic
    attention variants at large n - the 100 warmup iterations alone would run
    for tens of minutes, so the schedule is trimmed to what fits.  It never
    drops below MIN_WARMUP, which has to stay above the JIT thresholds.
    """
    if per_iter <= 0.0:
        return WARMUP_ITERS, PRESTEADY_ITERS
    budget = float(warmup_budget_s())
    afford = int(budget / per_iter)
    if afford < MIN_WARMUP:
        afford = MIN_WARMUP
    w1 = WARMUP_ITERS if afford > WARMUP_ITERS else afford
    w2 = PRESTEADY_ITERS if afford > PRESTEADY_ITERS else afford
    return w1, w2

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
    cfg.eff_n = n
    if variant >= 6:
        t0 = time.time()
        run_model(variant, n, 1)
        w1, w2 = warmup_plan(time.time() - t0)
        run_model(variant, n, w1)
        t0 = time.time()
        run_model(variant, n, w1)
        warm = time.time() - t0
        run_model(variant, n, w2)
        run_model(variant, n, w2)
        before = kernels.counter.n
        launches_before = device.launch_count()
        t0 = time.time()
        acc = run_model(variant, n, iters)
    else:
        w, b = chain.make_inputs(n)
        t0 = time.time()
        chain.run(variant, k, w, b, WARMUP_ITERS)
        warm = time.time() - t0
        chain.run(variant, k, w, b, PRESTEADY_ITERS)
        chain.run(variant, k, w, b, PRESTEADY_ITERS)
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
