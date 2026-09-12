import array, json, os, sys, time

import torch

import warmup_common


class Args(object):
    pass


def argv():
    a = Args()
    a.mode = sys.argv[1]
    a.outdir = sys.argv[2]
    a.iters = int(sys.argv[3]) if len(sys.argv) > 3 else 20
    a.warmup = int(sys.argv[4]) if len(sys.argv) > 4 else 10
    a.dtname = os.environ.get('RTENSOR_DTYPE', 'float32')
    a.dtype = getattr(torch, a.dtname)
    a.dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    a.cfg = json.load(open(os.path.join(a.outdir, 'index.json')))
    return a


def batch_argv():
    return int(sys.argv[5]) if len(sys.argv) > 5 else 1


def flat_weights(outdir):
    buf = array.array('f')
    path = os.path.join(outdir, 'weights.bin')
    buf.fromfile(open(path, 'rb'), os.path.getsize(path) // 4)
    if sys.byteorder != 'little':
        buf.byteswap()
    return torch.tensor(buf, dtype=torch.float32)


def image(a):
    off, shape = a.cfg['index']['image']
    n = shape[0] * shape[1] * shape[2]
    buf = array.array('f')
    path = os.path.join(a.outdir, 'weights.bin')
    buf.fromfile(open(path, 'rb'), os.path.getsize(path) // 4)
    return torch.tensor(buf[off:off + n]).view(1, *shape)


def hf_kwargs(a):
    """Torch-TensorRT 2.9 lowers scaled_dot_product_attention with a mask
    wrongly (BERT logits off by ~25); the eager attention path converts
    exactly, so HF models use it under tensorrt."""
    return {'attn_implementation': 'eager'} if a.mode == 'tensorrt' else {}


def compiled(fwd, a):
    """eager: as is; compile: Inductor; compile-ro/compile-mat: Inductor under
    reduce-overhead (CUDA graphs) / max-autotune; tensorrt: Torch-TensorRT
    through the torch.compile front end, so the same model definition serves
    all systems."""
    if a.mode == 'compile':
        return torch.compile(fwd)
    if a.mode == 'compile-ro':
        return torch.compile(fwd, mode='reduce-overhead')
    if a.mode == 'compile-mat':
        return torch.compile(fwd, mode='max-autotune')
    if a.mode == 'tensorrt':
        import torch_tensorrt  # noqa: F401  registers the backend
        return torch.compile(fwd, backend='tensorrt',
                             options={'enabled_precisions': {a.dtype},
                                      'disable_tf32': True,
                                      'min_block_size': 1})
    return fwd


def profile_launches(fwd, args, a, n=5):
    """Dead-code guard: CUDA kernels launched per forward.

    The timed loop observes its result once, so nothing in the timing itself
    proves that every iteration ran.  This counts the kernels a forward
    actually launches - but in a separate short window after the timed loop,
    never inside it, because the profiler adds milliseconds per iteration and
    would otherwise be measured as the model's cost."""
    if a.dev != 'cuda':
        return None
    try:
        from torch.profiler import profile, ProfilerActivity
        with profile(activities=[ProfilerActivity.CPU,
                                 ProfilerActivity.CUDA]) as prof:
            for _ in range(n):
                fwd(*args)
            torch.cuda.synchronize()
        total = 0
        for ev in prof.events():
            total += len(getattr(ev, 'kernels', None) or [])
        return float(total) / n
    except Exception as exc:
        sys.stderr.write('torch_common: launch profiling failed: %s\n' % exc)
        return None


def timed(fwd, args, a):
    """Warm-up, sync, timed loop, sync.  The first forward is timed on its
    own into a.first_run_ms: for compile mode that is compile plus one run."""
    with torch.no_grad():
        n = warmup_common.trace_n()
        if n:
            us = []
            for i in range(n):
                t0 = time.time()
                logits = fwd(*args)
                if a.dev == 'cuda':
                    torch.cuda.synchronize()
                dt = (time.time() - t0) * 1e6
                us.append(dt)
                print('iter=%d us=%.1f' % (i, dt))
            warmup_common.report_trace(us)
            sys.exit(0)
        t0 = time.time()
        logits = fwd(*args)
        if a.dev == 'cuda':
            torch.cuda.synchronize()
        a.first_run_ms = (time.time() - t0) * 1e3
        for i in range(a.warmup - 1):
            logits = fwd(*args)
        if a.dev == 'cuda':
            torch.cuda.synchronize()
        t0 = time.time()
        for i in range(a.iters):
            logits = fwd(*args)
        if a.dev == 'cuda':
            torch.cuda.synchronize()
        acc = logits.double().sum().item()
        steady_us = (time.time() - t0) / a.iters * 1e6
        if a.mode in ('compile-ro', 'compile-mat'):
            # Both modes turn cudagraphs on (reduce-overhead always,
            # max-autotune by default) and replay into a static output
            # buffer; the caller (argmax/compare) reads logits again after
            # this returns, and the timed loop above is already done, so
            # cloning here costs nothing in the measured region while making
            # that later read safe against the buffer being reused.
            logits = logits.clone()
        # After the clone: the profiling window runs more forwards, which
        # would overwrite the cudagraph output buffer logits still points at.
        a.launches_per_iter = profile_launches(fwd, args, a)
    return logits, acc, steady_us


def reference_name():
    dtype = os.environ.get('RTENSOR_DTYPE', 'float32')
    return 'logits_pypy.bin' if dtype == 'float32' else 'logits_pypy_%s.bin' % dtype


def tolerance():
    """The tolerance for this (workload class, dtype), from config.sh's
    tolerance_for table via MODEL_TOL.  One table for every system, so a
    backend cannot be checked more loosely than the one it is compared to."""
    try:
        return float(os.environ.get('MODEL_TOL', '1e-3'))
    except ValueError:
        return 1e-3


def compare(outdir, logits, argmax=None):
    ref = os.path.join(outdir, reference_name())
    if not os.path.exists(ref):
        return ''
    other = torch.frombuffer(open(ref, 'rb').read(),
                             dtype=torch.float32).to(logits.device)
    other = other.view_as(logits.float())
    d = (logits.float() - other).abs().max().item()
    tol = tolerance()
    diff = ' maxabsdiff=%.6g tol=%.6g pass=%d' % (d, tol, int(d <= tol))
    if argmax is not None:
        mine = other.argmax(-1).tolist()
        diff += ' argmax_match=%d/%d' % (
            sum(int(x == y) for x, y in zip(argmax, mine)), len(argmax))
    return diff


def report(line, order, diff, a=None):
    if a is not None and getattr(a, 'launches_per_iter', None) is not None:
        line += ' launches_per_iter=%.1f' % a.launches_per_iter
    if a is not None and getattr(a, 'first_run_ms', None) is not None:
        line += ' compile_ms=-1 first_run_ms=%.1f' % a.first_run_ms
    print(line)
    print('argmax %s%s' % (' '.join(str(a) for a in order), diff))
