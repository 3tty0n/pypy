#!/usr/bin/env python3
"""Time one suite model under torch, with the dashboard's own code.

    run_torch.py SUITE MODEL [--backends eager,inductor,inductor-cg]
                             [--repeat 30] [--batch B]

The model and inputs come from the dashboard's runner (suite_census.
make_runner), each measurement is benchmarks/dynamo/common.timed() with
times=1 - one forward, then torch.cuda.synchronize() - and the median of
`repeat` of them is reported, as speedup_experiment does.  Backends:

  eager         the runner's forward_pass as is
  inductor      torch.compile(backend="inductor") of the forward, as the
                dashboard's --inductor run compiles it
  inductor-cg   the same with inductor's CUDA graphs on (the dashboard's
                --inductor --cudagraphs configuration)

Each backend gets the dashboard's warm-up (common.py runs the model a few
times before timing so compilation is not measured); its first call is
reported on its own.
"""
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "paper"))
import suite_census  # noqa: E402


def main(argv):
    import statistics
    import torch
    suite, name = argv[1], argv[2]
    backends = (argv[argv.index("--backends") + 1].split(",")
                if "--backends" in argv else ["eager", "inductor",
                                              "inductor-cg"])
    repeat = int(argv[argv.index("--repeat") + 1]) if "--repeat" in argv \
        else 30
    batch = int(argv[argv.index("--batch") + 1]) if "--batch" in argv \
        else None
    runner, args = suite_census.make_runner(suite, name)
    import common
    # run() does this before any timing; timed() syncs through it
    common.synchronize = torch.cuda.synchronize
    _, _, model, inputs, batch = runner.load_model("cuda", name,
                                                   batch_size=batch)
    model, inputs = runner.cast_based_on_args(model, inputs)
    for backend in backends:
        torch._dynamo.reset()
        fn = runner.forward_pass
        if backend.startswith("inductor"):
            import torch._inductor.config as ic
            ic.triton.cudagraphs = backend == "inductor-cg"
            fn = torch._dynamo.optimize("inductor")(runner.forward_pass)
        with torch.no_grad():
            t0 = time.perf_counter()
            fn(model, inputs)
            torch.cuda.synchronize()
            first_ms = (time.perf_counter() - t0) * 1e3
            for _ in range(3):
                common.timed(model, fn, inputs, times=1)
            times = [common.timed(model, fn, inputs, times=1) * 1e3
                     for _ in range(repeat)]
        print("torch suite=%s model=%s batch=%s backend=%s median_ms=%.3f "
              "min_ms=%.3f first_ms=%.1f torch=%s"
              % (suite, name, batch, backend, statistics.median(times),
                 min(times), first_ms, torch.__version__), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
