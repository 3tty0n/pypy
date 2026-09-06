# MLSys paper evaluation harness

Build the two binaries first:

    RTENSOR_PYTHON=~/.venvs/triton/bin/python benchmark/build.sh   # -> benchmark/metatensor-bench
    ./translate.py --withmod-_metatensor -Ojit pypy/goal/pypy.py    # or your usual translate command

Point config at them and run:

    export PYPY=/path/to/pypy-c        # translated with --withmod-_metatensor
    export BENCH=/path/to/metatensor-bench
    export RTENSOR_PYTHON=~/.venvs/triton/bin/python
    export WEIGHTS=/path/to/weights    # default: scratch/paper-weights
    export OUT=/path/to/results        # default: benchmark/results/paper-<date>-<host>

    benchmark/paper/run_all.sh

Each stage is skippable: `SKIP_EXPORT=1`, `SKIP_MICRO=1`, `SKIP_MODELS=1`,
`SKIP_ABLATION=1`, `SKIP_DYNAMIC=1`, `SKIP_SUMMARIZE=1`.

Individual scripts (`export_weights.sh`, `run_micro.sh`, `run_models.sh`,
`run_ablation.sh`, `run_dynamic.sh`, `summarize.py`) can also be run on
their own; they all source `config.sh` for paths and defaults
(`ITERS`, `WARMUP`, `ROUNDS`, `JIT_FLAGS`).

Results land as TSVs plus `summary.md` under `$OUT`; `log.txt` there has
the full run_all.sh output, including any failed step.
