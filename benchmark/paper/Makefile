# Owns exactly the two translated binaries setup.sh needs: build/metatensor-bench
# and build/pypy-c. Rebuilds either one when its real sources change, so setup.sh
# no longer reimplements `find -newer` staleness checks by hand.
#
# Deliberately NOT a build graph for the rest of the harness:
#   - the measurement scripts (run_*.sh) append to TSVs and re-run the same
#     measurement ROUNDS times on purpose; "up to date, nothing to do" is the
#     wrong model for them.
#   - weight export (export_weights.sh) is not either: its input is a
#     HuggingFace model id, not a local file, and it already skips what exists.

BUILD ?= $(CURDIR)/build
REPO ?= $(abspath $(CURDIR)/../..)
PYTHON2 ?= python2
RTENSOR_PYTHON ?=
RTENSOR_CUBLAS ?=
MAKE_JOBS ?= $(shell nproc)

BENCH_SRCS := $(REPO)/benchmark/metatensor_bench.py \
              $(wildcard $(REPO)/benchmark/bench/*.py) \
              $(wildcard $(REPO)/rpython/metatensor/*.py) \
              $(wildcard $(REPO)/rpython/metatensor/*.c) \
              $(wildcard $(REPO)/rpython/metatensor/*.h)

PYPYC_SRCS := $(wildcard $(REPO)/pypy/module/_metatensor/*.py) \
              $(wildcard $(REPO)/lib_pypy/tensorpypy/*.py) \
              $(wildcard $(REPO)/rpython/metatensor/*.py) \
              $(wildcard $(REPO)/rpython/metatensor/*.c) \
              $(wildcard $(REPO)/rpython/metatensor/*.h)

.PHONY: all clean
all: $(BUILD)/metatensor-bench $(BUILD)/pypy-c

$(BUILD)/metatensor-bench: $(BENCH_SRCS)
	mkdir -p "$(BUILD)"
	cd "$(BUILD)" && RTENSOR_PYTHON="$(RTENSOR_PYTHON)" RTENSOR_CUBLAS="$(RTENSOR_CUBLAS)" \
	  PYTHON2="$(PYTHON2)" MAKE_JOBS="$(MAKE_JOBS)" \
	  "$(REPO)/benchmark/build.sh" "$(BUILD)/metatensor-bench"

$(BUILD)/pypy-c: $(PYPYC_SRCS)
	mkdir -p "$(BUILD)"
	cd "$(BUILD)" && PYTHONPATH="$(REPO)" "$(PYTHON2)" "$(REPO)/rpython/bin/rpython" \
	  --batch --make-jobs="$(MAKE_JOBS)" -Ojit --no-shared \
	  --output="$(BUILD)/pypy-c" \
	  "$(REPO)/pypy/goal/targetpypystandalone.py" --withmod-_metatensor

clean:
	rm -f "$(BUILD)/metatensor-bench" "$(BUILD)/pypy-c"
