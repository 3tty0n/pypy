
all: pypy-c cffi_imports

PYPY_EXECUTABLE := $(shell which pypy)

ifeq ($(PYPY_EXECUTABLE),)
RUNINTERP = python
else
RUNINTERP = $(PYPY_EXECUTABLE)
endif

URAM := $(shell $(RUNINTERP) -c "import sys; print(4.5 if sys.maxint>1<<32 else 2.5)")

JOBS=$(subst -j,--make-jobs ,$(filter -j%, $(MAKEFLAGS)))

.PHONY: pypy-c cffi_imports

pypy-jit-ext-c:
	@echo
	@echo "===================================================================="
ifeq ($(PYPY_EXECUTABLE),)
	@echo "Building the GenExtension + phase-speculation PyPy binary, using CPython."
	@echo "This takes around 2 hours and $(URAM) GB of RAM."
	@echo "Note that pre-installing a PyPy binary would reduce this time"
	@echo "and produce basically the same result."
else
	@echo "Building the GenExtension + phase-speculation PyPy binary, using"
	@echo "$(PYPY_EXECUTABLE) to run the translation itself."
	@echo "This takes up to 1 hour and $(URAM) GB of RAM."
endif
	@echo
	@echo "Output: pypy/goal/pypy-jit-genext-c (with libpypy-jit-genext-c.so)."
	@echo "Phase-speculation is env-gated at runtime by PYPY_PHASE_AWARE=1."
	@echo "GenExtension is translation-time-gated by --gen-extension below."
	@echo "===================================================================="
	@echo
	@sleep 5
	cd pypy/goal && $(RUNINTERP) ../../rpython/bin/rpython $(JOBS) -Ojit --gen-extension targetpypystandalone.py

pypy-c:
	@echo
	@echo "===================================================================="
ifeq ($(PYPY_EXECUTABLE),)
	@echo "Building a regular (jitting) version of PyPy, using CPython."
	@echo "This takes around 2 hours and $(URAM) GB of RAM."
	@echo "Note that pre-installing a PyPy binary would reduce this time"
	@echo "and produce basically the same result."
else
	@echo "Building a regular (jitting) version of PyPy, using"
	@echo "$(PYPY_EXECUTABLE) to run the translation itself."
	@echo "This takes up to 1 hour and $(URAM) GB of RAM."
endif
	@echo
	@echo "For more control (e.g. to use multiple CPU cores during part of"
	@echo "the process) you need to run \`\`rpython/bin/rpython'' directly."
	@echo "For more information see \`\`http://pypy.org/download.html''."
	@echo "===================================================================="
	@echo
	@sleep 5
	cd pypy/goal && PYPY_EXE_NAME=pypy $(RUNINTERP) ../../rpython/bin/rpython $(JOBS) -Ojit --no-gen-extension --no-heapcache-genext-fastpath targetpypystandalone.py


pypy-c-heapcache:
	cd pypy/goal && PYPY_EXE_NAME=pypy-heapcache $(RUNINTERP) ../../rpython/bin/rpython $(JOBS) -Ojit --no-gen-extension --heapcache-genext-fastpath targetpypystandalone.py


pypy-jit-ext-c-no-heapcache:
	cd pypy/goal && PYPY_EXE_NAME=pypy-jit-ext-no-heapcache $(RUNINTERP) ../../rpython/bin/rpython $(JOBS) -Ojit --gen-extension --no-heapcache-ext-fastpath targetpypystandalone.py


cffi_imports: pypy-c
	cd lib_pypy && ../pypy/goal/pypy-c pypy_tools/build_cffi_imports.py || /bin/true
