# Online cogen: architecture map

This branch adds partial-evaluation-based code generation ("cogen") to the
RPython JIT: at translation time the interpreter's step function is turned
into per-opcode *templates*; at run time, a hot guest code object gets a
*residual program* built by concatenating those templates, and the JIT
traces that program instead of the generic dispatch loop.

Read this file first, then follow the pipeline below in order.

## Vocabulary

| term | meaning |
|---|---|
| template | per-opcode residual IR, generated once at translation time from the interpreter's step function |
| fragment | the runtime representation of one template, ready to be copied into a program |
| residual program | the fragments for one guest code object, concatenated and patched into one JitCode |
| linked program | a residual program installed on the portal (`PELinkedProgram`); the portal enters it instead of generic dispatch when its matcher accepts the (code, pc) greens |
| generating extension | the translation-time object that produces templates (Futamura's second projection) |
| cogen | generating and installing a residual program at run time |
| leave pc | a guest pc outside the program that one of its exits jumps to |
| `PE_LEAVE` | the residual opcode that ends a program's block by returning `W_ResidualExit(next_instr)` to `execute_frame`, which re-enters the portal at that pc |
| `pe_bailout_point` | a no-op marker at each block boundary; only the *blackhole* interpreter uses it, to stop early and replay from that pc instead of blackholing to the next real merge point |
| legit entry pc | a block boundary where a trace may start (program entry + loop headers) |

`PE_LEAVE` and `pe_bailout_point` are different mechanisms for the same
need — getting out of a residual program cheaply: `PE_LEAVE` is the exit
compiled code takes; `pe_bailout_point` is the shortcut the blackhole
interpreter takes after a guard failure.

## Pipeline: translation time

```
interpreter step function (pypy/interpreter/pyopcode.py, interp_step)
  | partialeval.py          specialize the flow graph per opcode
  v
partialeval_template.py     residual template IR, one per opcode
  | generating_extension.py drive template generation, decode bytecode
  v
native_fragments.py         convert templates to the runtime IR
jitcode_emitter.py          assemble each template once (translation time)
```

## Pipeline: run time (inside the translated pypy-c)

```
portal entry (rpython/jit/metainterp/warmstate.py, maybe_compile_and_run)
  v
pe_enter_root (pyjitpl.py) -> linked_program_for (codewriter/jitcode.py)
  |  miss counter reaches PYPY_COGEN_THRESHOLD, gate accepts
  v
runtime_cogen.py            generate_for_live_code: build the program
native_pipeline.py          emit + liveness + assemble (runtime port)
portal_linker.py            wrap it as a PELinkedProgram, set the matcher
  v
register_late_jitcode       the JIT traces the residual JitCode
```

Exits at run time: a guard failure in the compiled residual either gets a
bridge (`compile.py, must_compile`: eagerness, cost model, abort backoff)
or resumes in the blackhole (`blackhole.py, resume_in_blackhole`), which
stops at the next `pe_bailout_point`; compiled code leaves via `PE_LEAVE`
and `execute_frame` (`pypy/interpreter/pyframe.py`) re-enters the portal
at the leave pc (counted with bridge eagerness, `warmstate.py,
pe_entry_increment`).

## Module map

| module | role |
|---|---|
| `pypy/interpreter/pe_cogen.py` | wires cogen into the PyPy interpreter: env vars, gate, opcode table, exclusions |
| `rpython/rlib/pe.py` | `PEDriver` declaration an interpreter uses to opt in |
| `partialeval.py` | flow-graph specializer (translation time) |
| `partialeval_template.py` | template IR and lowering (translation time) |
| `generating_extension.py` | drives per-opcode template generation |
| `native_fragments.py` | template -> runtime IR conversion |
| `jitcode_emitter.py` | translation-time assembly of templates |
| `native_pipeline.py` | runtime emit/liveness/assemble (port of the codewriter passes onto the runtime IR) |
| `portal_linker.py` | installs a program on the jitdriver's portal |
| `runtime_cogen.py` | the run-time entry: build + install for one live code object |
| `rpython/jit/codewriter/jitcode.py` | `PELinkedProgram`, `PEJitCodeMetadata`, program matching, `dump_jitcode` |
| `rpython/jit/metainterp/pyjitpl.py` | `pe_enter_root` and friends: entering programs while tracing |
| `rpython/jit/metainterp/warmstate.py` | counters: miss threshold, tick suppression, leave-pc eagerness |
| `rpython/jit/metainterp/compile.py` | per-guard bridge decision (eagerness, cost model, backoff) |

`jitcode_emitter.py` and `native_pipeline.py` implement the same passes
twice: once on translation-time objects, once on the runtime IR. The
runtime copy must stay behaviourally identical; an equivalence test gates
it (`test_native_pipeline_liveness_differential.py`).

## Debugging

- `PYPY_COGEN_THRESHOLD=0 PYPY_PE_GATE=0 PYPYLOG=jit-jitcode-dump,pe-cogen:log`
  forces generation on first miss and dumps every residual JitCode.
- `PYPYLOG=jit-summary:log` adds `Blackhole`, `Blackhole decode`,
  `guard failures >=2^k`, `bridges at 2^k`, `bridge model`, `pe cogen *`.
- `PYPYLOG=jit-bridge-cost:log` prints one line per bridge attempt.

Start reading with `runtime_cogen.py` (42 lines), then
`portal_linker.py`, then the two pipelines.
