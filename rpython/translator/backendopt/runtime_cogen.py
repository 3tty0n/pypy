"""Generate and install a residual program for a live code object, on demand.

Guards are bound directly to the code object generation ran against, so
this skips the snapshot/matcher machinery translation-time snapshots need
(``install_from_snapshots``, PySOM's ``som/interpreter/bc/offline.py``),
and delegates the rest to ``PortalLinker.install``.
"""


def generate_for_live_code(extension, linker, codewriter, code, guard, ref,
                           entry_pc=0, entry_state=None, emitter=None,
                           native_table=None):
    """Generate and install a residual program for ``code``, guarded on
    ``ref``.

    ``extension`` scans ``code`` from ``entry_pc``/``entry_state``;
    ``linker`` is the ``PortalLinker`` for this interpreter's portal.

    ``emitter``, when given, should be a ``ProgramEmitter`` whose
    fragments were already built by ``precompile_fragments`` -- this call
    then runs no codewriter at all. None falls back to a fresh per-call
    ``ProgramEmitter``.

    ``native_table``, when given, takes priority over ``emitter`` and
    routes generation through the native pipeline (native_pipeline.py)
    instead of SSARepr-based ``ProgramEmitter.emit`` -- the path that
    does not read translation-time-only flowspace/codewriter objects, so
    a caller running from inside a translated binary can use it.

    Returns the installed ``PELinkedProgram``, or None when ``code`` is
    declined: an instruction with no template, or the assembled program
    overflows a JitCode's per-kind capacity (256 constants, including
    registers).

    Declines are not cached here: nothing is attached to the portal on a
    decline, so a caller on a lookup-miss path must keep its own
    ref-keyed record, or every miss re-runs generation from scratch.
    """
    program = extension.generate(code, entry_pc, entry_state)
    if program is None:
        return None
    try:
        lowered = linker.install(codewriter, program, guard=guard,
                                 emitter=emitter, native_table=native_table)
    except Exception:
        # Catch-all: the usual cause is capacity overflow (a JitCode
        # indexes constants with one byte per kind), but any cause here
        # must decline, not crash -- one bad method must stay unlinked.
        return None
    lowered.linked_program.guard_ref = ref
    return lowered.linked_program
