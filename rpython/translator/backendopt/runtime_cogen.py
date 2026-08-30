"""Generate and install a residual program for a live code object, on demand."""


def generate_for_live_code(extension, linker, codewriter, code, guard, ref,
                           entry_pc=0, entry_state=None, emitter=None,
                           native_table=None, profiler=None):
    """Generate and install a residual program for ``code``, guarded on ref."""
    from rpython.rlib.debug import (debug_print, debug_start,
                                    debug_stop)
    if profiler is not None:
        profiler.start_pe_cogen_scan()
    debug_start("pe-cogen-scan")
    try:
        program = extension.generate(code, entry_pc, entry_state)
    except Exception as error:
        # One code object the scan cannot handle must stay unlinked.
        debug_print("pe-cogen-scan raised", str(error))
        program = None
    finally:
        debug_stop("pe-cogen-scan")
        if profiler is not None:
            profiler.end_pe_cogen_scan()
    if program is None:
        blocked_pc, blocked_key = extension.last_blocked
        debug_print("pe-cogen-scan declined: blocked pc", blocked_pc,
                    "key", blocked_key)
        return None
    debug_start("pe-cogen-install")
    if profiler is not None:
        profiler.start_pe_cogen_install()
    try:
        lowered = linker.install(codewriter, program, guard=guard,
                                 emitter=emitter, native_table=native_table)
    except Exception:
        # Usually capacity overflow; any cause here must decline, not crash.
        return None
    finally:
        debug_stop("pe-cogen-install")
        if profiler is not None:
            profiler.end_pe_cogen_install()
    lowered.linked_program.match_ref = ref
    return lowered.linked_program
