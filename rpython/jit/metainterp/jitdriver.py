

from rpython.rtyper.lltypesystem import lltype
from rpython.rtyper import rclass


class JitDriverStaticData(object):
    """There is one instance of this class per JitDriver used in the program.
    """
    # This is just a container with the following attributes (... set by):
    #    self.jitdriver         ... rpython.jit.metainterp.warmspot
    #    self.portal_graph      ... rpython.jit.metainterp.warmspot
    #    self.portal_runner_ptr ... rpython.jit.metainterp.warmspot
    #    self.portal_runner_adr ... rpython.jit.metainterp.warmspot
    #    self.portal_calldescr  ... rpython.jit.metainterp.warmspot
    #    self.num_green_args    ... rpython.jit.metainterp.warmspot
    #    self.num_red_args      ... rpython.jit.metainterp.warmspot
    #    self.red_args_types    ... rpython.jit.metainterp.warmspot
    #    self.result_type       ... rpython.jit.metainterp.warmspot
    #    self.virtualizable_info... rpython.jit.metainterp.warmspot
    #    self.greenfield_info   ... rpython.jit.metainterp.warmspot
    #    self.warmstate         ... rpython.jit.metainterp.warmspot
    #    self.handle_jitexc_from_bh rpython.jit.metainterp.warmspot
    #    self.no_loop_header    ... rpython.jit.metainterp.warmspot
    #    self.portal_finishtoken... rpython.jit.metainterp.pyjitpl
    #    self.propagate_exc_descr.. rpython.jit.metainterp.pyjitpl
    #    self.index             ... rpython.jit.codewriter.call
    #    self.mainjitcode       ... rpython.jit.codewriter.call

    # Set by the interpreter's PE setup (pe_linked_setup), optional:
    #    self.pe_recover_jitcode   jitcode of a function (virtualizable,
    #                              exception) -> portal result: unwinds a
    #                              guest exception that escaped a linked
    #                              program to its handler and carries on
    #                              from there through pe_resume_jitcode
    #    self.pe_resume_jitcode    the function (virtualizable, pc) ->
    #                              portal result it calls the portal from;
    #                              the tracer re-enters its root frame at
    #                              that call instead of nesting it
    #    self.pe_recover_exc_class vtable of the exceptions it handles
    pe_recover_jitcode = None
    pe_resume_jitcode = None
    pe_recover_exc_class = lltype.nullptr(rclass.CLASSTYPE.TO)

    # These attributes are read by the backend in CALL_ASSEMBLER:
    #    self.assembler_helper_adr
    #    self.index_of_virtualizable
    #    self.vable_token_descr
    #    self.portal_calldescr

    # warmspot sets extra attributes starting with '_' for its own use.

    def _freeze_(self):
        return True
