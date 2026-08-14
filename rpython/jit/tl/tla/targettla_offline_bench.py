"""Translation target for comparing native TLA JIT binaries."""

import os

from rpython.jit.codewriter.policy import JitPolicy
from rpython.jit.tl.tla import offline, tla


BYTECODE = ''.join(chr(op) for op in [
    tla.CONST_INT, 1, tla.SUB, tla.DUP, tla.JUMP_IF, 0, tla.RETURN])
USE_OFFLINE_PE = os.environ.get("TLA_OFFLINE_PE", "0") == "1"


def entry_point(argv):
    if len(argv) != 3:
        print "usage: %s bytecode countdown" % argv[0]
        return 2
    from rpython.rlib.streamio import open_file_as_stream
    stream = open_file_as_stream(argv[1])
    bytecode = stream.readall()
    stream.close()
    result = tla.run(bytecode, tla.W_IntObject(int(argv[2])))
    print result.getrepr()
    return 0


def target(driver, args):
    return entry_point


def jitpolicy(driver):
    if USE_OFFLINE_PE:
        def install(codewriter, jitdriver_sd, translator):
            return offline.lower_and_install(
                codewriter, jitdriver_sd, translator, BYTECODE)
        driver.translator._pe_linked_setup = install
    return JitPolicy()
