import os
import py
py.path.local(__file__)

from rpython.rlib.rtime import time
from rpython.rlib import jit
from rpython.jit.tl.threadedcode import tla
from rpython.jit.tl.threadedcode import frames
from rpython.jit.tl.threadedcode.bytecode import Bytecode

def entry_point(args):
    usage = "Usage: %s filename x n" % (args[0],)

    if len(args) < 3:
        print usage
        return 2

    debug = False
    tier = 1
    i = 0
    while True:
        if not i < len(args):
            break

        if args[i] == "--jit":
            if len(args) == i + 1:
                print "missing argument after --jit"
                return 2
            jitarg = args[i+1]
            del args[i:i+2]
            jit.set_user_param(None, jitarg)
            continue
        elif args[i] == "--debug":
            debug = True
            del args[i]
            continue
        elif args[i] == "--tier":
            tier = int(args[i+1])
            del args[i:i+2]
            continue
        i += 1

    filename = args[1]
    x = int(args[2])

    n = 100
    if len(args) > 3:
        n = int(args[3])

    if tier == 0:
        jit.set_user_param(None, "off")
    elif tier == 1:
        jit.set_user_param(None, "inlining=0")
    elif tier >= 2:
        # TLA_THRESHOLD lowers the loop/function compile threshold -- the "when to
        # compile" warmup knob: a smaller value reaches compiled code after fewer
        # interpreted iterations (faster warmup for interpretation-bound loops) at
        # the cost of compiling colder loops.  Keep TLA_FREEZE below it so the
        # tier-4 type decision still settles before the loop compiles.
        params = "inlining=1"
        th = os.environ.get('TLA_THRESHOLD')
        if th:
            params = "inlining=1,threshold=%s,function_threshold=%s" % (th, th)
        jit.set_user_param(None, params)

    frames._t4_configure()
    w_x = tla.W_IntObject(x)
    bytecode = load_bytecode(filename)
    w_res = tla.W_IntObject(0)
    for _ in range(n):
        n1 = time()
        w_res = tla.run(bytecode, w_x, debug=debug, tier=tier)
        n2 = time()
        print n2 - n1
    print w_res.getrepr()
    if os.environ.get('TLA_DUMP_PROFILE'):
        os.write(2, "CFG ratio=%d freeze=%d min=%d\n" % (
            frames._t4cfg.ratio, frames._t4cfg.freeze, frames._t4cfg.minn))
        i = 0
        while i < len(bytecode):
            ca = bytecode.cnt_a[i]
            cb = bytecode.cnt_b[i]
            if ca != 0 and cb != 0:
                os.write(2, "PROFILE site=%d cnt_a=%d cnt_b=%d poly=%d\n" % (
                    i, ca, cb, bytecode.poly[i]))
            i += 1
    return 0

def load_bytecode(filename):
    from rpython.rlib.streamio import open_file_as_stream
    f = open_file_as_stream(filename)
    bytecode = f.readall()
    f.close()
    return Bytecode(bytecode)

def target(driver, args):
    return entry_point

# ____________________________________________________________


if __name__ == '__main__':
    import sys
    sys.exit(entry_point(sys.argv))
