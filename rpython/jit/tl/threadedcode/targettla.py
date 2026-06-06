import os
import py
py.path.local(__file__)

from rpython.rlib.rtime import time
from rpython.rlib import jit
from rpython.rlib.rstack import _stack_set_length_fraction
from rpython.rlib.rgc import increase_root_stack_depth
from rpython.rlib.rstring import StringBuilder
from rpython.jit.tl.threadedcode import tla
from rpython.jit.tl.threadedcode import frames
from rpython.jit.tl.threadedcode.bytecode import Bytecode


def _set_recursion_limit():
    # RPython's stack_check() aborts with a fatal StackOverflow once the C
    # stack passes MAX_STACK_SIZE (2.8 MB by default), independently of the OS
    # ulimit.  Deep non-tail recursion (e.g. Takeuchi) trips this long before
    # MAX_INTERP_DEPTH, and the float path costs more stack per level than the
    # int path, so it fails sooner.  Raise the ceiling the same way CPython's
    # sys.setrecursionlimit does.  TLA_STACK is a recursionlimit-style integer
    # (N reserves N/1000 * 2.8 MB); keep it under the OS ulimit to stay safe.
    limit = 4000
    env = os.environ.get('TLA_STACK')
    if env:
        limit = int(env)
    if limit < 1000:
        limit = 1000
    _stack_set_length_fraction(limit * 0.001)
    increase_root_stack_depth(int(limit * 0.001 * 163840))

def entry_point(args):
    usage = "Usage: %s filename x n" % (args[0],)

    if len(args) < 3:
        print usage
        return 2

    _set_recursion_limit()
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
        params = "inlining=0,threshold=1039,function_threshold=1039"
        th = os.environ.get('TLA_THRESHOLD')
        if th:
            params = "inlining=0,threshold=%s,function_threshold=%s" % (th, th)
        jit.set_user_param(None, params)
    elif tier >= 2:
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
        os.write(2, "CFG ratio=%d freeze=%d min=%d adaptive_inv=%d "
                    "adaptive_tier=%d\n" % (
            frames._t4cfg.ratio, frames._t4cfg.freeze, frames._t4cfg.minn,
            bytecode.adaptive_invocations, bytecode.adaptive_tier))
        t3, t4 = tla._cb_estimate(bytecode)
        thr = frames._t4cfg.cnt_base + frames._t4cfg.cnt_slope * len(bytecode)
        recomp = (frames._t4cfg.recomp_base +
                  frames._t4cfg.recomp_slope * len(bytecode))
        os.write(2, "CB cbmodel=%d observed=%d thr=%d t3=%d t4=%d recomp=%d "
                    "should_t4=%d reopt_retry=%d\n" % (
            frames._t4cfg.cbmodel, tla._cb_observed_ops(bytecode), thr,
            t3, t4, recomp, 1 if tla._cb_should_tier4(bytecode) else 0,
            bytecode.reopt_retry))
        i = 0
        while i < len(bytecode):
            ca = bytecode.cnt_a[i]
            cb = bytecode.cnt_b[i]
            if ca != 0 and cb != 0:
                os.write(2, "PROFILE site=%d cnt_a=%d cnt_b=%d bails=%d "
                            "inl_runs=%d poly=%d\n" % (
                    i, ca, cb, bytecode.bails[i], bytecode.inl_runs[i],
                    bytecode.poly[i]))
            i += 1
    return 0

def load_bytecode(filename):
    from rpython.rlib.streamio import open_file_as_stream
    if len(filename) >= 4 and filename.endswith('.tla'):
        b = StringBuilder()
        i = 0
        n = len(filename) - 4
        while i < n:
            b.append(filename[i])
            i += 1
        b.append('.tlc')
        filename = b.build()
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
