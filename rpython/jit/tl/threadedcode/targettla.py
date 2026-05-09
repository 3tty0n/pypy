import py
py.path.local(__file__)

from rpython.rlib.rtime import time
from rpython.rlib import jit
from rpython.jit.tl.threadedcode import tla
from rpython.jit.tl.threadedcode.bytecode import Bytecode
from rpython.jit.tl.threadedcode.tl_pipeline import compile_string_from_source

def entry_point(args):
    usage = (
        "Usage: %s [options] file.tl x [n]\n"
        "       %s [options] --raw-bytecode file.bin x [n]\n"
        "  file.tl: TL source (see grammar.txt); compiled then run.\n"
        "  --raw-bytecode: load pre-assembled bytecode (legacy).\n"
        "  x: initial int on the operand stack; n: repeat count (default 100)."
    ) % (args[0], args[0])

    if len(args) < 3:
        print usage
        return 2

    debug = False
    tier = 1
    raw_bytecode = False
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
        elif args[i] == "--tla-inline-depth":
            tla.set_global_inline_cap(int(args[i + 1]))
            del args[i:i+2]
            continue
        elif args[i] == "--raw-bytecode":
            raw_bytecode = True
            del args[i]
            continue
        i += 1

    filename = args[1]
    x = int(args[2])

    n = 100
    if len(args) > 3:
        n = int(args[3])

    w_x = tla.W_IntObject(x)
    bytecode = load_program(filename, raw_bytecode=raw_bytecode)
    w_res = tla.W_IntObject(0)
    for _ in range(n):
        n1 = time()
        w_res = tla.run(bytecode, w_x, debug=debug, tier=tier)
        n2 = time()
        print n2 - n1
    print w_res.getrepr()
    return 0

def load_program(filename, raw_bytecode=False):
    from rpython.rlib.streamio import open_file_as_stream
    f = open_file_as_stream(filename)
    data = f.readall()
    f.close()
    if raw_bytecode:
        return Bytecode(data)
    packed = compile_string_from_source(data)
    return Bytecode(packed)

def target(driver, args):
    return entry_point

# ____________________________________________________________


if __name__ == '__main__':
    import sys
    sys.exit(entry_point(sys.argv))
