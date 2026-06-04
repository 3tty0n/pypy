#!/usr/bin/env python2
"""Disassemble a .tlc bytecode blob to learn the exact encoding."""
import sys, os
THIS = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(THIS, '..', '..', '..', '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
from rpython.jit.tl.threadedcode import bytecode as bc


def dis(code):
    i = 0
    out = []
    n = len(code)
    while i < n:
        op = ord(code[i])
        name = bc.bytecodes[op]
        nargs = bc.hasarg[op]
        args = [ord(code[i + 1 + j]) for j in range(nargs)]
        out.append("%4d: %-16s %s" % (i, name, args if args else ""))
        i += 1 + nargs
    return "\n".join(out)


if __name__ == '__main__':
    for path in sys.argv[1:]:
        with open(path, 'rb') as f:
            code = f.read()
        print "===== %s (%d bytes) =====" % (path, len(code))
        print dis(code)
