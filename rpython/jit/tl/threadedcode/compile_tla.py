#!/usr/bin/env python2
"""Compile a high-level TLA language source (.tla) into packed bytecode (.tlc).

The .tla surface language is the OCaml-flavoured, integer-only functional
language in lang/*.tla (see parser.py / grammar.txt: `let rec` functions,
`if/then/else`, `let .. in`, arithmetic/comparison, recursion).  The .tlc output
is the packed byte string that the translated `targettla-c` binary and
tla.run() load at runtime via Bytecode(...) -- i.e. the same bytes that
bench_inline.assemble() / eval_perf.py feed to `targettla-c <prog>.tlc x n`.

This is a *build-time* tool: it runs under plain CPython/PyPy2 and is never
translated.

Usage:
  python2 compile_tla.py INPUT.tla [INPUT.tla ...]   # -> INPUT.tlc beside each
  python2 compile_tla.py INPUT.tla -o OUTPUT.tlc      # single input, explicit out
  python2 compile_tla.py INPUT.tla ... -d OUTDIR      # write each .tlc into OUTDIR
  python2 compile_tla.py --all [-d OUTDIR]            # all lang/*.tla
  python2 compile_tla.py NAME                         # bare name -> lang/NAME.tla
  cat prog.tla | python2 compile_tla.py - -o prog.tlc # read source from stdin

Options:
  -o, --output FILE   output path (only valid with a single input)
  -d, --outdir DIR    directory for the .tlc outputs (created if missing)
  --all               compile every lang/*.tla
  -q, --quiet         do not print the per-file summary line
  -h, --help          show this help
"""
import os
import sys
import glob

THIS = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(THIS, '..', '..', '..', '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from rpython.jit.tl.threadedcode import parser
from rpython.jit.tl.threadedcode import bytecode as bc

SRCDIR = os.path.join(THIS, 'lang')


def compile_source_to_tlc(source):
    """TLA source text -> (bytecode int list, packed .tlc byte string)."""
    code = parser.compile_source(source)
    return code, bc.assemble(code)


def compile_file(src_path, out_path):
    if src_path == '-':
        source = sys.stdin.read()
    else:
        with open(src_path) as f:
            source = f.read()
    code, blob = compile_source_to_tlc(source)
    with open(out_path, 'wb') as f:
        f.write(blob)
    return len(code), len(blob)


def _resolve_input(arg):
    """Accept a path, a bare program name (-> lang/NAME.tla), or '-'."""
    if arg == '-' or os.path.exists(arg):
        return arg
    cand = os.path.join(SRCDIR, arg if arg.endswith('.tla') else arg + '.tla')
    if os.path.exists(cand):
        return cand
    return arg                       # report the original as not-found


def _out_path(src_path, outdir):
    if src_path == '-':
        base = 'out.tlc'
    else:
        base = os.path.basename(src_path)
        base = (base[:-4] if base.endswith('.tla') else base) + '.tlc'
    return os.path.join(outdir or os.path.dirname(os.path.abspath(src_path)), base)


def main(argv):
    outfile = outdir = None
    do_all = quiet = False
    inputs = []
    i = 0
    while i < len(argv):
        a = argv[i]
        if a in ('-o', '--output'):
            outfile = argv[i + 1]; i += 2; continue
        if a in ('-d', '--outdir'):
            outdir = argv[i + 1]; i += 2; continue
        if a == '--all':
            do_all = True; i += 1; continue
        if a in ('-q', '--quiet'):
            quiet = True; i += 1; continue
        if a in ('-h', '--help'):
            sys.stdout.write(__doc__); return 0
        inputs.append(a); i += 1

    if do_all:
        inputs = sorted(glob.glob(os.path.join(SRCDIR, '*.tla')))
        if not inputs:
            sys.stderr.write('no .tla files in %s\n' % SRCDIR); return 1
    if not inputs:
        sys.stderr.write(__doc__); return 2
    if outfile and len(inputs) != 1:
        sys.stderr.write('error: -o/--output requires exactly one input\n'); return 2
    if outdir and not os.path.isdir(outdir):
        os.makedirs(outdir)

    rc = 0
    for arg in inputs:
        src = _resolve_input(arg)
        if src != '-' and not os.path.exists(src):
            sys.stderr.write('error: not found: %s\n' % arg); rc = 1; continue
        out = outfile if outfile else _out_path(src, outdir)
        try:
            nops, nbytes = compile_file(src, out)
        except Exception as e:
            sys.stderr.write('error: %s: %s\n' % (arg, e)); rc = 1; continue
        if not quiet:
            shown = 'stdin' if src == '-' else src
            print '%s -> %s  (%d opcodes, %d bytes)' % (shown, out, nops, nbytes)
    return rc


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
