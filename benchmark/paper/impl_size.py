"""Non-blank, non-comment line counts per MetaTensor component.

    benchmark/paper/impl_size.py               # markdown table to stdout
    benchmark/paper/impl_size.py --tex PATH     # also write a .tex table
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import plot  # reuse write_table's exact format

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def count(path):
    n = 0
    with open(path) as f:
        for line in f:
            s = line.strip()
            if s and not s.startswith("#"):
                n += 1
    return n


def group(paths):
    return sum(count(os.path.join(REPO, p)) for p in paths if os.path.exists(os.path.join(REPO, p)))


COMPONENTS = [
    ("Triton codegen", ["rpython/metatensor/ttir.py", "rpython/metatensor/triton_compile.py"]),
    ("kernel cache/launch", ["rpython/metatensor/kernels.py"]),
    ("device runtime", ["rpython/metatensor/device.py", "rpython/metatensor/devops.py",
                        "rpython/metatensor/cuda.c"]),
    ("tensor ops", ["rpython/metatensor/core.py", "rpython/metatensor/ops.py",
                    "rpython/metatensor/nn.py", "rpython/metatensor/runtime.py"]),
    ("JIT fusion pass", ["rpython/jit/metainterp/optimizeopt/metatensor.py"]),
    ("app-level module", ["pypy/module/_metatensor/interp_tensor.py",
                          "pypy/module/_metatensor/moduledef.py",
                          "pypy/module/_metatensor/app_tensor.py"]),
    ("Python library", ["lib_pypy/tensorpypy/__init__.py", "lib_pypy/tensorpypy/functional.py",
                        "lib_pypy/tensorpypy/models.py", "lib_pypy/tensorpypy/nn.py",
                        "lib_pypy/tensorpypy/optim.py"]),
]

TEST_COMPONENTS = [
    ("rpython/metatensor tests", ["rpython/metatensor/test/test_tensor.py"]),
    ("pypy/module/_metatensor tests", ["pypy/module/_metatensor/test/test_tensor.py",
                                       "pypy/module/_metatensor/test/test_bert.py",
                                       "pypy/module/_metatensor/test/test_llama.py",
                                       "pypy/module/_metatensor/test/test_mixer.py",
                                       "pypy/module/_metatensor/test/test_resnet.py",
                                       "pypy/module/_metatensor/test/test_vit.py"]),
]


def rows():
    return [(name, group(paths)) for name, paths in COMPONENTS] + \
           [(name, group(paths)) for name, paths in TEST_COMPONENTS]


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--tex", help="also write a .tex table here")
    args = p.parse_args()

    data = rows()
    total = sum(n for _, n in data if "tests" not in _)

    print("| component | lines |")
    print("|---|---|")
    for name, n in data:
        print("| %s | %d |" % (name, n))
    print("| **total (impl)** | **%d** |" % total)

    if args.tex:
        os.makedirs(os.path.dirname(args.tex), exist_ok=True)
        plot.write_table(args.tex, "non-blank, non-comment lines per component",
                          ["component", "lines"],
                          [[name, str(n)] for name, n in data] +
                          [["total (impl)", str(total)]])
        print("wrote %s" % args.tex, file=sys.stderr)


if __name__ == "__main__":
    main()
