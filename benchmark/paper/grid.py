"""Read benchmarks.toml, the one place the microbenchmark grid is written down.

    grid.py points              # "variant k n" per line, the float64 grid
    grid.py precision           # "variant k n dtype" per line, the sweep
    grid.py labels              # "variant<TAB>label" per line
"""

import os
import sys

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10 and older
    import tomli as tomllib

DEFAULT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "benchmarks.toml")


def load(path=None):
    with open(path or os.environ.get("BENCHMARKS_TOML") or DEFAULT, "rb") as f:
        return tomllib.load(f)


def points(cfg):
    for b in cfg.get("benchmark", []):
        for k in b.get("k", [1]):
            for n in b["sizes"]:
                yield b["variant"], k, n


def precision_points(cfg):
    p = cfg.get("precision")
    if not p:
        return
    for v in p["variants"]:
        for n in p["sizes"]:
            for dtype in p["dtypes"]:
                yield v, 1, n, dtype


def labels(cfg):
    out = {}
    for b in cfg.get("benchmark", []):
        v, label = b["variant"], b["label"]
        if out.setdefault(v, label) != label:
            raise ValueError("variant %d has two labels: %r and %r"
                             % (v, out[v], label))
    return out


def main(argv):
    what = argv[1] if len(argv) > 1 else "points"
    cfg = load(argv[2] if len(argv) > 2 else None)
    if what == "points":
        for v, k, n in points(cfg):
            print("%d %d %d" % (v, k, n))
    elif what == "precision":
        for v, k, n, dtype in precision_points(cfg):
            print("%d %d %d %s" % (v, k, n, dtype))
    elif what == "labels":
        for v, label in sorted(labels(cfg).items()):
            print("%d\t%s" % (v, label))
    else:
        print("usage: grid.py [points|precision|labels] [benchmarks.toml]",
              file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
