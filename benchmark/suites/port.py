#!/usr/bin/env python3
"""Port an upstream model file to Python 2 by transformation, not by hand.

    port.py SPEC_NAME            write benchmark/suites/ports/<name>.py
    port.py --all                every entry of SPECS
    port.py --check              regenerate in memory, diff against disk

The output is the upstream source run through a fixed set of syntax-only
rewrites, restricted to the definitions the model needs:

  - annotations removed (parameters, returns, annotated assignments)
  - f-strings become str.format calls with the same fields
  - super() inside a method becomes super(Class, self)
  - keyword-only parameters become ordinary parameters with their defaults
  - typing.cast(T, v) becomes v (it is the identity at run time)
  - only the SPEC's `keep` top-level definitions (functions, classes,
    assignments) are emitted, in upstream order
  - imports are replaced by the port header's (torch, torch.nn as nn, ...)

Anything beyond syntax is listed in the SPEC as an explicit exception, each
with its reason: `drop_calls` removes statements that call a named function
(usage logging, weight downloads), and `prelude` supplies a standard-library
name Python 2 lacks; nothing else is changed.  The header
of each output names the upstream file, its package version and the sha256
of the source it was made from, so the port can be regenerated and checked.
"""
import ast
import hashlib
import importlib
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "ports")
TORCHBENCH = os.environ.get("TORCHBENCH", os.path.expanduser(
    "~/src/github.com/pytorch/benchmark"))
sys.path.append(TORCHBENCH)

SIMPLE_NAMESPACE = """\
class SimpleNamespace(object):
    def __init__(self, **kw):
        self.__dict__.update(kw)
"""

HEADER_IMPORTS = """\
import math
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
"""

SPECS = {
    "torchvision_resnet": dict(
        module="torchvision.models.resnet", package="torchvision",
        keep=["conv3x3", "conv1x1", "BasicBlock", "Bottleneck", "ResNet"],
        drop_calls={"_log_api_usage_once": "usage telemetry, no numerics"},
    ),
    "torchbench_phlippe_resnet": dict(
        module="torchbenchmark.models.phlippe_resnet", package="torchbenchmark",
        keep=["ResNetBlock", "ResNetModel"],
        prelude={"types.SimpleNamespace": SIMPLE_NAMESPACE},
    ),
    "torchvision_alexnet": dict(
        module="torchvision.models.alexnet", package="torchvision",
        keep=["AlexNet"],
        drop_calls={"_log_api_usage_once": "usage telemetry, no numerics"},
    ),
    "torchvision_vgg": dict(
        module="torchvision.models.vgg", package="torchvision",
        keep=["VGG", "make_layers", "cfgs"],
        drop_calls={"_log_api_usage_once": "usage telemetry, no numerics"},
    ),
}


class Py2(ast.NodeTransformer):
    def __init__(self, drop_calls, typing_names):
        self.drop_calls = drop_calls
        self.typing_names = typing_names
        self.cls = []

    def visit_ClassDef(self, node):
        self.cls.append(node.name)
        self.generic_visit(node)
        self.cls.pop()
        return node

    def visit_FunctionDef(self, node):
        a = node.args
        for arg in a.posonlyargs + a.args + a.kwonlyargs:
            arg.annotation = None
        if a.vararg:
            a.vararg.annotation = None
        if a.kwarg:
            a.kwarg.annotation = None
        if a.kwonlyargs:
            if a.vararg:
                raise NotImplementedError(
                    "%s: keyword-only after *args" % node.name)
            n_pos = len(a.posonlyargs + a.args)
            defaults = [None] * (n_pos - len(a.defaults)) + list(a.defaults)
            for arg, d in zip(a.kwonlyargs, a.kw_defaults):
                a.args.append(arg)
                defaults.append(d)
            first = next((i for i, d in enumerate(defaults)
                          if d is not None), len(defaults))
            if any(d is None for d in defaults[first:]):
                raise NotImplementedError(
                    "%s: a keyword-only parameter without a default follows "
                    "one with a default" % node.name)
            a.defaults = [x for x in defaults if x is not None]
            a.kwonlyargs, a.kw_defaults = [], []
        a.args = a.posonlyargs + a.args
        a.posonlyargs = []
        node.returns = None
        node.type_params = []
        self.generic_visit(node)
        return node

    def visit_AnnAssign(self, node):
        if node.value is None:
            return None
        return ast.copy_location(
            ast.Assign(targets=[node.target], value=self.visit(node.value)),
            node)

    def visit_Expr(self, node):
        c = node.value
        if isinstance(c, ast.Call):
            f = c.func
            name = f.id if isinstance(f, ast.Name) else (
                f.attr if isinstance(f, ast.Attribute) else None)
            if name in self.drop_calls:
                return None
        self.generic_visit(node)
        return node

    def visit_Call(self, node):
        self.generic_visit(node)
        f = node.func
        if (isinstance(f, ast.Name) and f.id == "cast" and
                "cast" in self.typing_names and len(node.args) == 2):
            return node.args[1]
        if (isinstance(f, ast.Name) and f.id == "super" and not node.args
                and self.cls):
            node.args = [ast.Name(self.cls[-1], ast.Load()),
                         ast.Name("self", ast.Load())]
        return node

    def visit_JoinedStr(self, node):
        fmt, args = [], []
        for part in node.values:
            if isinstance(part, ast.Constant):
                fmt.append(part.value.replace("{", "{{").replace("}", "}}"))
            else:
                spec = ""
                if part.format_spec is not None:
                    spec = ":" + "".join(
                        v.value for v in part.format_spec.values)
                conv = "!" + chr(part.conversion) if part.conversion != -1 \
                    else ""
                fmt.append("{%d%s%s}" % (len(args), conv, spec))
                args.append(self.visit(part.value))
        return ast.copy_location(ast.Call(
            func=ast.Attribute(ast.Constant("".join(fmt)), "format",
                               ast.Load()),
            args=args, keywords=[]), node)


def defined(node):
    """The top-level names a statement binds."""
    if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
        return {node.name}
    targets = node.targets if isinstance(node, ast.Assign) else (
        [node.target] if isinstance(node, ast.AnnAssign) else [])
    return {t.id for t in targets if isinstance(t, ast.Name)}


def version(package):
    if package == "torchbenchmark":
        return "git " + subprocess.run(
            ["git", "-C", TORCHBENCH, "rev-parse", "--short=10", "HEAD"],
            capture_output=True, text=True).stdout.strip()
    return importlib.import_module(package).__version__


def port(name):
    spec = SPECS[name]
    mod = importlib.import_module(spec["module"])
    path = mod.__file__
    src = open(path).read()
    tree = ast.parse(src)
    keep = set(spec["keep"])
    body = [n for n in tree.body if defined(n) & keep]
    missing = keep - set().union(*[defined(n) for n in body])
    if missing:
        raise SystemExit("%s: not found upstream: %s" % (name, missing))
    typing_names = set()
    for n in tree.body:
        if isinstance(n, ast.ImportFrom) and n.module == "typing":
            typing_names |= {a.asname or a.name for a in n.names}
    tree.body = body
    tree = Py2(spec.get("drop_calls", {}), typing_names).visit(tree)
    ast.fix_missing_locations(tree)
    exceptions = "".join("#   dropped calls to %s: %s\n" % kv
                         for kv in sorted(spec.get("drop_calls", {}).items()))
    prelude = spec.get("prelude", {})
    exceptions += "".join("#   %s supplied for Python 2\n" % k
                          for k in sorted(prelude))
    header = (
        "# Generated by benchmark/suites/port.py from %s\n"
        "# (%s %s, source sha256 %s).  Do not edit; change the spec.\n"
        "# Kept: %s\n%s"
        % (spec["module"], spec["package"], version(spec["package"]),
           hashlib.sha256(src.encode()).hexdigest()[:16],
           ", ".join(spec["keep"]), exceptions))
    return (header + HEADER_IMPORTS + "".join(
        "\n\n" + prelude[k] for k in sorted(prelude)) + "\n\n" +
        ast.unparse(tree) + "\n")


def write(name):
    os.makedirs(OUT, exist_ok=True)
    text = port(name)
    dst = os.path.join(OUT, name + ".py")
    open(dst, "w").write(text)
    py2 = os.environ.get("PYTHON2", "python2")
    r = subprocess.run([py2, "-c", "import sys; compile(open(sys.argv[1])"
                        ".read(), sys.argv[1], 'exec')", dst],
                       capture_output=True, text=True)
    if r.returncode:
        raise SystemExit("%s does not compile as Python 2:\n%s"
                         % (dst, r.stderr))
    print("wrote %s (%d lines)" % (dst, text.count("\n")))


def main(argv):
    if argv[1:2] == ["--all"]:
        for n in SPECS:
            write(n)
    elif argv[1:2] == ["--check"]:
        bad = 0
        for n in SPECS:
            dst = os.path.join(OUT, n + ".py")
            if not os.path.exists(dst) or open(dst).read() != port(n):
                print("stale: %s" % dst)
                bad += 1
        return 1 if bad else 0
    else:
        write(argv[1])
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
