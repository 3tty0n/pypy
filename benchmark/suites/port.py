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
  - upstream's imports are kept, relative ones made absolute, and filtered
    to the names the kept code still uses; `torch` and `transformers`
    resolve to the packages in lib_pypy/tensorpypy/compat
  - (*a, b) and [*a, b] become tuple(a) + (b,) and list(a) + [b], and
    f(*a, b) becomes f(*(tuple(a) + (b,)))
  - `raise X from Y` becomes `raise X`
  - every port starts with `from __future__ import absolute_import,
    division, print_function`, so / and imports mean what they meant

Constructs Python 2 has no rewrite for here (keyword-only parameters after
*args, {**d}, :=, nonlocal, yield from, async, match)
stop the port with an error rather than pass through.

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
COMPAT = os.path.join(HERE, "..", "..", "lib_pypy", "tensorpypy", "compat")
TORCHBENCH = os.environ.get("TORCHBENCH", os.path.expanduser(
    "~/src/github.com/pytorch/benchmark"))
sys.path.append(TORCHBENCH)

SIMPLE_NAMESPACE = """\
class SimpleNamespace(object):
    def __init__(self, **kw):
        self.__dict__.update(kw)
"""

FUTURE = "from __future__ import absolute_import, division, print_function\n"

HF_ACT_KEEP = ["logger", "GELUTanh", "PytorchGELUTanh", "NewGELUActivation",
               "GELUActivation", "SiLUActivation", "FastGELUActivation",
               "QuickGELUActivation", "ClippedGELUActivation",
               "AccurateGELUActivation", "MishActivation",
               "LinearActivation", "LaplaceActivation",
               "ReLUSquaredActivation", "SqrtSoftplusActivation",
               "ClassInstantier", "XIELUActivation", "ACT2CLS", "ACT2FN",
               "get_activation", "gelu"]

INSPECT_SIGNATURE = """\
class _Signature(object):
    def __init__(self, parameters):
        self.parameters = parameters


class inspect(object):
    \"\"\"inspect.signature(f).parameters, counted as Python 3 counts them:
    a bound method's self is not a parameter.  Memoised per function, as
    a signature never changes.\"\"\"
    _memo = {}

    @staticmethod
    def signature(f):
        fn = getattr(f, "__func__", f)
        bound = getattr(f, "__self__", None) is not None
        key = (fn, bound)
        sig = inspect._memo.get(key)
        if sig is None:
            import inspect as _inspect
            spec = _inspect.getargspec(fn)
            names = list(spec.args)[1 if bound else 0:]
            if spec.varargs:
                names.append(spec.varargs)
            if spec.keywords:
                names.append(spec.keywords)
            sig = inspect._memo[key] = _Signature(tuple(names))
        return sig
"""

SPECS = {
    # transformers' own infrastructure the modeling files call, ported into
    # the compat transformers package next to the hand-written stand-ins
    "transformers.activations": dict(
        module="transformers.activations", package="transformers",
        keep=HF_ACT_KEEP,
        out="transformers/activations.py"),
    "transformers.pytorch_utils": dict(
        module="transformers.pytorch_utils", package="transformers",
        keep=["apply_chunking_to_forward", "Conv1D"],
        prelude={"inspect": INSPECT_SIGNATURE},
        out="transformers/pytorch_utils.py"),
    "transformers.modeling_layers": dict(
        module="transformers.modeling_layers", package="transformers",
        keep=["logger", "GradientCheckpointingLayer"],
        out="transformers/modeling_layers.py"),
    "transformers.integrations.sdpa_attention": dict(
        module="transformers.integrations.sdpa_attention",
        package="transformers",
        keep=["logger", "_is_torch_greater_or_equal_than_2_5",
              "_is_torch_greater_or_equal_than_2_8", "_is_torch_xpu_available",
              "_is_torch_npu_available", "repeat_kv", "use_gqa_in_sdpa",
              "sdpa_attention_forward"],
        out="transformers/integrations/sdpa_attention.py"),
    "hf_distilbert": dict(
        module="transformers.models.distilbert.modeling_distilbert",
        package="transformers",
        keep=["logger", "Embeddings", "eager_attention_forward",
              "DistilBertSelfAttention", "FFN", "TransformerBlock",
              "Transformer", "DistilBertPreTrainedModel", "DistilBertModel",
              "DistilBertForMaskedLM"],
        free_ok={"create_sinusoidal_embeddings":
                 "read by resize_position_embeddings only"},
    ),
    "hf_bert": dict(
        module="transformers.models.bert.modeling_bert",
        package="transformers",
        keep=["logger", "BertEmbeddings", "eager_attention_forward",
              "BertSelfAttention", "BertCrossAttention", "BertSelfOutput",
              "BertAttention", "BertIntermediate", "BertOutput", "BertLayer",
              "BertEncoder", "BertPooler", "BertPredictionHeadTransform",
              "BertLMPredictionHead", "BertOnlyMLMHead",
              "BertPreTrainedModel", "BertModel", "BertForMaskedLM"],
    ),
    "torchbench_eos_pytorch": dict(
        module="torchbenchmark.models.pyhpc_equation_of_state.eos_pytorch",
        package="torchbenchmark", keep=["gsw_dHdT"],
    ),
    "torchbench_pyhpc_equation_of_state": dict(
        module="torchbenchmark.models.pyhpc_equation_of_state",
        package="torchbenchmark", keep=["EquationOfState"],
    ),
    "hf_xlm_roberta": dict(
        module="transformers.models.xlm_roberta.modeling_xlm_roberta",
        package="transformers",
        keep=["logger", "XLMRobertaEmbeddings", "eager_attention_forward",
              "XLMRobertaSelfAttention", "XLMRobertaCrossAttention",
              "XLMRobertaSelfOutput", "XLMRobertaAttention",
              "XLMRobertaIntermediate", "XLMRobertaOutput", "XLMRobertaLayer",
              "XLMRobertaLMHead", "XLMRobertaPreTrainedModel",
              "XLMRobertaEncoder", "XLMRobertaPooler", "XLMRobertaModel",
              "XLMRobertaForMaskedLM"],
    ),
    "hf_albert": dict(
        module="transformers.models.albert.modeling_albert",
        package="transformers",
        keep=["logger", "AlbertEmbeddings", "eager_attention_forward",
              "AlbertAttention", "AlbertLayer", "AlbertLayerGroup",
              "AlbertTransformer", "AlbertPreTrainedModel", "AlbertModel",
              "AlbertMLMHead", "AlbertForMaskedLM"],
    ),
    "hf_gpt2": dict(
        module="transformers.models.gpt2.modeling_gpt2",
        package="transformers",
        keep=["logger", "eager_attention_forward", "GPT2Attention",
              "GPT2MLP", "GPT2Block", "GPT2PreTrainedModel", "GPT2Model",
              "GPT2LMHeadModel"],
    ),
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


def _tuple(elts):
    return ast.Tuple(list(elts), ast.Load())


def _list(elts):
    return ast.List(list(elts), ast.Load())


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

    def _starred_seq(self, node, make):
        if not any(isinstance(e, ast.Starred) for e in node.elts):
            return node
        parts, run = [], []
        for e in node.elts:
            if isinstance(e, ast.Starred):
                if run:
                    parts.append(make(run))
                    run = []
                conv = "tuple" if make is _tuple else "list"
                parts.append(ast.Call(ast.Name(conv, ast.Load()),
                                      [e.value], []))
            else:
                run.append(e)
        if run:
            parts.append(make(run))
        out = parts[0]
        for p in parts[1:]:
            out = ast.BinOp(out, ast.Add(), p)
        return ast.copy_location(out, node)

    def visit_Tuple(self, node):
        self.generic_visit(node)
        if isinstance(node.ctx, ast.Load):
            return self._starred_seq(node, _tuple)
        return node

    def visit_List(self, node):
        self.generic_visit(node)
        if isinstance(node.ctx, ast.Load):
            return self._starred_seq(node, _list)
        return node

    def visit_Raise(self, node):
        self.generic_visit(node)
        node.cause = None
        return node

    def visit_Dict(self, node):
        if any(k is None for k in node.keys):
            raise NotImplementedError("{**d} display")
        self.generic_visit(node)
        return node

    def _refuse(self, node):
        raise NotImplementedError("%s at line %d" % (
            type(node).__name__, node.lineno))

    visit_NamedExpr = visit_Nonlocal = visit_YieldFrom = _refuse
    visit_AsyncFunctionDef = visit_Await = visit_Match = _refuse

    def visit_Call(self, node):
        self.generic_visit(node)
        stars = [a for a in node.args if isinstance(a, ast.Starred)]
        if len(stars) > 1 or (stars and node.args[-1] is not stars[0]):
            # f(*a, b) -> f(*(tuple(a) + (b,)))
            seq = self._starred_seq(ast.Tuple(node.args, ast.Load()), _tuple)
            node.args = [ast.Starred(seq, ast.Load())]
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


PY2_BUILTINS = set(dir(__builtins__)) - {"print"} | {
    "unicode", "long", "xrange", "basestring", "reduce", "__name__",
    "__file__", "__doc__"}


def free_names(tree):
    """Names the code reads that nothing in it binds and Python has not
    built in: each is a NameError waiting for the line that reads it."""
    bound, read = set(), set()
    for n in ast.walk(tree):
        if isinstance(n, ast.Name):
            (read if isinstance(n.ctx, ast.Load) else bound).add(n.id)
        elif isinstance(n, (ast.FunctionDef, ast.ClassDef)):
            bound.add(n.name)
        elif isinstance(n, ast.arg):
            bound.add(n.arg)
        elif isinstance(n, (ast.Import, ast.ImportFrom)):
            bound |= {(a.asname or a.name).split(".")[0] for a in n.names}
        elif isinstance(n, ast.ExceptHandler) and n.name:
            bound.add(n.name)
    return sorted(read - bound - PY2_BUILTINS)


def used_names(tree):
    out = set()
    for n in ast.walk(tree):
        if isinstance(n, ast.Name):
            out.add(n.id)
    return out


def imports(tree, module, used, supplied, is_package, ported):
    """Upstream's top-level imports, absolute, restricted to used names; an
    import of a module that has a port of its own imports the port."""
    pkg = module.split(".") + ([""] if is_package else [])
    out = []
    for n in tree.body:
        if isinstance(n, ast.Import):
            names = [a for a in n.names
                     if (a.asname or a.name.split(".")[0]) in used
                     and (a.asname or a.name) not in supplied]
            if names:
                out.append(ast.Import(names))
        elif isinstance(n, ast.ImportFrom):
            if n.module == "__future__":
                continue
            base = n.module or ""
            if n.level:
                base = ".".join(pkg[:len(pkg) - n.level] +
                                ([n.module] if n.module else []))
            names = [a for a in n.names if (a.asname or a.name) in used
                     and (a.asname or a.name) not in supplied]
            rest = []
            for a in names:
                full = base + "." + a.name
                if full in ported:
                    out.append(ast.Import([ast.alias(ported[full],
                                                     a.asname or a.name)]))
                else:
                    rest.append(a)
            if rest:
                out.append(ast.ImportFrom(base, rest, 0))
    return out


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
    kept = ast.Module(body=body, type_ignores=[])
    kept = Py2(spec.get("drop_calls", {}), typing_names).visit(kept)
    prelude = spec.get("prelude", {})
    supplied = {k.rsplit(".", 1)[-1] for k in prelude}
    ported = dict((v["module"], k) for k, v in SPECS.items()
                   if "out" not in v)
    imps = imports(tree, spec["module"], used_names(kept), supplied,
                   path.endswith("__init__.py"), ported)
    kept.body = imps + kept.body
    ast.fix_missing_locations(kept)
    free = (set(free_names(kept)) - set(spec.get("free_ok", {})) -
            {k.rsplit(".", 1)[-1] for k in spec.get("prelude", {})})
    if free:
        raise SystemExit("%s: free names %s (keep them, or list them in "
                         "free_ok with the reason they are never read)"
                         % (name, sorted(free)))
    exceptions = "".join("#   dropped calls to %s: %s\n" % kv
                         for kv in sorted(spec.get("drop_calls", {}).items()))
    exceptions += "".join("#   %s supplied for Python 2\n" % k
                          for k in sorted(prelude))
    exceptions += "".join("#   %s left unbound: %s\n" % kv
                          for kv in sorted(spec.get("free_ok", {}).items()))
    header = (
        "# -*- coding: utf-8 -*-\n"
        "# Generated by benchmark/suites/port.py from %s\n"
        "# (%s %s, source sha256 %s).  Do not edit; change the spec.\n"
        "# Kept: %s\n%s"
        % (spec["module"], spec["package"], version(spec["package"]),
           hashlib.sha256(src.encode()).hexdigest()[:16],
           ", ".join(spec["keep"]), exceptions))
    text = ast.unparse(kept)
    if prelude:
        # after the imports, before the first kept definition
        split = len(ast.unparse(ast.Module(body=imps, type_ignores=[])))
        text = (text[:split] + "".join("\n\n" + prelude[k]
                                       for k in sorted(prelude)) +
                text[split:])
    return header + FUTURE + text + "\n"


def destination(name):
    spec = SPECS[name]
    if "out" in spec:
        return os.path.join(COMPAT, spec["out"])
    return os.path.join(OUT, name + ".py")


def write(name):
    text = port(name)
    dst = destination(name)
    os.makedirs(os.path.dirname(dst), exist_ok=True)
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
            dst = destination(n)
            if not os.path.exists(dst) or open(dst).read() != port(n):
                print("stale: %s" % dst)
                bad += 1
        return 1 if bad else 0
    else:
        write(argv[1])
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
