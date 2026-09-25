#!/usr/bin/env python3
"""Port an upstream model file to Python 2 by transformation, not by hand.

    port.py SPEC_NAME            write benchmark/suites/ports/<name>.py
    port.py --all                every entry of SPECS
    port.py --check              regenerate in memory, diff against disk
    port.py --generated          compat files generated from upstream data
                                 (transformers' ModelOutput field lists)

The output is the upstream source run through a fixed set of syntax-only
rewrites, restricted to the definitions the model needs:

  - annotations removed (parameters, returns, annotated assignments); a
    @dataclass class keeps its annotated names, in order, as the tuple
    __annotations__, since they are its fields
  - f-strings become str.format calls with the same fields
  - super() inside a method becomes super(Class, self)
  - keyword-only parameters become ordinary parameters with their defaults
  - typing.cast(T, v) becomes v (it is the identity at run time)
  - only the SPEC's `keep` top-level definitions (functions, classes,
    assignments) are emitted, in upstream order, together with every
    top-level definition they reference, transitively
  - upstream's imports are kept, relative ones made absolute, and filtered
    to the names the kept code still uses; `torch` and `transformers`
    resolve to the packages in lib_pypy/tensorpypy/compat
  - (*a, b) and [*a, b] become tuple(a) + (b,) and list(a) + [b], and
    f(*a, b) becomes f(*(tuple(a) + (b,)))
  - f(**a, b=1, **c) becomes f(**dict(list(a.items()) + [('b', 1)] +
    list(c.items()))), {**a, 'k': v} becomes dict(list(a.items()) +
    [('k', v)]), and a, *b = x becomes a, b = (lambda _t: _t[:1] +
    (list(_t[1:len(_t) - 0]),) + _t[len(_t) - 0:])(tuple(x))
  - `raise X from Y` becomes `raise X`
  - `a @ b` becomes `a.__matmul__(b)`, the method Python 3 calls for it
  - an if-test's (name := value), evaluated unconditionally (not under
    and/or, a conditional expression, a comprehension or a lambda), becomes
    `name = value` before the if
  - an import from a standard-library module that Python 2 names
    differently (collections.abc) imports from the Python 2 name
  - every port starts with `from __future__ import absolute_import,
    division, print_function`, so / and imports mean what they meant

Constructs Python 2 has no rewrite for here (keyword-only parameters after
*args, a conditionally evaluated :=, :=, nonlocal, yield from, async, match)
stop the port with an error rather than pass through.

Anything beyond syntax is listed in the SPEC as an explicit exception, each
with its reason: `drop_calls` removes statements that call a named function
(usage logging, weight downloads), `keep_calls` keeps the top-level
statements that call one (a registration the kept code looks up; the calls'
arguments must be kept too), and `prelude` supplies a standard-library
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

DATACLASS = """\
def dataclass(cls=None, **options):
    \"\"\"dataclasses.dataclass for a class built from its field defaults and
    keywords, the way the ported models build theirs.  Python 2 keeps no
    order for class attributes, so positional fields are refused.\"\"\"
    def wrap(cls):
        fields = [k for k, v in vars(cls).items()
                  if not k.startswith("_") and not callable(v)]

        def __init__(self, *args, **kw):
            if args:
                raise TypeError("%s: positional dataclass fields" %
                                cls.__name__)
            for k in fields:
                setattr(self, k, kw.pop(k, getattr(cls, k)))
            if kw:
                raise TypeError("%s has no field %s" % (cls.__name__,
                                                        sorted(kw)[0]))
        cls.__init__ = __init__
        return cls
    return wrap(cls) if cls is not None else wrap
"""

MATH_COMB = """\
def comb(n, k):
    \"\"\"math.comb: the binomial coefficient, 0 for k > n.\"\"\"
    if k < 0 or k > n:
        return 0
    r = 1
    for i in range(min(k, n - k)):
        r = r * (n - i) // (i + 1)
    return r
"""

MATH3 = """\
import math as _math2


class _Math3(object):
    \"\"\"The math module with Python 3's ceil and floor, which return ints
    (Python 2's return floats, and a float channel count or padding is not
    what upstream computes).\"\"\"

    def __getattr__(self, name):
        return getattr(_math2, name)

    @staticmethod
    def ceil(x):
        return int(_math2.ceil(x))

    @staticmethod
    def floor(x):
        return int(_math2.floor(x))


math = _Math3()
"""

ROUND3 = """\
def round(number, ndigits=None):
    \"\"\"Python 3's round(x): the nearest int, ties to even (Python 2's
    returns a float and rounds ties away from zero).\"\"\"
    if ndigits is not None:
        raise NotImplementedError("round with ndigits")
    import math
    f = math.floor(number)
    diff = number - f
    if diff > 0.5 or (diff == 0.5 and int(f) % 2):
        f += 1
    return int(f)
"""

DC_DATACLASS = """\
class _Field(object):
    \"\"\"What dataclasses.field() returns: a default_factory.\"\"\"
    def __init__(self, default_factory):
        self.default_factory = default_factory


_REQUIRED = object()


def dataclass(cls=None, **options):
    \"\"\"dataclasses.dataclass: the fields are the annotated names, bases'
    first, in order; __init__ takes them by position or keyword, a field
    without a default is required, a field(default_factory=f) default is
    f() per instance, and __post_init__ runs after; __eq__ compares the
    fields in order.\"\"\"
    if options:
        raise NotImplementedError("dataclass(%s)" % ", ".join(sorted(options)))

    def wrap(cls):
        import inspect
        mro = inspect.getmro(cls)  # a Python 2 class statement without
        names = []                 # bases makes a class with no __mro__
        for c in reversed(mro):
            for k in c.__dict__.get("__annotations__", ()):
                if k not in names:
                    names.append(k)

        def default(k):
            for c in mro:
                if k in c.__dict__:
                    d = c.__dict__[k]
                    return d.default_factory() if isinstance(d, _Field) else d
            return _REQUIRED

        def __init__(self, *args, **kw):
            if len(args) > len(names):
                raise TypeError("%s takes %d fields" % (cls.__name__,
                                                        len(names)))
            for i, k in enumerate(names):
                if i < len(args):
                    if k in kw:
                        raise TypeError("%s: multiple values for %s"
                                        % (cls.__name__, k))
                    v = args[i]
                else:
                    v = kw.pop(k) if k in kw else default(k)
                    if v is _REQUIRED:
                        raise TypeError("%s: missing field %s"
                                        % (cls.__name__, k))
                setattr(self, k, v)
            if kw:
                raise TypeError("%s has no field %s" % (cls.__name__,
                                                        sorted(kw)[0]))
            if hasattr(self, "__post_init__"):
                self.__post_init__()

        def __eq__(self, other):
            if other.__class__ is not self.__class__:
                return NotImplemented
            return all(getattr(self, k) == getattr(other, k) for k in names)
        cls.__init__ = __init__
        cls.__eq__ = __eq__
        cls.__ne__ = lambda self, other: not __eq__(self, other)
        cls.__dataclass_fields__ = tuple(names)
        return cls
    return wrap(cls) if cls is not None else wrap
"""

DC_FIELD = """\
def field(default_factory=None, **kw):
    \"\"\"dataclasses.field, for the default_factory form.\"\"\"
    if kw or default_factory is None:
        raise NotImplementedError("field(%s)" % ", ".join(sorted(kw)))
    return _Field(default_factory)
"""

DC_REPLACE = """\
def replace(obj, **changes):
    \"\"\"dataclasses.replace: a new instance, through __init__, with the
    fields of obj and the changes.\"\"\"
    kw = dict((k, getattr(obj, k)) for k in obj.__dataclass_fields__)
    kw.update(changes)
    return obj.__class__(**kw)
"""

TYPING_ABCS = """\
from collections import Callable, Sequence
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
    "transformers.loss.loss_utils": dict(
        module="transformers.loss.loss_utils", package="transformers",
        keep=["fixed_cross_entropy", "ForCausalLMLoss", "ForMaskedLMLoss",
              "ForSequenceClassificationLoss", "ForQuestionAnsweringLoss",
              "ForTokenClassification"],
        out="transformers/loss/loss_utils.py"),
    "torchvision.utils": dict(
        module="torchvision.utils", package="torchvision",
        keep=["_make_ntuple"], out="torchvision/utils.py"),
    "torchvision.models._utils": dict(
        module="torchvision.models._utils", package="torchvision",
        keep=["_make_divisible"], out="torchvision/models/_utils.py"),
    "torchvision.ops.misc": dict(
        module="torchvision.ops.misc", package="torchvision",
        keep=["ConvNormActivation", "Conv2dNormActivation",
              "SqueezeExcitation", "FrozenBatchNorm2d"],
        drop_calls={"_log_api_usage_once": "usage telemetry, no numerics"},
        out="torchvision/ops/misc.py"),
    "transformers.modeling_rope_utils": dict(
        module="transformers.modeling_rope_utils", package="transformers",
        keep=["ROPE_INIT_FUNCTIONS", "dynamic_rope_update"],
        out="transformers/modeling_rope_utils.py"),
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
              "GPT2LMHeadModel", "GPT2ForSequenceClassification"],
    ),
    "hf_electra": dict(
        module="transformers.models.electra.modeling_electra",
        package="transformers", keep=["logger"] + ['ElectraForCausalLM'],
    ),
    "hf_roberta": dict(
        module="transformers.models.roberta.modeling_roberta",
        package="transformers", keep=["logger"] + ['RobertaForCausalLM'],
    ),
    "hf_megatron_bert": dict(
        module="transformers.models.megatron_bert.modeling_megatron_bert",
        package="transformers", keep=["logger"] + ['MegatronBertForCausalLM'],
    ),
    "hf_layoutlm": dict(
        module="transformers.models.layoutlm.modeling_layoutlm",
        package="transformers", keep=["logger"] + ['LayoutLMForMaskedLM'],
    ),
    "hf_bart": dict(
        module="transformers.models.bart.modeling_bart",
        package="transformers", keep=["logger"] + ['BartForCausalLM'],
    ),
    "hf_mbart": dict(
        module="transformers.models.mbart.modeling_mbart",
        package="transformers", keep=["logger"] + ['MBartForCausalLM'],
    ),
    "hf_plbart": dict(
        module="transformers.models.plbart.modeling_plbart",
        package="transformers", keep=["logger"] + ['PLBartForCausalLM'],
    ),
    "hf_blenderbot": dict(
        module="transformers.models.blenderbot.modeling_blenderbot",
        package="transformers", keep=["logger"] + ['BlenderbotForCausalLM'],
    ),
    "hf_pegasus": dict(
        module="transformers.models.pegasus.modeling_pegasus",
        package="transformers", keep=["logger"] + ['PegasusForCausalLM'],
    ),
    "hf_trocr": dict(
        module="transformers.models.trocr.modeling_trocr",
        package="transformers", keep=["logger"] + ['TrOCRForCausalLM'],
    ),
    "hf_opt": dict(
        module="transformers.models.opt.modeling_opt",
        package="transformers", keep=["logger"] + ['OPTForCausalLM'],
    ),
    "hf_xglm": dict(
        module="transformers.models.xglm.modeling_xglm",
        package="transformers", keep=["logger"] + ['XGLMForCausalLM'],
    ),
    "hf_gpt_neo": dict(
        module="transformers.models.gpt_neo.modeling_gpt_neo",
        package="transformers", keep=["logger"] + ['GPTNeoForCausalLM', 'GPTNeoForSequenceClassification'],
        free_ok={"_flash_attention_forward":
                 "read only when the config asks for flash attention"},
    ),
    "hf_qwen3": dict(
        module="transformers.models.qwen3.modeling_qwen3",
        package="transformers", keep=['Qwen3ForCausalLM'],
    ),
    "torchbench_nanogpt": dict(
        module="torchbenchmark.models.nanogpt.model", package="torchbenchmark",
        keep=["GPT", "GPTConfig"],
        prelude={"dataclasses.dataclass": DATACLASS}),
    "torchbench_dcgan": dict(
        module="torchbenchmark.models.dcgan", package="torchbenchmark",
        keep=["DCGAN", "Discriminator"]),
    "torchbench_phlippe_densenet": dict(
        module="torchbenchmark.models.phlippe_densenet",
        package="torchbenchmark", keep=["DenseNet"],
        prelude={"types.SimpleNamespace": SIMPLE_NAMESPACE}),
    "torchbench_bert_pytorch": dict(
        module=['torchbenchmark.models.BERT_pytorch.bert_pytorch.model.attention.single', 'torchbenchmark.models.BERT_pytorch.bert_pytorch.model.attention.multi_head', 'torchbenchmark.models.BERT_pytorch.bert_pytorch.model.embedding.token', 'torchbenchmark.models.BERT_pytorch.bert_pytorch.model.embedding.position', 'torchbenchmark.models.BERT_pytorch.bert_pytorch.model.embedding.segment', 'torchbenchmark.models.BERT_pytorch.bert_pytorch.model.embedding.bert', 'torchbenchmark.models.BERT_pytorch.bert_pytorch.model.utils.feed_forward', 'torchbenchmark.models.BERT_pytorch.bert_pytorch.model.utils.layer_norm', 'torchbenchmark.models.BERT_pytorch.bert_pytorch.model.utils.sublayer', 'torchbenchmark.models.BERT_pytorch.bert_pytorch.model.transformer', 'torchbenchmark.models.BERT_pytorch.bert_pytorch.model.bert', 'torchbenchmark.models.BERT_pytorch.bert_pytorch.model.language_model'],
        package="torchbenchmark", keep=["BERT", "BERTLM"]),
    "torchbench_isoneutral_pytorch": dict(
        module="torchbenchmark.models.pyhpc_isoneutral_mixing."
               "isoneutral_pytorch",
        package="torchbenchmark", keep=["isoneutral_diffusion_pre"]),
    "torchbench_pyhpc_isoneutral_mixing": dict(
        module="torchbenchmark.models.pyhpc_isoneutral_mixing",
        package="torchbenchmark", keep=["IsoneutralMixing"]),
    "torchbench_tke_pytorch": dict(
        module="torchbenchmark.models.pyhpc_turbulent_kinetic_energy."
               "tke_pytorch",
        package="torchbenchmark", keep=["integrate_tke"]),
    "torchbench_pyhpc_turbulent_kinetic_energy": dict(
        module="torchbenchmark.models.pyhpc_turbulent_kinetic_energy",
        package="torchbenchmark", keep=["TurbulentKineticEnergy"]),
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
    "torchvision_squeezenet": dict(
        module="torchvision.models.squeezenet", package="torchvision",
        keep=["Fire", "SqueezeNet"], drop_calls={"_log_api_usage_once": "usage telemetry, no numerics"}),
    "torchvision_mobilenetv2": dict(
        module="torchvision.models.mobilenetv2", package="torchvision",
        keep=["InvertedResidual", "MobileNetV2"], drop_calls={"_log_api_usage_once": "usage telemetry, no numerics"}),
    "torchvision_mobilenetv3": dict(
        module="torchvision.models.mobilenetv3", package="torchvision",
        keep=["InvertedResidualConfig", "InvertedResidual", "MobileNetV3",
              "_mobilenet_v3_conf"], drop_calls={"_log_api_usage_once": "usage telemetry, no numerics"}),
    "torchvision_mnasnet": dict(
        module="torchvision.models.mnasnet", package="torchvision",
        keep=["_BN_MOMENTUM", "_InvertedResidual", "_stack",
              "_round_to_multiple_of", "_get_depths", "MNASNet"],
        drop_calls={"_log_api_usage_once": "usage telemetry, no numerics"}),
    "torchvision_densenet": dict(
        module="torchvision.models.densenet", package="torchvision",
        keep=["_DenseLayer", "_DenseBlock", "_Transition", "DenseNet"],
        drop_calls={"_log_api_usage_once": "usage telemetry, no numerics"}),
    "torchvision_shufflenetv2": dict(
        module="torchvision.models.shufflenetv2", package="torchvision",
        keep=["channel_shuffle", "InvertedResidual", "ShuffleNetV2"],
        drop_calls={"_log_api_usage_once": "usage telemetry, no numerics"}),
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
    "timm.layers.activations": dict(
        module="timm.layers.activations", package="timm",
        keep=["swish", "Swish", "mish", "Mish", "sigmoid", "Sigmoid", "tanh",
              "Tanh", "hard_swish", "HardSwish", "hard_sigmoid",
              "HardSigmoid", "hard_mish", "HardMish", "PReLU", "gelu",
              "GELU", "gelu_tanh", "GELUTanh", "quick_gelu", "QuickGELU"],
        out="timm/layers/activations.py"),
    "timm.layers.fast_norm": dict(
        module="timm.layers.fast_norm", package="timm",
        keep=["is_fast_norm", "fast_group_norm", "fast_layer_norm",
              "fast_rms_norm", "rms_norm", "rms_norm2d", "fast_rms_norm2d",
              "fast_simple_norm", "simple_norm"],
        free_ok=dict((n, "bound by upstream's top-level try-import of apex, "
                      "read on the fast-norm path only, which is off "
                      "(_USE_FAST_NORM = False)")
                     for n in ["has_apex", "has_apex_rmsnorm",
                               "fused_layer_norm_affine", "fused_rms_norm",
                               "fused_rms_norm_affine"]),
        out="timm/layers/fast_norm.py"),
    "timm.layers.norm": dict(
        module="timm.layers.norm", package="timm",
        keep=["GroupNorm", "GroupNorm1", "LayerNorm", "LayerNorm2d",
              "LayerNormFp32", "LayerNorm2dFp32", "RmsNorm", "RmsNorm2d",
              "RmsNormFp32", "RmsNorm2dFp32", "SimpleNorm", "SimpleNorm2d",
              "SimpleNormFp32", "SimpleNorm2dFp32"],
        free_ok={"rms_norm": "bound by a top-level try-import; read by "
                 "RmsNorm.forward, which no ported model builds"},
        out="timm/layers/norm.py"),
    "timm.layers.create_norm": dict(
        module="timm.layers.create_norm", package="timm",
        keep=["get_norm_layer", "create_norm_layer"],
        out="timm/layers/create_norm.py"),
    "timm.layers.attention": dict(
        module="timm.layers.attention", package="timm",
        keep=["maybe_add_mask", "Attention"],
        out="timm/layers/attention.py"),
    "timm.layers.attention_pool": dict(
        module="timm.layers.attention_pool", package="timm",
        keep=["AttentionPoolLatent"],
        out="timm/layers/attention_pool.py"),
    "timm.layers.mlp": dict(
        module="timm.layers.mlp", package="timm",
        keep=["Mlp", "GluMlp", "SwiGLUPacked", "SwiGLU", "GatedMlp",
              "ConvMlp", "GlobalResponseNormMlp"],
        out="timm/layers/mlp.py"),
    "timm.layers.patch_embed": dict(
        module="timm.layers.patch_embed", package="timm",
        keep=["PatchEmbed", "resample_patch_embed"],
        out="timm/layers/patch_embed.py"),
    "timm.layers.pos_embed": dict(
        module="timm.layers.pos_embed", package="timm",
        keep=["resample_abs_pos_embed", "resample_abs_pos_embed_nhwc"],
        out="timm/layers/pos_embed.py"),
    "timm.layers.patch_dropout": dict(
        module="timm.layers.patch_dropout", package="timm",
        keep=["PatchDropout"],
        out="timm/layers/patch_dropout.py"),
    "timm.layers.drop": dict(
        module="timm.layers.drop", package="timm",
        keep=["drop_block_2d", "DropBlock2d", "drop_path", "DropPath",
              "calculate_drop_path_rates"],
        out="timm/layers/drop.py"),
    "timm.layers.grid": dict(
        module="timm.layers.grid", package="timm",
        keep=["ndgrid", "meshgrid"], out="timm/layers/grid.py"),
    "timm.layers.grn": dict(
        module="timm.layers.grn", package="timm",
        keep=["GlobalResponseNorm"], out="timm/layers/grn.py"),
    "timm.layers.layer_scale": dict(
        module="timm.layers.layer_scale", package="timm",
        keep=["LayerScale", "LayerScale2d"],
        out="timm/layers/layer_scale.py"),
    "timm.layers.norm_act": dict(
        module="timm.layers.norm_act", package="timm",
        keep=["BatchNormAct2d", "GroupNormAct", "GroupNorm1Act",
              "LayerNormAct", "LayerNormActFp32", "LayerNormAct2d",
              "LayerNormAct2dFp32", "RmsNormAct", "RmsNormActFp32",
              "RmsNormAct2d", "RmsNormAct2dFp32"],
        free_ok={"rms_norm": "bound by a top-level try-import; read by the "
                 "RmsNormAct forwards, which no ported model builds"},
        out="timm/layers/norm_act.py"),
    "timm.layers.padding": dict(
        module="timm.layers.padding", package="timm",
        keep=["get_padding", "get_same_padding", "is_static_pad",
              "pad_same_arg", "pad_same", "get_padding_value"],
        prelude={"math": MATH3},
        out="timm/layers/padding.py"),
    "timm.layers.conv2d_same": dict(
        module="timm.layers.conv2d_same", package="timm",
        keep=["conv2d_same", "Conv2dSame", "create_conv2d_pad"],
        out="timm/layers/conv2d_same.py"),
    "timm.layers.cond_conv2d": dict(
        module="timm.layers.cond_conv2d", package="timm",
        keep=["CondConv2d", "get_condconv_initializer"],
        out="timm/layers/cond_conv2d.py"),
    "timm.layers.mixed_conv2d": dict(
        module="timm.layers.mixed_conv2d", package="timm",
        keep=["MixedConv2d"], out="timm/layers/mixed_conv2d.py"),
    "timm.layers.create_conv2d": dict(
        module="timm.layers.create_conv2d", package="timm",
        keep=["create_conv2d"], out="timm/layers/create_conv2d.py"),
    "timm.layers.blur_pool": dict(
        module="timm.layers.blur_pool", package="timm",
        keep=["BlurPool2d", "create_aa"],
        prelude={"math.comb": MATH_COMB},
        out="timm/layers/blur_pool.py"),
    "timm.layers.conv_bn_act": dict(
        module="timm.layers.conv_bn_act", package="timm",
        keep=["ConvNormAct", "ConvNormActAa", "ConvBnAct"],
        out="timm/layers/conv_bn_act.py"),
    "timm.layers.adaptive_avgmax_pool": dict(
        module="timm.layers.adaptive_avgmax_pool", package="timm",
        keep=["adaptive_avgmax_pool2d", "select_adaptive_pool2d",
              "AdaptiveAvgMaxPool2d", "SelectAdaptivePool2d"],
        free_ok={"_int_tuple_2_t": "a type alias only annotations read"},
        out="timm/layers/adaptive_avgmax_pool.py"),
    "timm.layers.classifier": dict(
        module="timm.layers.classifier", package="timm",
        keep=["create_classifier", "ClassifierHead", "NormMlpClassifierHead",
              "ClNormMlpClassifierHead"],
        out="timm/layers/classifier.py"),
    "timm.layers.squeeze_excite": dict(
        module="timm.layers.squeeze_excite", package="timm",
        keep=["SEModule", "SqueezeExcite", "EffectiveSEModule",
              "EffectiveSqueezeExcite"],
        out="timm/layers/squeeze_excite.py"),
    "timm.layers.linear": dict(
        module="timm.layers.linear", package="timm",
        keep=["Linear"], out="timm/layers/linear.py"),
    "timm.layers.std_conv": dict(
        module="timm.layers.std_conv", package="timm",
        keep=["StdConv2d", "StdConv2dSame", "ScaledStdConv2d",
              "ScaledStdConv2dSame"],
        out="timm/layers/std_conv.py"),
    "timm.layers.split_attn": dict(
        module="timm.layers.split_attn", package="timm",
        keep=["SplitAttn"], out="timm/layers/split_attn.py"),
    "timm.layers.pool2d_same": dict(
        module="timm.layers.pool2d_same", package="timm",
        keep=["AvgPool2dSame", "MaxPool2dSame", "create_pool2d"],
        out="timm/layers/pool2d_same.py"),
    "timm.layers.separable_conv": dict(
        module="timm.layers.separable_conv", package="timm",
        keep=["SeparableConvNormAct", "SeparableConv2d"],
        out="timm/layers/separable_conv.py"),
    "timm.models._features": dict(
        module="timm.models._features", package="timm",
        keep=["feature_take_indices", "FeatureInfo", "FeatureHooks"],
        free_ok={"OutIndicesT": "a type alias only annotations read"},
        out="timm/models/_features.py"),
    "timm.models.vision_transformer": dict(
        module="timm.models.vision_transformer", package="timm",
        keep=["Block", "VisionTransformer", "checkpoint_filter_fn",
              "_create_vision_transformer", "vit_small_patch16_224",
              "vit_giant_patch14_224", "vit_base_patch16_siglip_256",
              "vit_base_patch14_dinov2"],
        out="timm/models/vision_transformer.py"),
    "timm_vovnet": dict(
        module="timm.models.vovnet", package="timm",
        keep=["SequentialAppendList", "OsaBlock", "OsaStage", "VovNet",
              "model_cfgs", "_create_vovnet", "vovnet39a"]),
    "timm_visformer": dict(
        module="timm.models.visformer", package="timm",
        keep=["Visformer", "_create_visformer", "visformer_small"],
        prelude={"round": ROUND3}),
    "timm_nfnet": dict(
        module="timm.models.nfnet", package="timm",
        keep=["NormFreeNet", "model_cfgs", "_create_normfreenet",
              "dm_nfnet_f0", "nfnet_l0"],
        prelude={"dataclasses.dataclass": DC_DATACLASS,
                 "dataclasses.replace": DC_REPLACE}),
    "timm.layers.interpolate": dict(
        module="timm.layers.interpolate", package="timm",
        keep=["RegularGridInterpolator"],
        out="timm/layers/interpolate.py"),
    "timm.layers.pos_embed_rel": dict(
        module="timm.layers.pos_embed_rel", package="timm",
        keep=["gen_relative_position_index", "resize_rel_pos_bias_table",
              "resize_rel_pos_bias_table_simple",
              "resize_rel_pos_bias_table_levit"],
        out="timm/layers/pos_embed_rel.py"),
    "timm_swin_transformer": dict(
        module="timm.models.swin_transformer", package="timm",
        keep=["SwinTransformer", "_create_swin_transformer",
              "swin_base_patch4_window7_224"],
        free_ok={"_int_or_tuple_2_t": "a type alias only annotations read"},
        prelude={"math": MATH3}),
    "timm_convnext": dict(
        module="timm.models.convnext", package="timm",
        keep=["ConvNeXt", "_create_convnext", "convnextv2_nano"]),
    "timm_regnet": dict(
        module="timm.models.regnet", package="timm",
        keep=["RegNet", "model_cfgs", "_create_regnet", "regnety_120"],
        prelude={"dataclasses.dataclass": DC_DATACLASS,
                 "dataclasses.replace": DC_REPLACE, "round": ROUND3}),
    "timm.models.resnet": dict(
        module="timm.models.resnet", package="timm",
        keep=["BasicBlock", "Bottleneck", "ResNet"],
        out="timm/models/resnet.py"),
    "timm_resnest": dict(
        module="timm.models.resnest", package="timm",
        keep=["ResNestBottleneck", "_create_resnest", "resnest14d"]),
    "timm.layers.pos_embed_sincos": dict(
        module="timm.layers.pos_embed_sincos", package="timm",
        keep=["apply_rot_embed", "RotaryEmbedding"],
        out="timm/layers/pos_embed_sincos.py"),
    "timm.layers.attention_pool2d": dict(
        module="timm.layers.attention_pool2d", package="timm",
        keep=["RotAttentionPool2d", "AttentionPool2d"],
        out="timm/layers/attention_pool2d.py"),
    "timm.models.byobnet": dict(
        module="timm.models.byobnet", package="timm",
        keep=["ByoBlockCfg", "ByoModelCfg", "LayerFn", "num_groups",
              "register_block", "create_block", "ByobNet", "model_cfgs",
              "_create_byobnet", "repvgg_a2"],
        prelude={"dataclasses.dataclass": DC_DATACLASS,
                 "dataclasses.field": DC_FIELD,
                 "dataclasses.replace": DC_REPLACE, "round": ROUND3,
                 "typing.Sequence": TYPING_ABCS,
                 "typing.Callable": ""},
        out="timm/models/byobnet.py"),
    "timm_mobilevit": dict(
        module="timm.models.mobilevit", package="timm",
        keep=["MobileVitBlock", "MobileVitV2Block", "model_cfgs",
              "_create_mobilevit", "mobilevit_s"],
        keep_calls={"register_block": "adds the MobileViT blocks to "
                    "byobnet's block registry, which ByobNet builds from"},
        prelude={"math": MATH3}),
    "timm_inception_v3": dict(
        module="timm.models.inception_v3", package="timm",
        keep=["InceptionA", "InceptionB", "InceptionC", "InceptionD",
              "InceptionE", "InceptionAux", "InceptionV3",
              "_create_inception_v3", "inception_v3"]),
    "timm.models._efficientnet_blocks": dict(
        module="timm.models._efficientnet_blocks", package="timm",
        keep=["num_groups", "SqueezeExcite", "ConvBnAct",
              "DepthwiseSeparableConv", "InvertedResidual",
              "UniversalInvertedResidual", "MobileAttention",
              "CondConvResidual", "EdgeResidual"],
        free_ok={"ModuleType": "a type alias only annotations read"},
        prelude={"round": ROUND3},
        out="timm/models/_efficientnet_blocks.py"),
    "timm.layers.attention2d": dict(
        module="timm.layers.attention2d", package="timm",
        keep=["MultiQueryAttentionV2", "MultiQueryAttention2d",
              "Attention2d"],
        out="timm/layers/attention2d.py"),
    "timm.utils.model": dict(
        module="timm.utils.model", package="timm",
        keep=["reparameterize_model"], out="timm/utils/model.py"),
    "timm.models._efficientnet_builder": dict(
        module="timm.models._efficientnet_builder", package="timm",
        keep=["BN_MOMENTUM_TF_DEFAULT", "BN_EPS_TF_DEFAULT", "get_bn_args_tf",
              "resolve_bn_args", "resolve_act_layer", "round_channels",
              "decode_arch_def", "EfficientNetBuilder",
              "efficientnet_init_weights"],
        prelude={"round": ROUND3},
        out="timm/models/_efficientnet_builder.py"),
    "timm_efficientnet": dict(
        module="timm.models.efficientnet", package="timm",
        keep=["EfficientNet", "_create_effnet", "_gen_mobilenet_v2",
              "_gen_efficientnet", "mobilenetv2_100", "efficientnet_b0",
              "tf_efficientnet_b0"]),
    "timm_mobilenetv3": dict(
        module="timm.models.mobilenetv3", package="timm",
        keep=["MobileNetV3", "_create_mnv3", "_gen_mobilenet_v3",
              "mobilenetv3_large_100"]),
    "timm_ghostnet": dict(
        module="timm.models.ghostnet", package="timm",
        keep=["GhostModule", "GhostBottleneck", "GhostNet",
              "_create_ghostnet", "ghostnet_100"],
        prelude={"math": MATH3}),
    "timm_beit": dict(
        module="timm.models.beit", package="timm",
        keep=["gen_relative_position_index", "Attention", "Block",
              "RelativePositionBias", "Beit", "checkpoint_filter_fn",
              "_create_beit", "beit_base_patch16_224"]),
    "timm_deit": dict(
        module="timm.models.deit", package="timm",
        keep=["VisionTransformerDistilled", "_create_deit",
              "deit_tiny_patch16_224", "deit_base_distilled_patch16_224"]),
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
        dc = any(getattr(d.func if isinstance(d, ast.Call) else d, "id",
                         None) == "dataclass" for d in node.decorator_list)
        names = [n.target.id for n in node.body
                 if isinstance(n, ast.AnnAssign) and
                 isinstance(n.target, ast.Name)]
        self.generic_visit(node)
        if dc:
            doc = 1 if node.body and isinstance(node.body[0], ast.Expr) \
                and isinstance(node.body[0].value, ast.Constant) else 0
            node.body.insert(doc, ast.Assign(
                [ast.Name("__annotations__", ast.Store())],
                _tuple([ast.Constant(n) for n in names])))
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

    def visit_If(self, node):
        hoisted = []

        def take(n):
            if isinstance(n, ast.NamedExpr):
                hoisted.append(ast.Assign(targets=[ast.Name(n.target.id,
                                                            ast.Store())],
                                          value=n.value))
                return ast.Name(n.target.id, ast.Load())
            if isinstance(n, (ast.BoolOp, ast.IfExp, ast.Lambda,
                              ast.ListComp, ast.SetComp, ast.DictComp,
                              ast.GeneratorExp)):
                return n
            for field, value in ast.iter_fields(n):
                if isinstance(value, list):
                    setattr(n, field, [take(v) if isinstance(v, ast.AST)
                                       else v for v in value])
                elif isinstance(value, ast.AST):
                    setattr(n, field, take(value))
            return n
        node.test = take(node.test)
        self.generic_visit(node)
        return [ast.copy_location(h, node) for h in hoisted] + [node] \
            if hoisted else node

    def visit_BinOp(self, node):
        self.generic_visit(node)
        if isinstance(node.op, ast.MatMult):
            return ast.copy_location(ast.Call(
                ast.Attribute(node.left, "__matmul__", ast.Load()),
                [node.right], []), node)
        return node

    def visit_AugAssign(self, node):
        self.generic_visit(node)
        if isinstance(node.op, ast.MatMult):
            raise NotImplementedError("@= at line %d" % node.lineno)
        return node

    def visit_Assign(self, node):
        self.generic_visit(node)
        t = node.targets[0]
        stars = [i for i, e in enumerate(getattr(t, "elts", []))
                 if isinstance(e, ast.Starred)]
        if len(node.targets) != 1 or not stars:
            return node
        i, n = stars[0], len(t.elts)
        tv = lambda: ast.Name("_t", ast.Load())
        end = ast.BinOp(ast.Call(ast.Name("len", ast.Load()), [tv()], []),
                        ast.Sub(), ast.Constant(n - 1 - i))
        rest = ast.Call(ast.Name("list", ast.Load()), [ast.Subscript(
            tv(), ast.Slice(ast.Constant(i), end), ast.Load())], [])
        body = ast.BinOp(ast.BinOp(
            ast.Subscript(tv(), ast.Slice(None, ast.Constant(i)), ast.Load()),
            ast.Add(), _tuple([rest])), ast.Add(),
            ast.Subscript(tv(), ast.Slice(end, None), ast.Load()))
        fn = ast.Lambda(ast.arguments([], [ast.arg("_t")], None, [], [],
                                      None, []), body)
        node.targets = [type(t)([e.value if isinstance(e, ast.Starred) else e
                                 for e in t.elts], ast.Store())]
        node.value = ast.Call(fn, [ast.Call(ast.Name("tuple", ast.Load()),
                                            [node.value], [])], [])
        return node

    def visit_Raise(self, node):
        self.generic_visit(node)
        node.cause = None
        return node

    def visit_Dict(self, node):
        self.generic_visit(node)
        if all(k is not None for k in node.keys):
            return node
        parts = [ast.Call(ast.Name("list", ast.Load()), [ast.Call(
            ast.Attribute(v, "items", ast.Load()), [], [])], [])
            if k is None else _list([_tuple([k, v])])
            for k, v in zip(node.keys, node.values)]
        merged = parts[0]
        for p in parts[1:]:
            merged = ast.BinOp(merged, ast.Add(), p)
        return ast.copy_location(ast.Call(ast.Name("dict", ast.Load()),
                                          [merged], []), node)

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
        kws = node.keywords
        first = next((i for i, k in enumerate(kws) if k.arg is None), None)
        if first is not None and len(kws) - first > 1:
            parts = [ast.Call(ast.Name("list", ast.Load()), [ast.Call(
                ast.Attribute(k.value, "items", ast.Load()), [], [])], [])
                if k.arg is None else
                _list([_tuple([ast.Constant(k.arg), k.value])])
                for k in kws[first:]]
            merged = parts[0]
            for p in parts[1:]:
                merged = ast.BinOp(merged, ast.Add(), p)
            node.keywords = kws[:first] + [ast.keyword(None, ast.Call(
                ast.Name("dict", ast.Load()), [merged], []))]
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


PY2_BUILTINS = set(dir(__builtins__)) | {
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


def closure(tree, roots, never):
    """roots and every top-level definition they reference, transitively,
    except names listed as never read."""
    defs = {}
    for n in tree.body:
        for name in defined(n):
            defs.setdefault(name, []).append(n)
    keep, todo = set(), list(roots)
    while todo:
        name = todo.pop()
        if name in keep:
            continue
        keep.add(name)
        for n in defs.get(name, []):
            for ref in used_names(n):
                if ref in defs and ref not in keep and ref not in never:
                    todo.append(ref)
    return keep


def used_names(tree):
    out = set()
    for n in ast.walk(tree):
        if isinstance(n, ast.Name):
            out.add(n.id)
    return out


# standard-library modules that moved between Python 2 and 3; an import of
# the Python 3 name imports the Python 2 one
PY2_MODULES = {"collections.abc": "collections"}


TRUE_GUARDS = {"is_torch_available"}


def top_imports(tree):
    for n in tree.body:
        if isinstance(n, ast.If) and isinstance(n.test, ast.Call) and \
                isinstance(n.test.func, ast.Name) and \
                n.test.func.id in TRUE_GUARDS:
            for m in n.body:
                yield m
        else:
            yield n


def imports(tree, module, used, supplied, is_package, ported):
    """Upstream's top-level imports, absolute, restricted to used names; an
    import of a module that has a port of its own imports the port."""
    pkg = module.split(".") + ([""] if is_package else [])
    out = []
    for n in top_imports(tree):
        if isinstance(n, ast.Import):
            names = [a for a in n.names
                     if (a.asname or a.name.split(".")[0]) in used
                     and (a.asname or a.name) not in supplied]
            if names:
                out.append(ast.Import(names))
        elif isinstance(n, ast.ImportFrom):
            if n.module == "__future__":
                continue
            base = PY2_MODULES.get(n.module, n.module or "")
            if n.level:
                base = ".".join(pkg[:len(pkg) - n.level] +
                                ([n.module] if n.module else []))
            aliases = n.names
            if aliases[0].name == "*":
                aliases = [ast.alias(x) for x in
                           importlib.import_module(base).__all__]
            names = [a for a in aliases if (a.asname or a.name) in used
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


def parse_group(modules):
    """One tree from several upstream modules, in order: a model spread over
    a package becomes one port, with imports between its members dropped
    (the names they import are defined in the same file)."""
    trees, srcs, paths = [], [], []
    for m in modules:
        mod = importlib.import_module(m)
        paths.append(mod.__file__)
        srcs.append(open(mod.__file__).read())
        trees.append(ast.parse(srcs[-1]))
    tree = ast.Module(body=[n for t in trees for n in t.body],
                      type_ignores=[])
    return tree, "".join(srcs), paths


def port(name):
    spec = SPECS[name]
    group = spec["module"] if isinstance(spec["module"], list) \
        else [spec["module"]]
    tree, src, paths = parse_group(group)
    path = paths[0]
    keep = closure(tree, set(spec["keep"]),
                   set(spec.get("free_ok", {})))
    calls = spec.get("keep_calls", {})
    body = [n for n in tree.body if defined(n) & keep or
            isinstance(n, ast.Expr) and isinstance(n.value, ast.Call) and
            getattr(n.value.func, "id", None) in calls]
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
                   if "out" not in v and not isinstance(v["module"], list))
    local = set().union(*[defined(n) for n in kept.body]) if kept.body \
        else set()
    imps = []
    for m, p_ in zip(group, paths):
        imps += imports(ast.parse(open(p_).read()), m, used_names(kept),
                        supplied | local, p_.endswith("__init__.py"),
                        ported)
    seen, dedup = set(), []
    for n in imps:
        key = ast.dump(n)
        if key not in seen:
            seen.add(key)
            dedup.append(n)
    imps = dedup
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
    exceptions += "".join("#   kept top-level calls to %s: %s\n" % kv
                          for kv in sorted(calls.items()))
    exceptions += "".join("#   %s supplied for Python 2\n" % k
                          for k in sorted(prelude))
    exceptions += "".join("#   %s left unbound: %s\n" % kv
                          for kv in sorted(spec.get("free_ok", {}).items()))
    header = (
        "# -*- coding: utf-8 -*-\n"
        "# Generated by benchmark/suites/port.py from %s\n"
        "# (%s %s, source sha256 %s).  Do not edit; change the spec.\n"
        "# Kept: %s\n%s"
        % (", ".join(group), spec["package"], version(spec["package"]),
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


OUTPUT_BASE = '''"""The ModelOutput classes of transformers.modeling_outputs.  Upstream they
are dataclasses over an OrderedDict; here each lists its fields in the
order upstream declares them (generated below from the dataclass fields),
and indexing, to_tuple() and iteration skip the fields left None, as
upstream's do."""


class ModelOutput(object):
    _fields = ()

    def __init__(self, *args, **kwargs):
        for k in kwargs:
            if k not in self._fields:
                raise TypeError("%s has no field %s"
                                % (type(self).__name__, k))
        for i, f in enumerate(self._fields):
            setattr(self, f, args[i] if i < len(args) else kwargs.get(f))

    def keys(self):
        return [f for f in self._fields if getattr(self, f) is not None]

    def to_tuple(self):
        return tuple(getattr(self, f) for f in self.keys())

    def __getitem__(self, k):
        if isinstance(k, str):
            return getattr(self, k)
        return self.to_tuple()[k]

    def __setitem__(self, k, v):
        setattr(self, k, v)

    def __iter__(self):
        return iter(self.keys())

    def __len__(self):
        return len(self.keys())


def _output(name, fields):
    return type(name, (ModelOutput,), {"_fields": tuple(fields)})
'''


def model_outputs():
    """compat/transformers/modeling_outputs.py from upstream's dataclasses."""
    import dataclasses
    import inspect as _inspect
    mod = importlib.import_module("transformers.modeling_outputs")
    base = importlib.import_module("transformers.utils.generic").ModelOutput
    src = open(mod.__file__).read()
    lines = []
    for name, cls in sorted(vars(mod).items()):
        if (_inspect.isclass(cls) and issubclass(cls, base) and
                cls is not base and cls.__module__ == mod.__name__):
            fields = [f.name for f in dataclasses.fields(cls)]
            lines.append("%s = _output(%r, %r)" % (name, name, fields))
    header = ("# -*- coding: utf-8 -*-\n"
              "# Generated by benchmark/suites/port.py from the dataclass "
              "fields of\n# transformers.modeling_outputs (transformers %s, "
              "source sha256 %s).\n"
              % (version("transformers"),
                 hashlib.sha256(src.encode()).hexdigest()[:16]))
    return header + OUTPUT_BASE + "\n\n" + "\n".join(lines) + "\n"


GENERATED = {"transformers/modeling_outputs.py": model_outputs}


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


def write_generated():
    for rel, fn in GENERATED.items():
        dst = os.path.join(COMPAT, rel)
        open(dst, "w").write(fn())
        print("wrote %s" % dst)


def main(argv):
    if argv[1:2] == ["--all"]:
        for n in SPECS:
            write(n)
        write_generated()
    elif argv[1:2] == ["--generated"]:
        write_generated()
    elif argv[1:2] == ["--check"]:
        bad = 0
        for n in SPECS:
            dst = destination(n)
            if not os.path.exists(dst) or open(dst).read() != port(n):
                print("stale: %s" % dst)
                bad += 1
        for rel, fn in GENERATED.items():
            dst = os.path.join(COMPAT, rel)
            if not os.path.exists(dst) or open(dst).read() != fn():
                print("stale: %s" % dst)
                bad += 1
        return 1 if bad else 0
    else:
        write(argv[1])
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
