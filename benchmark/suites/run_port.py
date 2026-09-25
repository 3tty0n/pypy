"""Run one ported suite model under pypy-c and time it the dashboard's way.

    pypy-c run_port.py EXPORT_DIR [--repeat 30] [--warmup 10] [--dump DIR]

EXPORT_DIR is what suite_export.py wrote.  The model is built by the ported
upstream source (ports/), its state_dict and inputs come from the export.
--dump writes the last forward's outputs (flattened in pytree order) for
suite_check.py, which judges them with the dashboard's own same().

Timing follows benchmarks/dynamo/common.py: `repeat` measurements, each one
forward run to completion on the device, and the median of them.  The
dashboard ends a measurement with torch.cuda.synchronize(); here the
forward's output is reduced and read on the host (out.sum().item()), which
waits for the device the same way and adds one small kernel.  Warm-up
iterations before the first measurement let the JIT compile; the first
call's time is reported on its own.
"""
import array
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.join(HERE, "..", "..")
sys.path.insert(0, os.path.join(REPO, "lib_pypy", "tensorpypy", "compat"))
sys.path.insert(0, os.path.join(HERE, "ports"))

import _metatensor  # noqa: E402
import torch  # noqa: E402


def resnet(block, layers, **kw):
    def build():
        import torchvision_resnet as m
        return m.ResNet(getattr(m, block), layers, **kw)
    return build


def tv(module, cls, *args, **kw):
    def build():
        import importlib
        m = importlib.import_module(module)
        return getattr(m, cls)(*args, **kw)
    return build


def mobilenet_v3(arch):
    def build():
        import torchvision_mobilenetv3 as m
        setting, last = m._mobilenet_v3_conf(arch)
        return m.MobileNetV3(setting, last)
    return build


def phlippe_resnet():
    import torchbench_phlippe_resnet as m
    return m.ResNetModel()


def hf(module, cls):
    def build(config):
        import importlib
        from transformers.configuration_utils import PreTrainedConfig
        m = importlib.import_module(module)
        return getattr(m, cls)(PreTrainedConfig(**config))
    build.wants_config = True
    return build


def lennard_jones():
    # TorchBench's Model.__init__ builds this Sequential inline
    nn = torch.nn
    return nn.Sequential(nn.Linear(1, 16), nn.Tanh(), nn.Linear(16, 16),
                         nn.Tanh(), nn.Linear(16, 16), nn.Tanh(),
                         nn.Linear(16, 16), nn.Tanh(), nn.Linear(16, 1))


def pyhpc_eos():
    import torchbench_pyhpc_equation_of_state as m
    return m.EquationOfState()


def nanogpt():
    import torchbench_nanogpt as m
    return m.GPT(m.GPTConfig())


def dcgan():
    import torchbench_dcgan as m

    class _Bench(object):
        device = "cuda"
    return m.Discriminator(m.DCGAN(_Bench()))


def bert_pytorch(idx):
    import torchbench_bert_pytorch as m
    a = idx["attrs"]
    vocab = idx["params"]["bert.embedding.token.weight"][1][0]
    bert = m.BERT(vocab, hidden=a["bert.hidden"], n_layers=a["bert.n_layers"],
                  attn_heads=a["bert.attn_heads"])
    return m.BERTLM(bert, vocab)


bert_pytorch.wants_index = True


def alexnet():
    import torchvision_alexnet as m
    return m.AlexNet()


def vgg(cfg):
    def build():
        import torchvision_vgg as m
        return m.VGG(m.make_layers(m.cfgs[cfg], batch_norm=False))
    return build


def timm_model(module, name, pretrained=True):
    """timm.create_model as the TIMM runner's _download_model and
    TorchBench's TimmModel call it, after importing the ported file that
    registers the entrypoint (upstream's timm.models imports them all).
    TorchBench builds vovnet39a and vit_giant_patch14_224 without
    pretrained weights; either way the tensors come from the export."""
    def build():
        import importlib
        import timm
        importlib.import_module(module)
        kw = dict(in_chans=3, scriptable=False, num_classes=None,
                  pretrained=pretrained)
        try:
            return timm.create_model(name, drop_rate=0.0, drop_path_rate=None,
                                     drop_block_rate=None, **kw)
        except TypeError as e:
            if "unexpected keyword argument" not in str(e):
                raise
            return timm.create_model(name, **kw)
    return build


# (suite, model) -> how the suite's model constructor builds it upstream
# (torchvision.models.resnet18() is _resnet(BasicBlock, [2, 2, 2, 2]),
# vgg16() is _vgg("D", False, ...), ...)
MODELS = {
    ("torchbench", "hf_DistilBert"): hf("hf_distilbert",
                                        "DistilBertForMaskedLM"),
    ("torchbench", "hf_Bert"): hf("hf_bert", "BertForMaskedLM"),
    ("torchbench", "hf_Bert_large"): hf("hf_bert", "BertForMaskedLM"),
    ("torchbench", "lennard_jones"): lennard_jones,
    ("torchbench", "pyhpc_equation_of_state"): pyhpc_eos,
    ("torchbench", "pyhpc_isoneutral_mixing"): tv(
        "torchbench_pyhpc_isoneutral_mixing", "IsoneutralMixing"),
    ("torchbench", "pyhpc_turbulent_kinetic_energy"): tv(
        "torchbench_pyhpc_turbulent_kinetic_energy",
        "TurbulentKineticEnergy"),
    ("torchbench", "hf_Roberta_base"): hf("hf_xlm_roberta",
                                          "XLMRobertaForMaskedLM"),
    ("torchbench", "hf_Albert"): hf("hf_albert", "AlbertForMaskedLM"),
    ("torchbench", "hf_GPT2"): hf("hf_gpt2", "GPT2LMHeadModel"),
    ("torchbench", "hf_GPT2_large"): hf("hf_gpt2", "GPT2LMHeadModel"),
    ("huggingface", "BertForMaskedLM"): hf("hf_bert", "BertForMaskedLM"),
    ("huggingface", "DistilBertForMaskedLM"): hf("hf_distilbert",
                                                "DistilBertForMaskedLM"),
    ("huggingface", "AlbertForMaskedLM"): hf("hf_albert",
                                            "AlbertForMaskedLM"),
    ("huggingface", "DistillGPT2"): hf("hf_gpt2", "GPT2LMHeadModel"),
    ("huggingface", 'ElectraForCausalLM'): hf('hf_electra', 'ElectraForCausalLM'),
    ("huggingface", 'RobertaForCausalLM'): hf('hf_roberta', 'RobertaForCausalLM'),
    ("huggingface", 'MegatronBertForCausalLM'): hf('hf_megatron_bert', 'MegatronBertForCausalLM'),
    ("huggingface", 'LayoutLMForMaskedLM'): hf('hf_layoutlm', 'LayoutLMForMaskedLM'),
    ("huggingface", 'BartForCausalLM'): hf('hf_bart', 'BartForCausalLM'),
    ("huggingface", 'MBartForCausalLM'): hf('hf_mbart', 'MBartForCausalLM'),
    ("huggingface", 'PLBartForCausalLM'): hf('hf_plbart', 'PLBartForCausalLM'),
    ("huggingface", 'BlenderbotForCausalLM'): hf('hf_blenderbot', 'BlenderbotForCausalLM'),
    ("huggingface", 'TrOCRForCausalLM'): hf('hf_trocr', 'TrOCRForCausalLM'),
    ("huggingface", 'OPTForCausalLM'): hf('hf_opt', 'OPTForCausalLM'),
    ("huggingface", 'XGLMForCausalLM'): hf('hf_xglm', 'XGLMForCausalLM'),
    ("huggingface", 'GPTNeoForCausalLM'): hf('hf_gpt_neo', 'GPTNeoForCausalLM'),
    ("huggingface", 'GPTNeoForSequenceClassification'): hf('hf_gpt_neo', 'GPTNeoForSequenceClassification'),
    ("huggingface", 'GPT2ForSequenceClassification'): hf('hf_gpt2', 'GPT2ForSequenceClassification'),
    ("huggingface", 'Qwen/Qwen3-0.6B'): hf('hf_qwen3', 'Qwen3ForCausalLM'),
    ("torchbench", "nanogpt"): nanogpt,
    ("torchbench", "dcgan"): dcgan,
    ("torchbench", "phlippe_densenet"): tv("torchbench_phlippe_densenet",
                                           "DenseNet"),
    ("torchbench", "BERT_pytorch"): bert_pytorch,
    ("torchbench", "alexnet"): alexnet,
    ("torchbench", "phlippe_resnet"): phlippe_resnet,
    ("torchbench", "vgg16"): vgg("D"),
    ("torchbench", "resnet18"): resnet("BasicBlock", [2, 2, 2, 2]),
    ("torchbench", "resnet50"): resnet("Bottleneck", [3, 4, 6, 3]),
    ("torchbench", "resnet152"): resnet("Bottleneck", [3, 8, 36, 3]),
    ("torchbench", "resnext50_32x4d"): resnet("Bottleneck", [3, 4, 6, 3],
                                              groups=32, width_per_group=4),
    ("torchbench", "squeezenet1_1"): tv("torchvision_squeezenet",
                                        "SqueezeNet", "1_1"),
    ("torchbench", "mobilenet_v2"): tv("torchvision_mobilenetv2",
                                       "MobileNetV2"),
    ("torchbench", "mobilenet_v3_large"): mobilenet_v3("mobilenet_v3_large"),
    ("torchbench", "mnasnet1_0"): tv("torchvision_mnasnet", "MNASNet", 1.0),
    ("torchbench", "densenet121"): tv("torchvision_densenet", "DenseNet", 32,
                                      (6, 12, 24, 16), 64),
    ("torchbench", "shufflenet_v2_x1_0"): tv(
        "torchvision_shufflenetv2", "ShuffleNetV2", [4, 8, 4],
        [24, 116, 232, 464, 1024]),
}

MODELS.update({
    ("timm", "deit_tiny_patch16_224.fb_in1k"): timm_model(
        "timm_deit", "deit_tiny_patch16_224.fb_in1k"),
    ("timm", "deit_base_distilled_patch16_224"): timm_model(
        "timm_deit", "deit_base_distilled_patch16_224"),
    ("timm", "vit_base_patch16_siglip_256"): timm_model(
        "timm.models.vision_transformer", "vit_base_patch16_siglip_256"),
    ("timm", "vit_base_patch14_dinov2.lvd142m"): timm_model(
        "timm.models.vision_transformer", "vit_base_patch14_dinov2.lvd142m"),
    ("timm", "beit_base_patch16_224"): timm_model(
        "timm_beit", "beit_base_patch16_224"),
    ("timm", "repvgg_a2"): timm_model("timm.models.byobnet", "repvgg_a2"),
    ("timm", "mobilenetv2_100"): timm_model(
        "timm_efficientnet", "mobilenetv2_100"),
    ("timm", "mobilenetv3_large_100"): timm_model(
        "timm_mobilenetv3", "mobilenetv3_large_100"),
    ("timm", "tf_efficientnet_b0"): timm_model(
        "timm_efficientnet", "tf_efficientnet_b0"),
    ("timm", "ghostnet_100"): timm_model("timm_ghostnet", "ghostnet_100"),
    ("timm", "inception_v3"): timm_model("timm_inception_v3",
                                         "inception_v3"),
    ("timm", "adv_inception_v3"): timm_model("timm_inception_v3",
                                             "adv_inception_v3"),
    ("timm", "convnextv2_nano.fcmae_ft_in22k_in1k"): timm_model(
        "timm_convnext", "convnextv2_nano.fcmae_ft_in22k_in1k"),
    ("timm", "visformer_small"): timm_model("timm_visformer",
                                            "visformer_small"),
    ("timm", "dm_nfnet_f0"): timm_model("timm_nfnet", "dm_nfnet_f0"),
    ("timm", "nfnet_l0"): timm_model("timm_nfnet", "nfnet_l0"),
    ("timm", "mobilevit_s"): timm_model("timm_mobilevit", "mobilevit_s"),
    ("timm", "swin_base_patch4_window7_224"): timm_model(
        "timm_swin_transformer", "swin_base_patch4_window7_224"),
    ("torchbench", "timm_vision_transformer"): timm_model(
        "timm.models.vision_transformer", "vit_small_patch16_224"),
    ("torchbench", "timm_vision_transformer_large"): timm_model(
        "timm.models.vision_transformer", "vit_giant_patch14_224",
        pretrained=False),
    ("torchbench", "timm_regnet"): timm_model("timm_regnet", "regnety_120"),
    ("torchbench", "timm_resnest"): timm_model("timm_resnest", "resnest14d"),
    ("torchbench", "timm_vovnet"): timm_model("timm_vovnet", "vovnet39a",
                                              pretrained=False),
    ("torchbench", "timm_efficientnet"): timm_model("timm_efficientnet",
                                                    "efficientnet_b0"),
    ("torchbench", "timm_nfnet"): timm_model("timm_nfnet", "dm_nfnet_f0"),
})

TYPECODE = {"float32": "f", "float64": "d", "int64": "l", "int32": "i"}


def read(blob, entry):
    off, shape, dtype = entry
    code = TYPECODE[dtype]
    a = array.array(code)
    n = 1
    for d in shape:
        n *= d
    assert a.itemsize == {"f": 4, "d": 8, "l": 8, "i": 4}[code]
    blob.seek(off)
    a.fromstring(blob.read(n * a.itemsize))
    return a, shape, dtype


def tensor(blob, entry):
    a, shape, dt = read(blob, entry)
    if dt in ("int64", "int32"):
        return torch.tensor([int(x) for x in a], dtype=torch.int64).view(
            *shape) if shape else torch.tensor(int(a[0]))
    return torch.from_flat([float(x) for x in a], shape, dt)


def outputs_of(out):
    if isinstance(out, torch.Tensor):
        return [out]
    if hasattr(out, "to_tuple"):
        out = out.to_tuple()
    return [o for o in out if isinstance(o, torch.Tensor)]


def dump(outputs, path):
    if not os.path.isdir(path):
        os.makedirs(path)
    blob = open(os.path.join(path, "data.bin"), "wb")
    index = []
    for t in outputs:
        a = array.array("f", t.tolist())
        index.append([blob.tell(), list(t.shape), "float32"])
        a.tofile(blob)
    blob.close()
    json.dump({"outputs": index}, open(os.path.join(path, "index.json"), "w"))


def median(xs):
    s = sorted(xs)
    m = len(s)
    return s[m // 2] if m % 2 else 0.5 * (s[m // 2 - 1] + s[m // 2])


def main(argv):
    d = argv[1]
    repeat = int(argv[argv.index("--repeat") + 1]) if "--repeat" in argv \
        else 30
    warmup = int(argv[argv.index("--warmup") + 1]) if "--warmup" in argv \
        else 10
    dump_dir = argv[argv.index("--dump") + 1] if "--dump" in argv else None
    idx = json.load(open(os.path.join(d, "index.json")))
    key = (idx["suite"], idx["model"])
    if key not in MODELS:
        sys.stderr.write("run_port: no port for %s/%s\n" % key)
        return 2
    # benchmarks/dynamo/common.py sets this for every run
    torch.backends.cuda.matmul.allow_tf32 = True
    blob = open(os.path.join(d, "data.bin"), "rb")
    build = MODELS[key]
    if getattr(build, "wants_index", False):
        model = build(idx)
    elif getattr(build, "wants_config", False):
        model = build(idx["config"])
    else:
        model = build()
    sd = {}
    for name, entry in idx["params"].items():
        if entry[2] in ("float32", "float64", "float16"):
            sd[name] = tensor(blob, entry)
    model.load_state_dict(sd)
    model.eval()
    args = [tensor(blob, e) for e in idx["inputs"]]
    kwargs = dict((k, tensor(blob, e)) for k, e in idx["kwargs"].items())

    def step():
        out = model(*args, **kwargs)
        outputs_of(out)[0].sum().item()
        return out

    t0 = time.time()
    out = step()
    first_ms = (time.time() - t0) * 1e3
    for i in range(warmup):
        out = step()
    launches0 = _metatensor.launch_count()
    cpu0 = _metatensor.cpu_fallbacks() if hasattr(_metatensor,
                                                  "cpu_fallbacks") else 0
    host0 = torch.host_writes()
    times = []
    for i in range(repeat):
        t0 = time.time()
        out = step()
        times.append((time.time() - t0) * 1e3)
    launches = (_metatensor.launch_count() - launches0) / float(repeat)
    if dump_dir:
        dump([o for o in outputs_of(out) if o.dtype.startswith("float")],
             dump_dir)
    # an allocation that failed, or any op computed by a host loop, means
    # what was timed is not the GPU run this row claims to be
    cpu = _metatensor.alloc_failed() or (
        hasattr(_metatensor, "cpu_fallbacks") and
        _metatensor.cpu_fallbacks() > cpu0) or torch.host_writes() > host0
    print("port suite=%s model=%s batch=%s median_ms=%.3f min_ms=%.3f "
          "first_ms=%.1f launches=%.1f cpu_fallback=%d"
          % (idx["suite"], idx["model"], idx["batch"], median(times),
             min(times), first_ms, launches, int(bool(cpu))))
    return 3 if cpu else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
