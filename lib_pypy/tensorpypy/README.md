# tensorpypy

Python layer on top of the `_metatensor` built-in module (see
`pypy/module/_metatensor/`). Replaces the old single-file `tensorlite.py`.

Two sub-APIs:

- Array API: `__init__.py` (`asarray`, `zeros`, `ones_like`, `matmul`, `sum`,
  `max`, `exp`, `sqrt`, `reshape`, `take`, `astype`, `Tensor`, dtype
  constants) and `functional.py` (`relu`, `gelu`, `gelu_erf`, `silu`,
  `softmax`, `layer_norm`, `rms_norm`).  `gelu_erf` is the exact (erf) GELU
  built out of `relu`/`mul`/`add`/`div`/`exp` with the Abramowitz-Stegun
  7.1.26 rational approximation of `erf` (max error 1.4e-7); it needs no new
  kernel and fuses into the surrounding elementwise chain.
- NN API: `nn.py` (`Module`, `Linear`, `LayerNorm`, `RMSNorm`, `Embedding`,
  `MultiheadAttention`, `Conv2d`, `BatchNorm2d`, `MaxPool2d`), `optim.py`
  (`sgd_step`, `SGD`), and `models.py` (`MLP`, `TransformerBlock`, `CNN`,
  `GPT2`, `Llama`, `Bert`, `ViT`, `ResNet`, `Mixer`).

`Conv2d` and `MaxPool2d` take `k`, `stride` and `pad` (defaults 3/1/1 and
2/2/0), which is what ResNet's 7x7 stride-2 stem, its stride-2 3x3 and 1x1
downsample convolutions and its 3x3 stride-2 pad-1 max pool need.
`CausalSelfAttention` takes `mask=None` for the bidirectional (BERT, ViT)
case, and `GPT2MLP` takes the activation, so the same attention and MLP code
serves GPT-2, BERT and ViT.

`nn.py`'s `Param` is a live (owner, attribute-name) reference rather than the
tensor itself, because `sgd_step` replaces a parameter with a new tensor
instead of mutating it; a list of tensors captured once would go stale after
the first update, so `parameters()` hands out rebindable slots.
