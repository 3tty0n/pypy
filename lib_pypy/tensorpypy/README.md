# tensorpypy

Python layer on top of the `_metatensor` built-in module (see
`pypy/module/_metatensor/`). Replaces the old single-file `tensorlite.py`.

Two sub-APIs:

- Array API: `__init__.py` (`asarray`, `zeros`, `ones_like`, `matmul`, `sum`,
  `max`, `exp`, `sqrt`, `reshape`, `take`, `astype`, `Tensor`, dtype
  constants) and `functional.py` (`relu`, `gelu`, `silu`, `softmax`,
  `layer_norm`, `rms_norm`).
- NN API: `nn.py` (`Module`, `Linear`, `LayerNorm`, `RMSNorm`, `Embedding`,
  `MultiheadAttention`, `Conv2d`, `BatchNorm2d`, `MaxPool2d`), `optim.py`
  (`sgd_step`, `SGD`), and `models.py` (`MLP`, `TransformerBlock`, `CNN`,
  `GPT2`, `Llama`).
