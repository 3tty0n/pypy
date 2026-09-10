import torch

import torch_common


def main():
    a = torch_common.argv()
    cfg = a.cfg
    px = torch_common.image(a).to(a.dev, a.dtype)
    from transformers import ViTForImageClassification
    hf = ViTForImageClassification.from_pretrained(cfg['source'], **torch_common.hf_kwargs(a))
    hf = hf.to(a.dev, a.dtype).eval()
    fwd = lambda x: hf(x).logits[0]
    fwd = torch_common.compiled(fwd, a)
    logits, acc, steady_us = torch_common.timed(fwd, (px,), a)
    torch_common.report(
        'vit torch-%s layers=%d embd=%d heads=%d tokens=%d classes=%d '
        'dtype=%s iters=%d steady_us=%.1f checksum=%.6f' %
        (a.mode, cfg['n_layer'], cfg['n_embd'], cfg['n_head'], cfg['tokens'],
         cfg['classes'], a.dtname, a.iters, steady_us, acc),
        logits.topk(5).indices.tolist(), torch_common.compare(a.outdir,
                                                              logits), a)


if __name__ == '__main__':
    main()
