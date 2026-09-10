import torch

import torch_common


def main():
    a = torch_common.argv()
    cfg = a.cfg
    px = torch_common.image(a).to(a.dev, a.dtype)
    import timm
    m = timm.create_model(cfg['source'], pretrained=True).to(a.dev, a.dtype)
    fwd = m.eval()
    if a.mode == 'compile':
        fwd = torch.compile(fwd)
    logits, acc, steady_us = torch_common.timed(fwd, (px,), a)
    torch_common.report(
        'mixer torch-%s layers=%d embd=%d tokens=%d classes=%d dtype=%s '
        'iters=%d steady_us=%.1f checksum=%.6f' %
        (a.mode, cfg['n_layer'], cfg['n_embd'], cfg['tokens'], cfg['classes'],
         a.dtname, a.iters, steady_us, acc),
        logits[0].topk(5).indices.tolist(),
        torch_common.compare(a.outdir, logits[0]), a)


if __name__ == '__main__':
    main()
