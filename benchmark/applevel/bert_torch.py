import torch

import torch_common


def main():
    a = torch_common.argv()
    cfg = a.cfg
    from transformers import BertForMaskedLM
    hf = BertForMaskedLM.from_pretrained(cfg['source']).to(a.dev,
                                                           a.dtype).eval()
    idx = torch.tensor([cfg['tokens']], device=a.dev, dtype=torch.long)
    fwd = lambda i: hf(i).logits[0]
    if a.mode == 'compile':
        fwd = torch.compile(fwd)
    logits, acc, steady_us = torch_common.timed(fwd, (idx,), a)
    torch_common.report(
        'bert torch-%s layers=%d embd=%d heads=%d seq=%d vocab=%d dtype=%s '
        'iters=%d steady_us=%.1f checksum=%.6f' %
        (a.mode, cfg['n_layer'], cfg['n_embd'], cfg['n_head'], cfg['seq'],
         cfg['vocab'], a.dtname, a.iters, steady_us, acc),
        logits.argmax(-1).tolist(), torch_common.compare(a.outdir, logits), a)


if __name__ == '__main__':
    main()
