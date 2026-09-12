import torch

import torch_common


def main():
    a = torch_common.argv()
    cfg = a.cfg
    from transformers import BertForMaskedLM
    hf = BertForMaskedLM.from_pretrained(cfg['source'], **torch_common.hf_kwargs(a)).to(a.dev,
                                                           a.dtype).eval()
    batch = torch_common.batch_argv()
    idx = torch.tensor([cfg['tokens']] * batch, device=a.dev,
                       dtype=torch.long)
    fwd = lambda i: hf(i).logits
    fwd = torch_common.compiled(fwd, a)
    logits, acc, steady_us = torch_common.timed(fwd, (idx,), a)
    ident = torch_common.batch_identical(logits)
    first = logits[0]
    torch_common.report(
        'bert torch-%s layers=%d embd=%d heads=%d seq=%d vocab=%d dtype=%s '
        'batch=%d iters=%d steady_us=%.1f per_seq_us=%.1f '
        'batch_rows_identical=%d checksum=%.6f' %
        (a.mode, cfg['n_layer'], cfg['n_embd'], cfg['n_head'], cfg['seq'],
         cfg['vocab'], a.dtname, batch, a.iters, steady_us,
         steady_us / batch, ident, acc),
        first.argmax(-1).tolist(), torch_common.compare(a.outdir, first), a)


if __name__ == '__main__':
    main()
