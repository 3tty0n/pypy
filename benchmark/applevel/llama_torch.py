import torch

import torch_common


def main():
    a = torch_common.argv()
    cfg = a.cfg
    from transformers import AutoModelForCausalLM
    hf = AutoModelForCausalLM.from_pretrained(cfg['source'], dtype=a.dtype)
    hf = hf.to(a.dev).eval()
    idx = torch.tensor(cfg['tokens'], device=a.dev, dtype=torch.long)
    fwd = lambda i: hf(i.unsqueeze(0), use_cache=False).logits[0]
    if a.mode == 'compile':
        fwd = torch.compile(fwd)
    logits, acc, steady_us = torch_common.timed(fwd, (idx,), a)
    argmax = logits.argmax(-1).tolist()
    torch_common.report(
        'llama torch-%s layers=%d embd=%d heads=%d seq=%d vocab=%d dtype=%s '
        'iters=%d steady_us=%.1f checksum=%.6f' %
        (a.mode, cfg['n_layer'], cfg['n_embd'], cfg['n_head'], cfg['seq'],
         cfg['vocab'], a.dtname, a.iters, steady_us, acc),
        argmax, torch_common.compare(a.outdir, logits, argmax))


if __name__ == '__main__':
    main()
