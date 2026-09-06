import json, os, sys, time

import torch


def main():
    mode = sys.argv[1]
    outdir = sys.argv[2]
    iters = int(sys.argv[3]) if len(sys.argv) > 3 else 20
    warmup = int(sys.argv[4]) if len(sys.argv) > 4 else 10
    dtype = getattr(torch, os.environ.get('RTENSOR_DTYPE', 'float32'))
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    cfg = json.load(open(os.path.join(outdir, 'index.json')))
    from transformers import BertForMaskedLM
    hf = BertForMaskedLM.from_pretrained(cfg['source']).to(dev, dtype).eval()
    idx = torch.tensor([cfg['tokens']], device=dev, dtype=torch.long)
    fwd = lambda i: hf(i).logits[0]
    if mode == 'compile':
        fwd = torch.compile(fwd)
    with torch.no_grad():
        for i in range(warmup):
            logits = fwd(idx)
        if dev == 'cuda':
            torch.cuda.synchronize()
        t0 = time.time()
        for i in range(iters):
            logits = fwd(idx)
        if dev == 'cuda':
            torch.cuda.synchronize()
        acc = logits.double().sum().item()
        steady_us = (time.time() - t0) / iters * 1e6
    argmax = logits.argmax(-1).tolist()
    ref = os.path.join(outdir, 'logits_pypy.bin')
    diff = ''
    if os.path.exists(ref):
        other = torch.frombuffer(open(ref, 'rb').read(),
                                 dtype=torch.float32).to(dev)
        other = other.view_as(logits.float())
        diff = ' maxabsdiff=%.6g' % (logits.float() - other).abs().max().item()
    print('bert torch-%s layers=%d embd=%d heads=%d seq=%d vocab=%d dtype=%s '
          'iters=%d steady_us=%.1f checksum=%.6f' %
          (mode, cfg['n_layer'], cfg['n_embd'], cfg['n_head'], cfg['seq'],
           cfg['vocab'], os.environ.get('RTENSOR_DTYPE', 'float32'), iters,
           steady_us, acc))
    print('argmax %s%s' % (' '.join(str(a) for a in argmax), diff))


main()
