import array, json, os, sys, time

import torch


def main():
    mode = sys.argv[1]
    outdir = sys.argv[2]
    iters = int(sys.argv[3]) if len(sys.argv) > 3 else 20
    warmup = int(sys.argv[4]) if len(sys.argv) > 4 else 10
    dtype = getattr(torch, os.environ.get('RTENSOR_DTYPE', 'float32'))
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    cfg = json.load(open(os.path.join(outdir, 'index.json')))
    buf = array.array('f')
    path = os.path.join(outdir, 'weights.bin')
    buf.fromfile(open(path, 'rb'), os.path.getsize(path) // 4)
    off, shape = cfg['index']['image']
    n = shape[0] * shape[1] * shape[2]
    px = torch.tensor(buf[off:off + n]).view(1, *shape).to(dev, dtype)
    from transformers import ViTForImageClassification
    hf = ViTForImageClassification.from_pretrained(cfg['source'])
    hf = hf.to(dev, dtype).eval()
    fwd = lambda x: hf(x).logits[0]
    if mode == 'compile':
        fwd = torch.compile(fwd)
    with torch.no_grad():
        for i in range(warmup):
            logits = fwd(px)
        if dev == 'cuda':
            torch.cuda.synchronize()
        t0 = time.time()
        for i in range(iters):
            logits = fwd(px)
        if dev == 'cuda':
            torch.cuda.synchronize()
        acc = logits.double().sum().item()
        steady_us = (time.time() - t0) / iters * 1e6
    order = logits.topk(5).indices.tolist()
    ref = os.path.join(outdir, 'logits_pypy.bin')
    diff = ''
    if os.path.exists(ref):
        other = torch.frombuffer(open(ref, 'rb').read(),
                                 dtype=torch.float32).to(dev)
        other = other.view_as(logits.float())
        diff = ' maxabsdiff=%.6g' % (logits.float() - other).abs().max().item()
    print('vit torch-%s layers=%d embd=%d heads=%d tokens=%d classes=%d '
          'dtype=%s iters=%d steady_us=%.1f checksum=%.6f' %
          (mode, cfg['n_layer'], cfg['n_embd'], cfg['n_head'], cfg['tokens'],
           cfg['classes'], os.environ.get('RTENSOR_DTYPE', 'float32'), iters,
           steady_us, acc))
    print('argmax %s%s' % (' '.join(str(a) for a in order), diff))


main()
