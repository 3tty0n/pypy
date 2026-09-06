import array, json, os, sys, time

import torch


def main():
    mode = sys.argv[1]
    outdir = sys.argv[2]
    iters = int(sys.argv[3]) if len(sys.argv) > 3 else 20
    warmup = int(sys.argv[4]) if len(sys.argv) > 4 else 10
    batch = int(sys.argv[5]) if len(sys.argv) > 5 else 1
    dtype = getattr(torch, os.environ.get('RTENSOR_DTYPE', 'float32'))
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    cfg = json.load(open(os.path.join(outdir, 'index.json')))
    buf = array.array('f')
    path = os.path.join(outdir, 'weights.bin')
    buf.fromfile(open(path, 'rb'), os.path.getsize(path) // 4)
    off, shape = cfg['index']['image']
    n = shape[0] * shape[1] * shape[2]
    px = torch.tensor(buf[off:off + n]).view(1, *shape).to(dev, dtype)
    px = px.expand(batch, -1, -1, -1).contiguous()
    import timm
    m = timm.create_model(cfg['source'], pretrained=True).to(dev, dtype)
    m = m.eval()
    fwd = m
    if mode == 'compile':
        fwd = torch.compile(m)
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
    order = logits[0].topk(5).indices.tolist()
    ref = os.path.join(outdir, 'logits_pypy.bin')
    diff = ''
    if os.path.exists(ref):
        other = torch.frombuffer(open(ref, 'rb').read(),
                                 dtype=torch.float32).to(dev)
        other = other.view_as(logits[0].float())
        diff = ' maxabsdiff=%.6g' % (
            logits[0].float() - other).abs().max().item()
    print('resnet torch-%s model=%s batch=%d classes=%d dtype=%s iters=%d '
          'steady_us=%.1f checksum=%.6f' %
          (mode, cfg['source'], batch, cfg['classes'],
           os.environ.get('RTENSOR_DTYPE', 'float32'), iters, steady_us, acc))
    print('argmax %s%s' % (' '.join(str(a) for a in order), diff))


main()
