import array, json, os, sys, time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                '..', '..', 'lib_pypy'))

import _metatensor
from tensorpypy.models import LlamaAttention, LlamaMLP, LlamaBlock, Llama


def load(outdir):
    cfg = json.load(open(os.path.join(outdir, 'index.json')))
    buf = array.array('f')
    path = os.path.join(outdir, 'weights.bin')
    buf.fromfile(open(path, 'rb'), os.path.getsize(path) // 4)
    if sys.byteorder != 'little':
        buf.byteswap()
    return cfg, buf


def build(cfg, buf, dtype):
    index = cfg['index']

    def raw(name):
        off, shape = index[name]
        n = 1
        for d in shape:
            n *= d
        return list(buf[off:off + n]), shape

    def get(name):
        values, shape = raw(name)
        return _metatensor.tensor(values, shape, False, dtype)

    def cat(names):
        parts = [raw(n) for n in names]
        rows, cols = parts[0][1]
        out = []
        for r in range(rows):
            for values, _ in parts:
                out.extend(values[r * cols:(r + 1) * cols])
        return _metatensor.tensor(out, [rows, cols * len(parts)], False,
                                  dtype)

    t = cfg['seq']
    h = cfg['n_head']
    eps = cfg['eps']
    idx = _metatensor.tensor([float(tok) for tok in cfg['tokens']], [t], False,
                         dtype)
    mask = [0.0] * (h * t * t)
    for head in range(h):
        for i in range(t):
            for j in range(i + 1, t):
                mask[(head * t + i) * t + j] = -1e9
    mask = _metatensor.tensor(mask, [h * t, t], False, dtype)
    dh = cfg['head_dim']
    cvals, cshape = raw('rope.cos')
    svals, _ = raw('rope.sin')
    d = cshape[1]
    c3, s3 = [], []
    for r in range(cshape[0]):
        row = cvals[r * d:(r + 1) * d]
        srow = svals[r * d:(r + 1) * d]
        srow = [-v if (i % dh) < dh // 2 else v
                for i, v in enumerate(srow)]
        c3.extend(row + row + [1.0] * d)
        s3.extend(srow + srow + [0.0] * d)
    cos = _metatensor.tensor(c3, [cshape[0], 3 * d], False, dtype)
    sin = _metatensor.tensor(s3, [cshape[0], 3 * d], False, dtype)
    blocks = []
    for i in range(cfg['n_layer']):
        pre = 'h.%d.' % i
        attn = LlamaAttention(
            cat([pre + 'attn.q.w', pre + 'attn.k.w', pre + 'attn.v.w']),
            get(pre + 'attn.proj.w'), h, mask, cos, sin, dh)
        mlp = LlamaMLP(get(pre + 'mlp.gate.w'),
                                  get(pre + 'mlp.up.w'),
                                  get(pre + 'mlp.down.w'))
        blocks.append(LlamaBlock(attn, get(pre + 'norm1.g'),
                                            get(pre + 'norm2.g'), mlp, eps))
    head = None if cfg['tied'] else get('lm_head.w')
    return Llama(get('wte'), blocks, get('norm_f.g'), head,
                            eps), idx


def main():
    outdir = sys.argv[1]
    iters = int(sys.argv[2]) if len(sys.argv) > 2 else 20
    warmup = int(sys.argv[3]) if len(sys.argv) > 3 else 10
    dtype = os.environ.get('RTENSOR_DTYPE', 'float32')
    cfg, buf = load(outdir)
    model, idx = build(cfg, buf, dtype)
    for i in range(warmup):
        logits = model(idx)
    logits.sum().item()
    t0 = time.time()
    for i in range(iters):
        logits = model(idx)
    acc = logits.sum().item()
    steady_us = (time.time() - t0) / iters * 1e6
    flat = logits.tolist()
    t, v = cfg['seq'], cfg['vocab']
    argmax = []
    for i in range(t):
        best, bestj = flat[i * v], 0
        for j in range(1, v):
            if flat[i * v + j] > best:
                best, bestj = flat[i * v + j], j
        argmax.append(bestj)
    out = array.array('f', flat)
    if sys.byteorder != 'little':
        out.byteswap()
    out.tofile(open(os.path.join(outdir, 'logits_pypy.bin'), 'wb'))
    print('llama pypy layers=%d embd=%d heads=%d kv=%d seq=%d vocab=%d '
          'dtype=%s iters=%d steady_us=%.1f checksum=%.6f' %
          (cfg['n_layer'], cfg['n_embd'], cfg['n_head'], cfg['n_kv_head'],
           cfg['seq'], cfg['vocab'], dtype, iters, steady_us, acc))
    print('argmax %s' % ' '.join([str(a) for a in argmax]))


main()
