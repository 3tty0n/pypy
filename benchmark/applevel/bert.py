import array, json, os, sys, time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                '..', '..', 'lib_pypy'))

import _metatensor
from tensorpypy.functional import gelu_erf
from tensorpypy.models import CausalSelfAttention, GPT2MLP, BertBlock, Bert


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
        shape = parts[0][1]
        if len(shape) == 1:
            out = []
            for values, _ in parts:
                out.extend(values)
            return _metatensor.tensor(out, [len(out)], False, dtype)
        rows, cols = shape
        out = []
        for r in range(rows):
            for values, _ in parts:
                out.extend(values[r * cols:(r + 1) * cols])
        return _metatensor.tensor(out, [rows, cols * len(parts)], False,
                                  dtype)

    h = cfg['n_head']
    eps = cfg['eps']
    idx = _metatensor.tensor([float(tok) for tok in cfg['tokens']],
                             [cfg['seq']], False, dtype)
    blocks = []
    for i in range(cfg['n_layer']):
        p = 'h.%d.' % i
        attn = CausalSelfAttention(
            cat([p + 'attn.q.w', p + 'attn.k.w', p + 'attn.v.w']),
            cat([p + 'attn.q.b', p + 'attn.k.b', p + 'attn.v.b']),
            get(p + 'attn.proj.w'), get(p + 'attn.proj.b'), h, None)
        mlp = GPT2MLP(get(p + 'mlp.fc.w'), get(p + 'mlp.fc.b'),
                      get(p + 'mlp.proj.w'), get(p + 'mlp.proj.b'), gelu_erf)
        blocks.append(BertBlock(attn, get(p + 'ln_1.g'), get(p + 'ln_1.b'),
                                mlp, get(p + 'ln_2.g'), get(p + 'ln_2.b'),
                                eps))
    model = Bert(get('wte'), get('emb'), get('emb.g'), get('emb.b'), blocks,
                 get('mlm.dense.w'), get('mlm.dense.b'), get('mlm.ln.g'),
                 get('mlm.ln.b'), get('mlm.bias'), eps)
    return model, idx


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
    print('bert pypy layers=%d embd=%d heads=%d seq=%d vocab=%d dtype=%s '
          'iters=%d steady_us=%.1f checksum=%.6f' %
          (cfg['n_layer'], cfg['n_embd'], cfg['n_head'], cfg['seq'],
           cfg['vocab'], dtype, iters, steady_us, acc))
    print('argmax %s' % ' '.join([str(a) for a in argmax]))


main()
