import array, json, os, sys, time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                '..', '..', 'lib_pypy'))

import _metatensor
from tensorpypy.models import CausalSelfAttention, GPT2MLP, GPT2Block, GPT2

LENGTHS = [32, 48, 64, 96, 128]


def load(outdir):
    cfg = json.load(open(os.path.join(outdir, 'index.json')))
    buf = array.array('f')
    path = os.path.join(outdir, 'weights.bin')
    n = os.path.getsize(path) // 4
    buf.fromfile(open(path, 'rb'), n)
    if sys.byteorder != 'little':
        buf.byteswap()
    return cfg, buf


def build_model(cfg, buf, dtype):
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
    blocks = []
    for i in range(cfg['n_layer']):
        p = 'h.%d.' % i
        attn = CausalSelfAttention(
            cat([p + 'attn.q.w', p + 'attn.k.w', p + 'attn.v.w']),
            cat([p + 'attn.q.b', p + 'attn.k.b', p + 'attn.v.b']),
            get(p + 'attn.proj.w'), get(p + 'attn.proj.b'), h, None)
        mlp = GPT2MLP(get(p + 'mlp.fc.w'), get(p + 'mlp.fc.b'),
                                 get(p + 'mlp.proj.w'), get(p + 'mlp.proj.b'))
        blocks.append(GPT2Block(
            attn, get(p + 'ln_1.g'), get(p + 'ln_1.b'),
            get(p + 'ln_2.g'), get(p + 'ln_2.b'), mlp, eps))
    model = GPT2(get('wte'), blocks, get('ln_f.g'), get('ln_f.b'), eps)
    return model


def mask_for(h, t, dtype):
    mask = [0.0] * (h * t * t)
    for head in range(h):
        for i in range(t):
            for j in range(i + 1, t):
                mask[(head * t + i) * t + j] = -1e9
    return _metatensor.tensor(mask, [h * t, t], False, dtype)


def inputs_for(cfg, buf, t, dtype):
    tokens = cfg['tokens']
    ids = [tokens[i % len(tokens)] for i in range(t)]
    idx = _metatensor.tensor([float(tok) for tok in ids], [t], False, dtype)
    d = cfg['n_embd']
    wpe_off = cfg['index']['wpe'][0]
    pos = _metatensor.tensor(list(buf[wpe_off:wpe_off + t * d]), [t, d],
                             False, dtype)
    return idx, pos


def main():
    outdir = sys.argv[1]
    iters = int(sys.argv[2]) if len(sys.argv) > 2 else 200
    dtype = os.environ.get('RTENSOR_DTYPE', 'float32')
    cfg, buf = load(outdir)
    model = build_model(cfg, buf, dtype)
    h = cfg['n_head']
    for blk in model.blocks:
        blk.attn.mask = None
    masks = {}
    for length in LENGTHS:
        masks[length] = mask_for(h, length, dtype)
    for _ in range(5):
        t = LENGTHS[0]
        idx, pos = inputs_for(cfg, buf, t, dtype)
        for blk in model.blocks:
            blk.attn.mask = masks[t]
        model(idx, pos).sum().item()
    print('length\tus')
    for i in range(iters):
        t = LENGTHS[i % len(LENGTHS)]
        idx, pos = inputs_for(cfg, buf, t, dtype)
        for blk in model.blocks:
            blk.attn.mask = masks[t]
        t0 = time.time()
        out = model(idx, pos)
        out.sum().item()
        us = (time.time() - t0) * 1e6
        print('%d\t%.1f' % (t, us))


main()
