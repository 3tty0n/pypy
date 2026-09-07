import os, sys, time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                '..', 'applevel'))

import common
import gpt2

LENGTHS = [32, 48, 64, 96, 128]


def inputs_for(cfg, buf, t, dtype):
    tokens = cfg['tokens']
    ids = [tokens[i % len(tokens)] for i in range(t)]
    idx = common.tensor([float(tok) for tok in ids], [t], False, dtype)
    d = cfg['n_embd']
    wpe_off = cfg['index']['wpe'][0]
    pos = common.tensor(list(buf[wpe_off:wpe_off + t * d]), [t, d], False,
                        dtype)
    return idx, pos


def main():
    outdir = sys.argv[1]
    iters = int(sys.argv[2]) if len(sys.argv) > 2 else 200
    dtype = os.environ.get('RTENSOR_DTYPE', 'float32')
    cfg, buf = common.load(outdir)
    model = gpt2.build_model(cfg, buf, dtype)
    h = cfg['n_head']
    masks = {}
    for length in LENGTHS:
        masks[length] = common.causal_mask(h, length, dtype)
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


if __name__ == '__main__':
    main()
