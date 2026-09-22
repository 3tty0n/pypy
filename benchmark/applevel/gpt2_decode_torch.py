"""Greedy autoregressive decode with a key/value cache, PyTorch side.

    gpt2_decode_torch.py MODE WEIGHTS_DIR [NEW] [ROUNDS]

The same algorithm as gpt2_decode.py, on the same weights and prompt, with
the block math of gpt2_torch.py: prefill seq-1 prompt tokens into a
[seq+NEW, 3d] q|k|v cache per block, then NEW single-token steps, each ending
in an argmax read back to a host int.

MODE eager slices the cache to the rows written so far, as a hand-written
PyTorch decode loop does.  The compiled modes (compile, compile-ro) need
shapes that do not change every step, so they attend over the whole cache
under a mask that hides the unwritten rows and take the position as a
tensor: the static-cache layout that compiled decoding in Hugging Face uses.
compile-ro adds CUDA graphs.  Rows are reported per mode; nothing here picks
a winner.
"""

import math
import sys
import time

import torch

import gpt2_torch
import torch_common


def median(xs):
    s = sorted(xs)
    k = len(s)
    return 0.0 if not k else (s[k // 2] if k % 2
                              else 0.5 * (s[k // 2 - 1] + s[k // 2]))


def attend(model, x, layer, cache, t, tmax):
    d = x.shape[-1]
    h = model.h
    dh = d // h
    (g1, b1, wqkv, bqkv, wo, bo, g2, b2, wf, bf, wp, bp) = layer
    n = model.ln(x, g1, b1)
    qkv = n @ wqkv + bqkv
    if isinstance(t, int):
        cache[t] = qkv[0]
        kv = cache[:t + 1]
        mask = None
    else:
        cache.index_copy_(0, t, qkv)
        kv = cache
        mask = torch.where(torch.arange(tmax, device=x.device) <= t,
                           0.0, float('-inf')).to(x.dtype)
    rows = kv.shape[0]
    q = qkv[:, :d].view(1, h, dh).transpose(0, 1)
    k = kv[:, d:2 * d].reshape(rows, h, dh).transpose(0, 1)
    v = kv[:, 2 * d:].reshape(rows, h, dh).transpose(0, 1)
    s = q @ k.transpose(-1, -2) / math.sqrt(dh)
    if mask is not None:
        s = s + mask
    c = (s.softmax(-1) @ v).transpose(0, 1).reshape(1, d)
    x = x + (c @ wo + bo)
    n = model.ln(x, g2, b2)
    u = n @ wf + bf
    u = 0.5 * u * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) *
                                    (u + 0.044715 * u * u * u)))
    return x + (u @ wp + bp)


def main():
    a = torch_common.argv()
    new = int(sys.argv[3]) if len(sys.argv) > 3 else 64
    rounds = int(sys.argv[4]) if len(sys.argv) > 4 else 5
    cfg = a.cfg
    p = cfg['seq']
    d = cfg['n_embd']
    tmax = p + new
    flat = torch_common.flat_weights(a.outdir)
    model = gpt2_torch.Model(cfg, flat, a.dtype, a.dev)
    off = cfg['index']['wpe'][0]
    wpe = flat[off:off + tmax * d].view(tmax, d).to(a.dev, a.dtype)
    caches = [torch.zeros(tmax, 3 * d, device=a.dev, dtype=a.dtype)
              for _ in model.layers]
    tokens = cfg['tokens']
    static = a.mode != 'eager'

    def step(tok, t):
        x = model.wte[tok] + wpe[t].view(1, d)
        for layer, cache in zip(model.layers, caches):
            x = attend(model, x, layer, cache, t, tmax)
        return gpt2_torch.lm_head(model, x).argmax(-1)

    fstep = torch_common.compiled(step, a)
    idx0 = torch.tensor(tokens[:p - 1], device=a.dev, dtype=torch.long)
    mask0 = model.mask[:p - 1, :p - 1]

    @torch.no_grad()
    def prefill():
        x = model.wte[idx0] + wpe[:p - 1]
        t = p - 1
        for layer, cache in zip(model.layers, caches):
            gn = model.ln(x, layer[0], layer[1])
            cache[:t] = gn @ layer[2] + layer[3]
            x = gpt2_torch.block(model, x, mask0, layer)
        return gpt2_torch.lm_head(model, x)

    @torch.no_grad()
    def generate():
        t0 = time.time()
        prefill().sum().item()
        prefill_ms = (time.time() - t0) * 1e3
        tok = tokens[p - 1]
        out = []
        us = []
        for t in range(p - 1, p - 1 + new):
            t0 = time.time()
            ti = torch.tensor([tok], device=a.dev, dtype=torch.long)
            tt = torch.tensor([t], device=a.dev) if static else t
            tok = int(fstep(ti, tt).item())
            us.append((time.time() - t0) * 1e6)
            out.append(tok)
        return out, us, prefill_ms

    results = [generate() for _ in range(rounds)]
    out = results[0][0]
    same = int(all(o == out for o, _, _ in results))
    warm = [u for _, us, _ in results[1:] for u in us] or results[0][1]
    warm_sorted = sorted(warm)
    print('decode torch-%s layers=%d embd=%d prompt=%d new=%d rounds=%d '
          'dtype=%s token_us=%.1f p90_us=%.1f cold_token_us=%.1f '
          'prefill_ms=%.2f retained_bytes=%d rounds_agree=%d tokens=%s'
          % (a.mode, cfg['n_layer'], d, p, new, rounds, a.dtname,
             median(warm), warm_sorted[int(0.9 * (len(warm_sorted) - 1))],
             median(results[0][1]),
             results[-1][2],
             torch.cuda.memory_allocated() if a.dev == 'cuda' else -1,
             same, ','.join(str(x) for x in out)))
    return 0 if same else 1


if __name__ == '__main__':
    sys.exit(main())
