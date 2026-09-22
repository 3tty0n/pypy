"""Greedy autoregressive decode with a key/value cache, Hugging Face side.

    llama_decode_torch.py MODE WEIGHTS_DIR [NEW] [ROUNDS]

For the Llama-architecture models the baseline is the deployed code path
itself: the model class from transformers with its own cache.  The loop is
the one lm_decode.py runs - prefill seq-1 prompt tokens, then NEW single-token
steps each ending in an argmax read back to a host int - so the two sides
time the same thing.

MODE eager uses DynamicCache, transformers' default.  compile and compile-ro
use StaticCache with torch.compile over the model's forward, the recipe
transformers documents for compiled decoding; compile-ro adds CUDA graphs.

Unlike the GPT-2 mirror, this attends with the model's own grouped key/value
heads, where the MetaTensor side repeats them to every head at export time;
same function, fewer bytes of cache here.
"""

import sys
import time

import torch

import torch_common


def median(xs):
    s = sorted(xs)
    k = len(s)
    return 0.0 if not k else (s[k // 2] if k % 2
                              else 0.5 * (s[k // 2 - 1] + s[k // 2]))


def main():
    a = torch_common.argv()
    new = int(sys.argv[3]) if len(sys.argv) > 3 else 64
    rounds = int(sys.argv[4]) if len(sys.argv) > 4 else 5
    cfg = a.cfg
    p = cfg['seq']
    tokens = cfg['tokens']
    from transformers import AutoModelForCausalLM, DynamicCache, StaticCache
    hf = AutoModelForCausalLM.from_pretrained(cfg['source'], dtype=a.dtype)
    hf = hf.to(a.dev).eval()
    static = a.mode != 'eager'
    fwd = torch_common.compiled(hf.forward, a) if static else hf.forward
    idx0 = torch.tensor([tokens[:p - 1]], device=a.dev, dtype=torch.long)

    @torch.no_grad()
    def generate():
        if static:
            cache = StaticCache(config=hf.config, max_cache_len=p + new)
        else:
            cache = DynamicCache()
        t0 = time.time()
        out = hf(idx0, past_key_values=cache, use_cache=True,
                 cache_position=torch.arange(p - 1, device=a.dev))
        out.logits.sum().item()
        prefill_ms = (time.time() - t0) * 1e3
        tok = tokens[p - 1]
        gen = []
        us = []
        for t in range(p - 1, p - 1 + new):
            t0 = time.time()
            ti = torch.tensor([[tok]], device=a.dev, dtype=torch.long)
            pos = torch.tensor([t], device=a.dev)
            logits = fwd(ti, past_key_values=cache, use_cache=True,
                         cache_position=pos).logits
            tok = int(logits[0, -1].argmax().item())
            us.append((time.time() - t0) * 1e6)
            gen.append(tok)
        return gen, us, prefill_ms

    results = [generate() for _ in range(rounds)]
    out = results[0][0]
    same = int(all(o == out for o, _, _ in results))
    warm = [u for _, us, _ in results[1:] for u in us] or results[0][1]
    warm_sorted = sorted(warm)
    print('decode torch-%s layers=%d embd=%d prompt=%d new=%d rounds=%d '
          'dtype=%s token_us=%.1f p90_us=%.1f cold_token_us=%.1f '
          'prefill_ms=%.2f retained_bytes=%d rounds_agree=%d tokens=%s'
          % (a.mode, cfg['n_layer'], cfg['n_embd'], p, new, rounds, a.dtname,
             median(warm), warm_sorted[int(0.9 * (len(warm_sorted) - 1))],
             median(results[0][1]), results[-1][2],
             torch.cuda.memory_allocated() if a.dev == 'cuda' else -1,
             same, ','.join(str(x) for x in out)))
    return 0 if same else 1


if __name__ == '__main__':
    sys.exit(main())
