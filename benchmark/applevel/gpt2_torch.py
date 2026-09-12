import math

import torch

import torch_common


class Model(torch.nn.Module):
    def __init__(self, cfg, flat, dtype, dev):
        super().__init__()
        index = cfg['index']

        def get(name):
            off, shape = index[name]
            n = 1
            for d in shape:
                n *= d
            return flat[off:off + n].view(*shape).to(dev, dtype)

        self.cfg = cfg
        self.h = cfg['n_head']
        self.eps = cfg['eps']
        self.wte = get('wte')
        self.ln_f = (get('ln_f.g'), get('ln_f.b'))
        self.layers = []
        for i in range(cfg['n_layer']):
            p = 'h.%d.' % i
            # q, k and v are one [d, 3d] GEMM, as HF's Conv1D c_attn is and
            # as the pypy/tensorpypy side already was: three separate GEMMs
            # here would be a different architecture, not a different runtime.
            wqkv = torch.cat([get(p + 'attn.%s.w' % c) for c in 'qkv'], dim=1)
            bqkv = torch.cat([get(p + 'attn.%s.b' % c) for c in 'qkv'], dim=0)
            self.layers.append((get(p + 'ln_1.g'), get(p + 'ln_1.b'),
                                wqkv, bqkv) +
                               tuple(get(p + k) for k in [
                                   'attn.proj.w', 'attn.proj.b', 'ln_2.g',
                                   'ln_2.b', 'mlp.fc.w', 'mlp.fc.b',
                                   'mlp.proj.w', 'mlp.proj.b']))
        t = cfg['seq']
        d = cfg['n_embd']
        off = index['wpe'][0]
        self.pos = flat[off:off + t * d].view(t, d).to(dev, dtype)
        m = torch.zeros(t, t, device=dev, dtype=dtype)
        self.mask = m.masked_fill(
            torch.triu(torch.ones(t, t, device=dev), 1).bool(), -1e9)

    def ln(self, x, g, b):
        return torch.nn.functional.layer_norm(x, (x.shape[-1],), g, b,
                                              self.eps)

    def forward(self, idx):
        # [seq] is the unbatched form dynamic_gpt2_torch.py drives; [B, seq]
        # is B independent sequences, which for the random-weights path is a
        # loop - the HF path below carries the batch natively and is what
        # distilgpt2 uses.
        if idx.dim() == 1:
            return blocks_forward(self, self.wte[idx] + self.pos, self.mask)
        return torch.stack([blocks_forward(self, self.wte[i] + self.pos,
                                           self.mask) for i in idx])


def blocks_forward(model, x, mask):
    t, d = x.shape
    h = model.h
    dh = d // h
    for (g1, b1, wqkv, bqkv, wo, bo, g2, b2, wf, bf, wp,
         bp) in model.layers:
        n = model.ln(x, g1, b1)
        qkv = n @ wqkv + bqkv
        q, k, v = [qkv[:, i * d:(i + 1) * d].reshape(t, h, dh).transpose(0, 1)
                   for i in range(3)]
        s = q @ k.transpose(-1, -2) / math.sqrt(dh) + mask
        c = (s.softmax(-1) @ v).transpose(0, 1).reshape(t, d)
        x = x + (c @ wo + bo)
        n = model.ln(x, g2, b2)
        u = n @ wf + bf
        u = 0.5 * u * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) *
                                        (u + 0.044715 * u * u * u)))
        x = x + (u @ wp + bp)
    x = model.ln(x, *model.ln_f)
    return x @ model.wte.t()


def main():
    a = torch_common.argv()
    cfg = a.cfg
    batch = torch_common.batch_argv()
    idx = torch.tensor([cfg['tokens']] * batch, device=a.dev,
                       dtype=torch.long)
    if cfg['source'] == 'random':
        model = Model(cfg, torch_common.flat_weights(a.outdir), a.dtype,
                      a.dev)
        fwd = model.forward
    else:
        from transformers import GPT2LMHeadModel
        hf = GPT2LMHeadModel.from_pretrained(cfg['source'], **torch_common.hf_kwargs(a))
        hf = hf.to(a.dev, a.dtype).eval()
        fwd = lambda i: hf(i).logits
    fwd = torch_common.compiled(fwd, a)
    logits, acc, steady_us = torch_common.timed(fwd, (idx,), a)
    ident = torch_common.batch_identical(logits)
    first = logits[0]
    torch_common.report(
        'gpt2 torch-%s layers=%d embd=%d heads=%d seq=%d vocab=%d dtype=%s '
        'batch=%d iters=%d steady_us=%.1f per_seq_us=%.1f '
        'batch_rows_identical=%d checksum=%.6f' %
        (a.mode, cfg['n_layer'], cfg['n_embd'], cfg['n_head'], cfg['seq'],
         cfg['vocab'], a.dtname, batch, a.iters, steady_us,
         steady_us / batch, ident, acc),
        first.argmax(-1).tolist(), torch_common.compare(a.outdir, first), a)


if __name__ == '__main__':
    main()
