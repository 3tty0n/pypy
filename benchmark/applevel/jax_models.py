"""JAX/XLA side of the end-to-end models, from the exported weights.

    jax_models.py MODEL MODE WEIGHTS ITERS WARMUP   MODEL: gpt2 bert vit mixer
                                                    MODE: jax | iree

Each forward is a transcription of the tensorpypy.models class the pypy side
runs (Bert, GPT2, ViT, Mixer), on the same weights.bin/index.json, so the
three systems compute the same function on the same numbers.  The report
line is the one torch_common.report prints, plus compile_ms (jit lower +
XLA compile, no execution) and first_run_ms (first executed forward), so
the compile cost is separate from steady_us.  Weights are jit arguments,
not closure constants, so XLA does not fold a 50k x 768 embedding table.
"""
import array, json, math, os, sys, time

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import iree_adapter

jax.config.update("jax_enable_x64", True)
# torch eager and cuBLAS do full fp32 GEMMs by default; XLA would use TF32.
jax.config.update("jax_default_matmul_precision", "highest")


def load(outdir):
    cfg = json.load(open(os.path.join(outdir, 'index.json')))
    buf = array.array('f')
    path = os.path.join(outdir, 'weights.bin')
    buf.fromfile(open(path, 'rb'), os.path.getsize(path) // 4)
    if sys.byteorder != 'little':
        buf.byteswap()
    return cfg, np.frombuffer(buf.tobytes(), dtype=np.float32)


class Weights(object):
    def __init__(self, cfg, flat, dtype):
        self.index = cfg['index']
        self.flat = flat
        self.dtype = dtype

    def raw(self, name):
        off, shape = self.index[name]
        n = int(np.prod(shape))
        return self.flat[off:off + n].reshape(shape)

    def get(self, name):
        return jnp.asarray(self.raw(name), dtype=self.dtype)

    def cat(self, names):
        return jnp.asarray(np.concatenate([self.raw(n) for n in names], -1),
                           dtype=self.dtype)


def layer_norm(x, g, b, eps):
    mu = x.mean(-1, keepdims=True)
    var = ((x - mu) ** 2).mean(-1, keepdims=True)
    return (x - mu) / jnp.sqrt(var + eps) * g + b


def gelu_tanh(u):
    return 0.5 * u * (1.0 + jnp.tanh(math.sqrt(2.0 / math.pi) *
                                     (u + 0.044715 * u * u * u)))


def gelu_erf(u):
    return jax.nn.gelu(u, approximate=False)


def attention(x, p, h, mask):
    t, d = x.shape
    dh = d // h
    qkv = x @ p['wqkv'] + p['bqkv']
    q, k, v = [qkv[:, i * d:(i + 1) * d].reshape(t, h, dh).transpose(1, 0, 2)
               for i in range(3)]
    s = jnp.einsum('hqd,hkd->hqk', q, k) / math.sqrt(dh)
    if mask is not None:
        s = s + mask
    c = jnp.einsum('hqk,hkd->hqd', jax.nn.softmax(s, -1), v)
    c = c.transpose(1, 0, 2).reshape(t, d)
    return c @ p['wo'] + p['bo']


def mlp(x, p, act):
    return act(x @ p['wfc'] + p['bfc']) @ p['wproj'] + p['bproj']


def block_params(w, i):
    p = 'h.%d.' % i
    return {'wqkv': w.cat([p + 'attn.q.w', p + 'attn.k.w', p + 'attn.v.w']),
            'bqkv': w.cat([p + 'attn.q.b', p + 'attn.k.b', p + 'attn.v.b']),
            'wo': w.get(p + 'attn.proj.w'), 'bo': w.get(p + 'attn.proj.b'),
            'wfc': w.get(p + 'mlp.fc.w'), 'bfc': w.get(p + 'mlp.fc.b'),
            'wproj': w.get(p + 'mlp.proj.w'), 'bproj': w.get(p + 'mlp.proj.b'),
            'g1': w.get(p + 'ln_1.g'), 'b1': w.get(p + 'ln_1.b'),
            'g2': w.get(p + 'ln_2.g'), 'b2': w.get(p + 'ln_2.b')}


def gpt2_block(x, p, h, mask, eps, act):
    x = x + attention(layer_norm(x, p['g1'], p['b1'], eps), p, h, mask)
    return x + mlp(layer_norm(x, p['g2'], p['b2'], eps), p, act)


def bert_block(x, p, h, eps):
    x = layer_norm(x + attention(x, p, h, None), p['g1'], p['b1'], eps)
    return layer_norm(x + mlp(x, p, gelu_erf), p['g2'], p['b2'], eps)


# -- models: build(cfg, flat, dtype) -> (params, inputs, forward) -------------

def build_gpt2(cfg, flat, dtype):
    w = Weights(cfg, flat, dtype)
    t, d, h, eps = cfg['seq'], cfg['n_embd'], cfg['n_head'], cfg['eps']
    params = {'wte': w.get('wte'), 'pos': w.get('wpe')[:t],
              'gf': w.get('ln_f.g'), 'bf': w.get('ln_f.b'),
              'blocks': [block_params(w, i) for i in range(cfg['n_layer'])]}
    mask = jnp.where(jnp.triu(jnp.ones((t, t), bool), 1), -1e9, 0.0)
    params['mask'] = mask.astype(dtype)
    idx = jnp.asarray(cfg['tokens'], dtype=jnp.int32)

    def forward(p, idx):
        x = p['wte'][idx] + p['pos']
        for bp in p['blocks']:
            x = gpt2_block(x, bp, h, p['mask'], eps, gelu_tanh)
        x = layer_norm(x, p['gf'], p['bf'], eps)
        return x @ p['wte'].T
    return params, (idx,), forward


def build_bert(cfg, flat, dtype):
    w = Weights(cfg, flat, dtype)
    h, eps = cfg['n_head'], cfg['eps']
    params = {'wte': w.get('wte'), 'emb': w.get('emb'),
              'ge': w.get('emb.g'), 'be': w.get('emb.b'),
              'wd': w.get('mlm.dense.w'), 'bd': w.get('mlm.dense.b'),
              'gm': w.get('mlm.ln.g'), 'bm': w.get('mlm.ln.b'),
              'bias': w.get('mlm.bias'),
              'blocks': [block_params(w, i) for i in range(cfg['n_layer'])]}
    idx = jnp.asarray(cfg['tokens'], dtype=jnp.int32)

    def forward(p, idx):
        x = layer_norm(p['wte'][idx] + p['emb'], p['ge'], p['be'], eps)
        for bp in p['blocks']:
            x = bert_block(x, bp, h, eps)
        hh = gelu_erf(x @ p['wd'] + p['bd'])
        hh = layer_norm(hh, p['gm'], p['bm'], eps)
        return hh @ p['wte'].T + p['bias']
    return params, (idx,), forward


def patches(img, k):
    """[3, s, s] image -> [(s/k)^2, 3*k*k] patches in (c, kh, kw) order."""
    c, s, _ = img.shape
    g = s // k
    return img.reshape(c, g, k, g, k).transpose(1, 3, 0, 2, 4).reshape(
        g * g, c * k * k)


def build_vit(cfg, flat, dtype):
    w = Weights(cfg, flat, dtype)
    h, eps, k = cfg['n_head'], cfg['eps'], cfg['patch_size']
    params = {'wp': w.get('wp'), 'emb': w.get('emb'),
              'gf': w.get('ln_f.g'), 'bf': w.get('ln_f.b'),
              'head': w.get('head.w'), 'bhead': w.get('head.b'),
              'blocks': [block_params(w, i) for i in range(cfg['n_layer'])]}
    img = jnp.asarray(w.raw('image'), dtype=dtype)

    def forward(p, img):
        # token 0 is the cls slot: a zero patch, its embedding lives in emb[0]
        x = jnp.concatenate([jnp.zeros((1, cfg['patch']), dtype),
                             patches(img, k)])
        x = x @ p['wp'] + p['emb']
        for bp in p['blocks']:
            x = gpt2_block(x, bp, h, None, eps, gelu_erf)
        x = layer_norm(x, p['gf'], p['bf'], eps)
        return x[:1] @ p['head'] + p['bhead']
    return params, (img,), forward


def build_mixer(cfg, flat, dtype):
    w = Weights(cfg, flat, dtype)
    eps, k = cfg['eps'], cfg['patch_size']
    blocks = []
    for i in range(cfg['n_layer']):
        p = 'h.%d.' % i
        blocks.append({
            'g1': w.get(p + 'ln_1.g'), 'b1': w.get(p + 'ln_1.b'),
            'wt1': w.get(p + 'tok.fc.w'), 'bt1': w.get(p + 'tok.fc.b'),
            'wt2': w.get(p + 'tok.proj.w'), 'bt2': w.get(p + 'tok.proj.b'),
            'g2': w.get(p + 'ln_2.g'), 'b2': w.get(p + 'ln_2.b'),
            'wfc': w.get(p + 'mlp.fc.w'), 'bfc': w.get(p + 'mlp.fc.b'),
            'wproj': w.get(p + 'mlp.proj.w'), 'bproj': w.get(p + 'mlp.proj.b')})
    params = {'wp': w.get('wp'), 'bp': w.get('bp'),
              'gf': w.get('ln_f.g'), 'bf': w.get('ln_f.b'),
              'head': w.get('head.w'), 'bhead': w.get('head.b'),
              'blocks': blocks}
    img = jnp.asarray(w.raw('image'), dtype=dtype)

    def forward(p, img):
        x = patches(img, k) @ p['wp'] + p['bp']
        for bp in p['blocks']:
            hh = layer_norm(x, bp['g1'], bp['b1'], eps)
            u = gelu_erf(bp['wt1'] @ hh + bp['bt1'])
            x = x + (bp['wt2'] @ u + bp['bt2'])
            x = x + mlp(layer_norm(x, bp['g2'], bp['b2'], eps), bp, gelu_erf)
        x = layer_norm(x, p['gf'], p['bf'], eps)
        return x.mean(0, keepdims=True) @ p['head'] + p['bhead']
    return params, (img,), forward


BUILD = {'gpt2': build_gpt2, 'bert': build_bert, 'vit': build_vit,
         'mixer': build_mixer}


# -- driver -------------------------------------------------------------------

def compare(outdir, logits):
    ref = os.path.join(outdir, 'logits_pypy.bin')
    if not os.path.exists(ref):
        return ''
    other = np.fromfile(ref, dtype=np.float32).reshape(logits.shape)
    mine = np.asarray(logits, dtype=np.float32)
    return ' maxabsdiff=%.6g' % np.abs(mine - other).max()


def main():
    model, mode, outdir = sys.argv[1], sys.argv[2], sys.argv[3]
    iters = int(sys.argv[4]) if len(sys.argv) > 4 else 20
    warmup = int(sys.argv[5]) if len(sys.argv) > 5 else 10
    dtname = os.environ.get('RTENSOR_DTYPE', 'float32')
    dtype = jnp.dtype(dtname)
    cfg, flat = load(outdir)
    params, args, forward = BUILD[model](cfg, flat, dtype)
    jax.block_until_ready(params)

    t0 = time.perf_counter()
    if mode == 'iree':
        exe = iree_adapter.compile(forward, params, *args)
        compile_ms = exe.compile_ms
        run = exe.bind(params, *args)
        block = exe.sync
    else:
        fwd = jax.jit(forward).lower(params, *args).compile()
        compile_ms = (time.perf_counter() - t0) * 1e3
        run = lambda: fwd(params, *args)
        block = jax.block_until_ready
    t0 = time.perf_counter()
    logits = block(run())
    first_ms = (time.perf_counter() - t0) * 1e3
    for i in range(warmup):
        logits = run()
    block(logits)
    t0 = time.time()
    for i in range(iters):
        logits = run()
    logits = block(logits)
    steady_us = (time.time() - t0) / iters * 1e6
    acc = float(np.asarray(logits, np.float64).sum())

    if model in ('gpt2', 'bert'):
        desc = 'layers=%d embd=%d heads=%d seq=%d vocab=%d' % (
            cfg['n_layer'], cfg['n_embd'], cfg['n_head'], cfg['seq'],
            cfg['vocab'])
        order = np.asarray(logits).argmax(-1).tolist()
    elif model == 'vit':
        desc = 'layers=%d embd=%d heads=%d tokens=%d classes=%d' % (
            cfg['n_layer'], cfg['n_embd'], cfg['n_head'], cfg['tokens'],
            cfg['classes'])
        order = np.argsort(-np.asarray(logits)[0])[:5].tolist()
    else:
        desc = 'layers=%d embd=%d tokens=%d classes=%d' % (
            cfg['n_layer'], cfg['n_embd'], cfg['tokens'], cfg['classes'])
        order = np.argsort(-np.asarray(logits)[0])[:5].tolist()
    print('%s %s %s dtype=%s iters=%d steady_us=%.1f checksum=%.6f '
          'compile_ms=%.1f first_run_ms=%.1f' % (
              model, mode, desc, dtname, iters, steady_us, acc, compile_ms,
              first_ms))
    print('argmax %s%s' % (' '.join(str(a) for a in order),
                           compare(outdir, logits)))


if __name__ == '__main__':
    main()
