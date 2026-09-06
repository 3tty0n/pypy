import argparse, io, json, os, struct, urllib.request

URL = 'http://images.cocodataset.org/val2017/000000039769.jpg'


def pixel_values(model_id, size):
    from transformers import ViTImageProcessor
    proc = ViTImageProcessor.from_pretrained(model_id)
    try:
        from PIL import Image
        raw = urllib.request.urlopen(URL, timeout=30).read()
        img = Image.open(io.BytesIO(raw)).convert('RGB')
        src = URL
    except Exception as exc:
        print('no image (%s), using synthetic' % exc)
        import torch
        px = torch.arange(3 * size * size).float().reshape(1, 3, size, size)
        return (px % 251.0) / 251.0 * 2.0 - 1.0, 'synthetic'
    return proc(img, return_tensors='pt')['pixel_values'], src


def hf_weights(model_id, cfg):
    from transformers import ViTForImageClassification
    m = ViTForImageClassification.from_pretrained(model_id)
    sd = m.state_dict()
    c = m.config
    d = c.hidden_size
    px, src = pixel_values(model_id, c.image_size)
    ntok = (c.image_size // c.patch_size) ** 2 + 1
    cfg.update({'n_embd': d, 'n_head': c.num_attention_heads,
                'n_layer': c.num_hidden_layers, 'eps': c.layer_norm_eps,
                'image_size': c.image_size, 'patch_size': c.patch_size,
                'tokens': ntok, 'patch': 3 * c.patch_size ** 2,
                'classes': c.num_labels, 'image': src})
    w = {}

    def put(key, t, transpose=False):
        if transpose:
            t = t.t()
        w[key] = (list(t.shape), t.contiguous().view(-1).float().tolist())

    put('image', px[0])
    pre = 'vit.embeddings.'
    put('wp', sd[pre + 'patch_embeddings.projection.weight'].reshape(
        d, cfg['patch']), True)
    emb = (sd[pre + 'position_embeddings'][0] +
           sd[pre + 'patch_embeddings.projection.bias'])
    emb[0] = sd[pre + 'cls_token'][0, 0] + sd[pre + 'position_embeddings'][0, 0]
    put('emb', emb)
    for i in range(c.num_hidden_layers):
        p = 'vit.layers.%d.' % i
        for src_k, dst in [('attention.q_proj', 'attn.q'),
                           ('attention.k_proj', 'attn.k'),
                           ('attention.v_proj', 'attn.v'),
                           ('attention.o_proj', 'attn.proj'),
                           ('mlp.fc1', 'mlp.fc'), ('mlp.fc2', 'mlp.proj')]:
            put('h.%d.%s.w' % (i, dst), sd[p + src_k + '.weight'], True)
            put('h.%d.%s.b' % (i, dst), sd[p + src_k + '.bias'])
        for src_k, dst in [('layernorm_before', 'ln_1'),
                           ('layernorm_after', 'ln_2')]:
            put('h.%d.%s.g' % (i, dst), sd[p + src_k + '.weight'])
            put('h.%d.%s.b' % (i, dst), sd[p + src_k + '.bias'])
    put('ln_f.g', sd['vit.layernorm.weight'])
    put('ln_f.b', sd['vit.layernorm.bias'])
    put('head.w', sd['classifier.weight'], True)
    put('head.b', sd['classifier.bias'])
    return w


def write(outdir, cfg, weights):
    if not os.path.isdir(outdir):
        os.makedirs(outdir)
    index = {}
    off = 0
    with open(os.path.join(outdir, 'weights.bin'), 'wb') as f:
        for name in sorted(weights):
            shape, data = weights[name]
            f.write(struct.pack('<%df' % len(data), *data))
            index[name] = [off, shape]
            off += len(data)
    cfg['index'] = index
    with open(os.path.join(outdir, 'index.json'), 'w') as f:
        json.dump(cfg, f)
    print('wrote %s: %d tensors, %d floats' % (outdir, len(index), off))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('outdir')
    ap.add_argument('--model', default='WinKawaks/vit-tiny-patch16-224')
    a = ap.parse_args()
    cfg = {'source': a.model}
    write(a.outdir, cfg, hf_weights(a.model, cfg))


main()
