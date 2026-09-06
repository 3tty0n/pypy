import argparse, io, json, os, struct, urllib.request

URL = 'http://images.cocodataset.org/val2017/000000039769.jpg'


def pixel_values(model):
    import timm, torch
    cfg = timm.data.resolve_data_config({}, model=model)
    try:
        from PIL import Image
        raw = urllib.request.urlopen(URL, timeout=30).read()
        img = Image.open(io.BytesIO(raw)).convert('RGB')
        return timm.data.create_transform(**cfg)(img).unsqueeze(0), URL
    except Exception as exc:
        print('no image (%s), using synthetic' % exc)
        s = cfg['input_size'][1]
        px = torch.arange(3 * s * s).float().reshape(1, 3, s, s)
        return (px % 251.0) / 251.0 * 2.0 - 1.0, 'synthetic'


def hf_weights(model_id, cfg):
    import timm
    m = timm.create_model(model_id, pretrained=True).eval()
    sd = m.state_dict()
    px, src = pixel_values(m)
    w = {}

    def put(key, t, transpose=False):
        if transpose:
            t = t.t()
        w[key] = (list(t.shape), t.contiguous().view(-1).float().tolist())

    put('image', px[0])
    o, c, k, _ = sd['stem.proj.weight'].shape
    put('wp', sd['stem.proj.weight'].reshape(o, c * k * k), True)
    put('bp', sd['stem.proj.bias'])
    n = 0
    while 'blocks.%d.norm1.weight' % n in sd:
        p = 'blocks.%d.' % n
        key = 'h.%d.' % n
        for src_k, dst in [('norm1', 'ln_1'), ('norm2', 'ln_2')]:
            put(key + dst + '.g', sd[p + src_k + '.weight'])
            put(key + dst + '.b', sd[p + src_k + '.bias'])
        for src_k, dst, col in [('mlp_tokens.fc1', 'tok.fc', True),
                                ('mlp_tokens.fc2', 'tok.proj', True),
                                ('mlp_channels.fc1', 'mlp.fc', False),
                                ('mlp_channels.fc2', 'mlp.proj', False)]:
            put(key + dst + '.w', sd[p + src_k + '.weight'], not col)
            b = sd[p + src_k + '.bias']
            put(key + dst + '.b', b.reshape(-1, 1) if col else b)
        n += 1
    put('ln_f.g', sd['norm.weight'])
    put('ln_f.b', sd['norm.bias'])
    put('head.w', sd['head.weight'], True)
    put('head.b', sd['head.bias'])
    size = px.shape[-1]
    cfg.update({'image': src, 'image_size': size, 'patch_size': k,
                'n_layer': n, 'n_embd': o, 'tokens': (size // k) ** 2,
                'classes': sd['head.bias'].shape[0], 'eps': 1e-6})
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
    ap.add_argument('--model', default='mixer_b16_224.goog_in21k_ft_in1k')
    a = ap.parse_args()
    cfg = {'source': a.model}
    write(a.outdir, cfg, hf_weights(a.model, cfg))


main()
