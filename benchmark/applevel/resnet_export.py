import argparse, io, json, os, struct, urllib.request

URL = 'http://images.cocodataset.org/val2017/000000039769.jpg'


def pixel_values(model):
    import timm, torch
    cfg = timm.data.resolve_data_config({}, model=model)
    try:
        from PIL import Image
        raw = urllib.request.urlopen(URL, timeout=30).read()
        img = Image.open(io.BytesIO(raw)).convert('RGB')
        tf = timm.data.create_transform(**cfg)
        return tf(img).unsqueeze(0), URL
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

    def conv(key, t):
        o, c, k, _ = t.shape
        put(key, t.permute(2, 3, 1, 0).reshape(k * k * c, o))

    def bn(key, p):
        put(key + '.g', sd[p + '.weight'])
        put(key + '.b', sd[p + '.bias'])
        put(key + '.m', sd[p + '.running_mean'])
        put(key + '.v', sd[p + '.running_var'])

    put('image', px[0].permute(1, 2, 0))
    conv('conv1.w', sd['conv1.weight'])
    bn('bn1', 'bn1')
    layers = []
    for li in range(1, 5):
        n = 0
        while 'layer%d.%d.conv1.weight' % (li, n) in sd:
            p = 'layer%d.%d.' % (li, n)
            key = 'l%d.%d.' % (li, n)
            conv(key + 'conv1.w', sd[p + 'conv1.weight'])
            bn(key + 'bn1', p + 'bn1')
            conv(key + 'conv2.w', sd[p + 'conv2.weight'])
            bn(key + 'bn2', p + 'bn2')
            down = p + 'downsample.0.weight' in sd
            if down:
                conv(key + 'down.w', sd[p + 'downsample.0.weight'])
                bn(key + 'bnd', p + 'downsample.1')
            layers.append([li, n, list(sd[p + 'conv1.weight'].shape),
                           int(sd[p + 'conv1.weight'].shape[2]),
                           2 if down or (li > 1 and n == 0) else 1, down])
            n += 1
    put('fc.w', sd['fc.weight'], True)
    put('fc.b', sd['fc.bias'])
    cfg.update({'image': src, 'image_size': px.shape[-1],
                'layers': layers, 'classes': sd['fc.bias'].shape[0],
                'eps': 1e-5,
                'stem': [list(sd['conv1.weight'].shape),
                         int(sd['conv1.weight'].shape[2])]})
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
    ap.add_argument('--model', default='resnet18')
    a = ap.parse_args()
    cfg = {'source': a.model}
    write(a.outdir, cfg, hf_weights(a.model, cfg))


main()
