from .nn import Module
from .functional import (layer_norm, rms_norm, softmax, gelu, gelu_erf,
                         silu, _scalar)


class MLP(Module):
    def __init__(self, layers):
        self.layers = layers

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class TransformerBlock(Module):
    def __init__(self, attn, gamma1, beta1, gamma2, beta2, mlp, eps=1e-5):
        self.attn = attn
        self.gamma1 = gamma1
        self.beta1 = beta1
        self.gamma2 = gamma2
        self.beta2 = beta2
        self.mlp = mlp
        self.eps = eps

    def forward(self, x):
        h = layer_norm(x, self.gamma1, self.beta1, self.eps)
        x = x.add(self.attn(h))
        return x.add(self.mlp(layer_norm(x, self.gamma2, self.beta2,
                                          self.eps)))


class CNN(Module):
    def __init__(self, conv, bn, pool, fc):
        self.conv = conv
        self.bn = bn
        self.pool = pool
        self.fc = fc

    def forward(self, x):
        y = self.bn(self.conv(x), self.pool.h * self.pool.w)
        return self.fc(self.pool(y.relu()))


class CausalSelfAttention(Module):
    def __init__(self, wqkv, bqkv, wo, bo, heads, mask):
        self.wqkv = wqkv
        self.bqkv = bqkv
        self.wo = wo
        self.bo = bo
        self.heads = heads
        self.mask = mask

    def forward(self, x):
        import math
        h = self.heads
        d = x.shape[1]
        dh = d // h
        qkv = x.matmul(self.wqkv).add(self.bqkv)
        s = qkv.attn_scores(qkv, h, d, 0, d).mul(
            _scalar(1.0 / math.sqrt(dh), x.dtype))
        if self.mask is not None:
            s = s.add(self.mask)
        c = softmax(s).attn_context(qkv, h, d, 2 * d)
        return c.matmul(self.wo).add(self.bo)


class GPT2MLP(Module):
    def __init__(self, wfc, bfc, wproj, bproj, act=None):
        self.wfc = wfc
        self.bfc = bfc
        self.wproj = wproj
        self.bproj = bproj
        self.act = act or gelu

    def forward(self, x):
        h = self.act(x.matmul(self.wfc).add(self.bfc))
        return h.matmul(self.wproj).add(self.bproj)


class GPT2Block(Module):
    def __init__(self, attn, g1, b1, g2, b2, mlp, eps=1e-5):
        self.attn = attn
        self.g1 = g1
        self.b1 = b1
        self.g2 = g2
        self.b2 = b2
        self.mlp = mlp
        self.eps = eps

    def forward(self, x):
        x = x.add(self.attn(layer_norm(x, self.g1, self.b1, self.eps)))
        return x.add(self.mlp(layer_norm(x, self.g2, self.b2, self.eps)))


class GPT2(Module):
    def __init__(self, wte, blocks, gf, bf, eps=1e-5):
        self.wte = wte
        self.blocks = blocks
        self.gf = gf
        self.bf = bf
        self.eps = eps

    def forward(self, idx, pos):
        x = self.wte.take(idx).add(pos)
        for block in self.blocks:
            x = block(x)
        x = layer_norm(x, self.gf, self.bf, self.eps)
        return x.matmul(self.wte, True)


class LlamaAttention(Module):
    def __init__(self, wqkv, wo, heads, mask, cos, sin, head_dim):
        self.wqkv = wqkv
        self.wo = wo
        self.heads = heads
        self.mask = mask
        self.cos = cos
        self.sin = sin
        self.head_dim = head_dim

    def forward(self, x):
        import math
        h = self.heads
        d = x.shape[1]
        dh = d // h
        qkv = x.matmul(self.wqkv)
        qkv = qkv.mul(self.cos).add(qkv.rot_half(self.head_dim).mul(self.sin))
        s = qkv.attn_scores(qkv, h, d, 0, d).mul(
            _scalar(1.0 / math.sqrt(dh), x.dtype)).add(self.mask)
        c = softmax(s).attn_context(qkv, h, d, 2 * d)
        return c.matmul(self.wo)


class LlamaMLP(Module):
    def __init__(self, wgate, wup, wdown):
        self.wgate = wgate
        self.wup = wup
        self.wdown = wdown

    def forward(self, x):
        return silu(x.matmul(self.wgate)).mul(x.matmul(self.wup)).matmul(
            self.wdown)


class LlamaBlock(Module):
    def __init__(self, attn, g1, g2, mlp, eps=1e-5):
        self.attn = attn
        self.g1 = g1
        self.g2 = g2
        self.mlp = mlp
        self.eps = eps

    def forward(self, x):
        x = x.add(self.attn(rms_norm(x, self.g1, self.eps)))
        return x.add(self.mlp(rms_norm(x, self.g2, self.eps)))


class Llama(Module):
    def __init__(self, wte, blocks, gf, head=None, eps=1e-5):
        self.wte = wte
        self.blocks = blocks
        self.gf = gf
        self.head = head
        self.eps = eps

    def forward(self, idx):
        x = self.wte.take(idx)
        for block in self.blocks:
            x = block(x)
        x = rms_norm(x, self.gf, self.eps)
        if self.head is None:
            return x.matmul(self.wte, True)
        return x.matmul(self.head, True)


class BertBlock(Module):
    def __init__(self, attn, g1, b1, mlp, g2, b2, eps=1e-12):
        self.attn = attn
        self.g1 = g1
        self.b1 = b1
        self.mlp = mlp
        self.g2 = g2
        self.b2 = b2
        self.eps = eps

    def forward(self, x):
        x = layer_norm(x.add(self.attn(x)), self.g1, self.b1, self.eps)
        return layer_norm(x.add(self.mlp(x)), self.g2, self.b2, self.eps)


class Bert(Module):
    def __init__(self, wte, emb, ge, be, blocks, wd, bd, gm, bm, bias,
                 eps=1e-12):
        self.wte = wte
        self.emb = emb
        self.ge = ge
        self.be = be
        self.blocks = blocks
        self.wd = wd
        self.bd = bd
        self.gm = gm
        self.bm = bm
        self.bias = bias
        self.eps = eps

    def forward(self, idx):
        x = layer_norm(self.wte.take(idx).add(self.emb), self.ge, self.be,
                       self.eps)
        for block in self.blocks:
            x = block(x)
        h = gelu_erf(x.matmul(self.wd).add(self.bd))
        h = layer_norm(h, self.gm, self.bm, self.eps)
        return h.matmul(self.wte, True).add(self.bias)


class ViT(Module):
    def __init__(self, idx, tokens, patch, wp, emb, blocks, gf, bf, head,
                 bhead, cls, eps=1e-12):
        self.idx = idx
        self.tokens = tokens
        self.patch = patch
        self.wp = wp
        self.emb = emb
        self.blocks = blocks
        self.gf = gf
        self.bf = bf
        self.head = head
        self.bhead = bhead
        self.cls = cls
        self.eps = eps

    def forward(self, img):
        x = img.take(self.idx).reshape([self.tokens, self.patch])
        x = x.matmul(self.wp).add(self.emb)
        for block in self.blocks:
            x = block(x)
        x = layer_norm(x, self.gf, self.bf, self.eps)
        return x.take(self.cls).matmul(self.head).add(self.bhead)


class ResNetBlock(Module):
    def __init__(self, conv1, bn1, conv2, bn2, down=None, bnd=None):
        self.conv1 = conv1
        self.bn1 = bn1
        self.conv2 = conv2
        self.bn2 = bn2
        self.down = down
        self.bnd = bnd

    def forward(self, x):
        y = self.bn1(self.conv1(x), self.conv1.oh * self.conv1.ow).relu()
        y = self.bn2(self.conv2(y), self.conv2.oh * self.conv2.ow)
        if self.down is not None:
            x = self.bnd(self.down(x), self.down.oh * self.down.ow)
        return y.add(x).relu()


class ResNet(Module):
    def __init__(self, conv1, bn1, pool, blocks, channels, hw, fcw, fcb,
                 mean=None):
        self.mean = mean
        self.conv1 = conv1
        self.bn1 = bn1
        self.pool = pool
        self.blocks = blocks
        self.channels = channels
        self.hw = hw
        self.fcw = fcw
        self.fcb = fcb

    def forward(self, x):
        y = self.bn1(self.conv1(x), self.conv1.oh * self.conv1.ow).relu()
        y = self.pool(y)
        for block in self.blocks:
            y = block(y)
        if self.mean is not None:
            return self.mean.matmul(y).matmul(self.fcw).add(self.fcb)
        n = y.size // (self.channels * self.hw)
        y = y.reshape([n * self.channels, self.hw]).sum(1)
        y = y.reshape([n, self.channels]).mul(
            _scalar(1.0 / self.hw, y.dtype))
        return y.matmul(self.fcw).add(self.fcb)


class MixerBlock(Module):
    def __init__(self, g1, b1, wt1, bt1, wt2, bt2, g2, b2, mlp, eps=1e-6):
        self.g1 = g1
        self.b1 = b1
        self.wt1 = wt1
        self.bt1 = bt1
        self.wt2 = wt2
        self.bt2 = bt2
        self.g2 = g2
        self.b2 = b2
        self.mlp = mlp
        self.eps = eps

    def forward(self, x):
        h = layer_norm(x, self.g1, self.b1, self.eps)
        u = gelu_erf(self.wt1.matmul(h).add(self.bt1))
        x = x.add(self.wt2.matmul(u).add(self.bt2))
        return x.add(self.mlp(layer_norm(x, self.g2, self.b2, self.eps)))


class Mixer(Module):
    def __init__(self, size, patch, wp, bp, blocks, gf, bf, mean, head,
                 bhead, eps=1e-6):
        self.size = size
        self.patch = patch
        self.wp = wp
        self.bp = bp
        self.blocks = blocks
        self.gf = gf
        self.bf = bf
        self.mean = mean
        self.head = head
        self.bhead = bhead
        self.eps = eps

    def forward(self, img):
        k = self.patch
        x = img.im2col(3, self.size, self.size, k, 0, k)
        x = x.matmul(self.wp).add(self.bp)
        for block in self.blocks:
            x = block(x)
        x = layer_norm(x, self.gf, self.bf, self.eps)
        return self.mean.matmul(x).matmul(self.head).add(self.bhead)
