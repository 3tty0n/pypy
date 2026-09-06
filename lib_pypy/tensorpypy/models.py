from .nn import Module
from .functional import layer_norm, rms_norm, softmax, gelu, silu, _scalar


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
            _scalar(1.0 / math.sqrt(dh), x.dtype)).add(self.mask)
        c = softmax(s).attn_context(qkv, h, d, 2 * d)
        return c.matmul(self.wo).add(self.bo)


class GPT2MLP(Module):
    def __init__(self, wfc, bfc, wproj, bproj):
        self.wfc = wfc
        self.bfc = bfc
        self.wproj = wproj
        self.bproj = bproj

    def forward(self, x):
        h = gelu(x.matmul(self.wfc).add(self.bfc))
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
