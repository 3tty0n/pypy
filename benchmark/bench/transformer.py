from rpython.rlib import jit
from rpython.metatensor import device, nn
from rpython.metatensor.ops import tensor_sum, tensor_item
from bench.common import (ATT_D, ATT_H, MEM_ATTN_LIN, MEM_ATTN_SQ,
    MEM_BLOCK_LIN, MEM_BLOCK_SQ, MEM_TF_TRAIN_LIN, MEM_TF_TRAIN_SQ, TB_D,
    TB_EPS, TB_H, TF_BLOCKS, fit_rows, make_lr, make_mlp_input, zeros)

block_driver = jit.JitDriver(greens=[], reds='auto', is_recursive=True)
tf_driver = jit.JitDriver(greens=[], reds='auto', is_recursive=True)
attn_driver = jit.JitDriver(greens=[], reds='auto', is_recursive=True)

def tb_weight(rows, cols):
    w = zeros([rows, cols])
    for i in range(rows * cols):
        w.host[i] = float((i * 7) % 13 - 6) / TB_D
    device.dev(w)
    return nn.Tensor(w)

def tb_vector(v):
    t = zeros([TB_D])
    for i in range(TB_D):
        t.host[i] = v
    device.dev(t)
    return nn.Tensor(t)

def tb_qkv(d, h_count):
    dh = d // h_count
    w = zeros([d, d])
    for r in range(d):
        for h in range(h_count):
            for c in range(dh):
                w.host[r * d + h * dh + c] = float(
                    ((r * dh + c) * 7) % 13 - 6) / d
    device.dev(w)
    return nn.Tensor(w)

def tb_proj(d, h_count):
    dh = d // h_count
    w = zeros([d, d])
    for h in range(h_count):
        for r in range(dh):
            for c in range(d):
                w.host[(h * dh + r) * d + c] = float(
                    ((r * d + c) * 7) % 13 - 6) / d
    device.dev(w)
    return nn.Tensor(w)

def make_mlp_layers():
    layers = []
    for i in range(2):
        layers.append(nn.Linear(tb_weight(TB_D, TB_D), tb_vector(0.01)))
    return layers

def make_block():
    attn = nn.MultiHead(tb_qkv(TB_D, TB_H), tb_qkv(TB_D, TB_H),
                                tb_qkv(TB_D, TB_H), tb_proj(TB_D, TB_H), TB_H)
    return nn.TransformerBlock(attn, tb_vector(1.0), tb_vector(0.0),
                                       tb_vector(1.0), tb_vector(0.0),
                                       nn.MLP(make_mlp_layers()), TB_EPS)

def run_block(n, iters):
    rows = fit_rows(n, TB_D, MEM_BLOCK_SQ, MEM_BLOCK_LIN)
    block = make_block()
    x = make_mlp_input(rows, TB_D)
    i = 0
    while i < iters:
        block_driver.jit_merge_point()
        x = nn.Tensor(block.forward(x).t)
        i += 1
    return tensor_item(tensor_sum(x.t, -1))

def make_train_block():
    attn = nn.MultiHead(tb_qkv(TB_D, TB_H), tb_qkv(TB_D, TB_H),
                                tb_qkv(TB_D, TB_H), tb_proj(TB_D, TB_H),
                                TB_H, True)
    return nn.TransformerBlock(attn, tb_vector(1.0), tb_vector(0.0),
                                       tb_vector(1.0), tb_vector(0.0),
                                       nn.MLP(make_mlp_layers()), TB_EPS, True)

def make_transformer():
    blocks = []
    for i in range(TF_BLOCKS):
        blocks.append(make_train_block())
    head = nn.Linear(tb_weight(TB_D, TB_D), tb_vector(0.01))
    return nn.Transformer(blocks, head)

def run_transformer_train(n, iters):
    rows = fit_rows(n, TB_D, MEM_TF_TRAIN_SQ, MEM_TF_TRAIN_LIN)
    model = make_transformer()
    x = make_mlp_input(rows, TB_D)
    params = model.parameters()
    lr = make_lr()
    loss = 0.0
    i = 0
    while i < iters:
        tf_driver.jit_merge_point()
        out = model.forward(x).sum()
        out.backward()
        nn.sgd_step(params, lr)
        loss = out.item()
        i += 1
    return loss

def make_attn():
    return nn.MultiHead(tb_qkv(ATT_D, ATT_H), tb_qkv(ATT_D, ATT_H),
                                tb_qkv(ATT_D, ATT_H), tb_proj(ATT_D, ATT_H),
                                ATT_H)

def run_attn(n, iters):
    rows = fit_rows(n, ATT_D, MEM_ATTN_SQ, MEM_ATTN_LIN)
    attn = make_attn()
    x = make_mlp_input(rows, ATT_D)
    i = 0
    while i < iters:
        attn_driver.jit_merge_point()
        x = nn.Tensor(attn.forward(x).t)
        i += 1
    return tensor_item(tensor_sum(x.t, -1))
