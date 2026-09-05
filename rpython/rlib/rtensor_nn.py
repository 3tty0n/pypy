import math
from rpython.rlib import rtensor, jit


class Tensor(object):
    def __init__(self, t, requires_grad=False):
        self.t = t
        self.requires_grad = requires_grad
        self.grad = None
        self.node = None
        self.acc = None
        self.seen = False

    def _wrap(self, t, node, needs):
        r = Tensor(t)
        if needs:
            r.requires_grad = True
            r.node = node
        return r

    def add(self, other, p=-1):
        if p < 0:
            p = rtensor.bcast(self.t, other.t)
        needs = self.requires_grad or other.requires_grad
        node = None
        if needs:
            node = AddNode(self, other, p)
        return self._wrap(rtensor.tensor_add(self.t, other.t, p), node, needs)

    def mul(self, other, p=-1):
        if p < 0:
            p = rtensor.bcast(self.t, other.t)
        needs = self.requires_grad or other.requires_grad
        node = None
        if needs:
            node = MulNode(self, other, p)
        return self._wrap(rtensor.tensor_mul(self.t, other.t, p), node, needs)

    def relu(self):
        r = Tensor(rtensor.relu(self.t))
        if self.requires_grad:
            r.requires_grad = True
            r.node = ReluNode(self, r)
        return r

    def sum(self, axis=rtensor.AXIS_ALL):
        node = None
        if self.requires_grad:
            node = SumNode(self, axis)
        return self._wrap(rtensor.tensor_sum(self.t, axis), node,
                          self.requires_grad)

    def item(self):
        return rtensor.item(self.t)

    def _forward_only(self, t, other):
        r = Tensor(t)
        needs = self.requires_grad
        inputs = [self]
        if other is not None:
            inputs.append(other)
            needs = needs or other.requires_grad
        if needs:
            r.requires_grad = True
            r.node = NoGradNode(inputs)
        return r

    def sub(self, other, p=-1):
        if p < 0:
            p = rtensor.bcast(self.t, other.t)
        needs = self.requires_grad or other.requires_grad
        node = None
        if needs:
            node = SubNode(self, other, p)
        return self._wrap(rtensor.tensor_sub(self.t, other.t, p), node, needs)

    def div(self, other, p=-1):
        if p < 0:
            p = rtensor.bcast(self.t, other.t)
        needs = self.requires_grad or other.requires_grad
        r = self._wrap(rtensor.tensor_div(self.t, other.t, p), None, needs)
        if needs:
            r.node = DivNode(self, other, p, r)
        return r

    def exp(self):
        r = Tensor(rtensor.tensor_exp(self.t))
        if self.requires_grad:
            r.requires_grad = True
            r.node = ExpNode(self, r)
        return r

    def sqrt(self):
        r = Tensor(rtensor.tensor_sqrt(self.t))
        if self.requires_grad:
            r.requires_grad = True
            r.node = SqrtNode(self, r)
        return r

    def max(self, axis=rtensor.AXIS_ALL):
        r = Tensor(rtensor.tensor_maxr(self.t, axis))
        if self.requires_grad:
            r.requires_grad = True
            if axis == 1:
                r.node = MaxNode(self, r)
            else:
                inputs = [self]
                r.node = NoGradNode(inputs)
        return r

    def _matmul(self, other, rows, cols, inner, tb):
        needs = self.requires_grad or other.requires_grad
        node = None
        if needs:
            node = MatmulNode(self, other, rows, cols, inner, tb)
        return self._wrap(
            rtensor.tensor_matmul(self.t, other.t, rows, cols, inner, 0, tb),
            node, needs)

    def matmul(self, other, transpose_b=False):
        if transpose_b:
            rows, cols, inner = rtensor.matmul_shape_t(self.t, other.t)
            return self._matmul(other, rows, cols, inner, 1)
        rows, cols, inner = rtensor.matmul_shape(self.t, other.t)
        return self._matmul(other, rows, cols, inner, 0)

    def bmm(self, other, batch, rows, cols, inner, tb):
        needs = self.requires_grad or other.requires_grad
        node = None
        if needs:
            node = BmmNode(self, other, batch, rows, cols, inner, tb)
        return self._wrap(
            rtensor.tensor_bmm(self.t, other.t, batch, rows, cols, inner,
                               0, tb),
            node, needs)

    def head_split(self, rows, dh, heads):
        node = None
        if self.requires_grad:
            node = HeadSplitNode(self, rows, dh, heads)
        return self._wrap(rtensor.head_split(self.t, rows, dh, heads), node,
                          self.requires_grad)

    def head_merge(self, rows, dh, heads):
        node = None
        if self.requires_grad:
            node = HeadMergeNode(self, rows, dh, heads)
        return self._wrap(rtensor.head_merge(self.t, rows, dh, heads), node,
                          self.requires_grad)

    @jit.unroll_safe
    def reshape(self, shape_list):
        node = None
        if self.requires_grad:
            old = []
            for i in range(rtensor.tensor_ndim(self.t)):
                old.append(rtensor.tensor_shape(self.t, i))
            node = ReshapeNode(self, old)
        return self._wrap(rtensor.reshape(self.t, shape_list), node,
                          self.requires_grad)

    def add_(self, other):
        return self.add(other)

    def mul_(self, other):
        return self.mul(other)

    def shape(self, axis):
        return rtensor.tensor_shape(self.t, axis)

    def size(self):
        return rtensor.tensor_size(self.t)

    @jit.unroll_safe
    def backward(self):
        order = []
        _topo(self, order)
        self.acc = Tensor(rtensor.ones_like(self.t))
        i = len(order) - 1
        while i >= 0:
            t = order[i]
            i -= 1
            t.seen = False
            g = t.acc
            t.acc = None
            if g is None:
                continue
            node = t.node
            if node is None:
                if t.requires_grad:
                    t.grad = _accumulate(t.grad, g)
                continue
            grads = node.apply(g)
            inputs = node.inputs
            for j in range(len(grads)):
                gj = grads[j]
                if gj is not None:
                    inputs[j].acc = _accumulate(inputs[j].acc, gj)


def _accumulate(old, g):
    if old is None:
        return g
    return Tensor(rtensor.add(old.t, g.t))


@jit.unroll_safe
def _topo(root, order):
    root.seen = True
    stack = [root]
    todo = [0]
    while len(stack) > 0:
        top = len(stack) - 1
        t = stack[top]
        node = t.node
        k = todo[top]
        if node is not None and k < len(node.inputs):
            todo[top] = k + 1
            nxt = node.inputs[k]
            if not nxt.seen:
                nxt.seen = True
                stack.append(nxt)
                todo.append(0)
            continue
        stack.pop()
        todo.pop()
        order.append(t)


def _unbroadcast(g, p, right):
    if right:
        row = rtensor.BC_R_ROW
        scalar = rtensor.BC_R_SCALAR
        col = rtensor.BC_R_COL
    else:
        row = rtensor.BC_L_ROW
        scalar = rtensor.BC_L_SCALAR
        col = rtensor.BC_L_COL
    if p == row:
        return Tensor(rtensor.sum(g.t, 0))
    if p == scalar:
        return Tensor(rtensor.sum(g.t, rtensor.AXIS_ALL))
    if p == col:
        return Tensor(rtensor.sum(g.t, 1))
    return g


def _rp(p):
    if (p == rtensor.BC_L_ROW or p == rtensor.BC_L_SCALAR or
            p == rtensor.BC_L_COL):
        return rtensor.BC_NONE
    return p


def _lp(p):
    if p == rtensor.BC_L_ROW:
        return rtensor.BC_R_ROW
    if p == rtensor.BC_L_SCALAR:
        return rtensor.BC_R_SCALAR
    if p == rtensor.BC_L_COL:
        return rtensor.BC_R_COL
    return rtensor.BC_NONE


def _neg(t):
    return rtensor.tensor_mul(t, rtensor.scalar(-1.0), rtensor.BC_R_SCALAR)


class Node(object):
    def __init__(self, inputs):
        self.inputs = inputs

    def apply(self, g):
        raise NotImplementedError


class NoGradNode(Node):
    def apply(self, g):
        raise ValueError("no backward for this operation")


class AddNode(Node):
    def __init__(self, a, b, p):
        inputs = [a]
        inputs.append(b)
        Node.__init__(self, inputs)
        self.p = p

    def apply(self, g):
        ga = None
        if self.inputs[0].requires_grad:
            ga = _unbroadcast(g, self.p, False)
        gb = None
        if self.inputs[1].requires_grad:
            gb = _unbroadcast(g, self.p, True)
        grads = [ga]
        grads.append(gb)
        return grads


class MulNode(Node):
    def __init__(self, a, b, p):
        inputs = [a]
        inputs.append(b)
        Node.__init__(self, inputs)
        self.p = p

    def apply(self, g):
        a = self.inputs[0]
        b = self.inputs[1]
        ga = None
        if a.requires_grad:
            ga = _unbroadcast(
                Tensor(rtensor.tensor_mul(g.t, b.t, _rp(self.p))),
                self.p, False)
        gb = None
        if b.requires_grad:
            gb = _unbroadcast(
                Tensor(rtensor.tensor_mul(g.t, a.t, _lp(self.p))),
                self.p, True)
        grads = [ga]
        grads.append(gb)
        return grads


class ReluNode(Node):
    def __init__(self, x, y):
        inputs = [x]
        Node.__init__(self, inputs)
        self.y = y

    def apply(self, g):
        return [Tensor(rtensor.relugrad(self.y.t, g.t))]


class SumNode(Node):
    def __init__(self, x, axis):
        inputs = [x]
        Node.__init__(self, inputs)
        self.axis = axis

    def apply(self, g):
        x = self.inputs[0]
        if self.axis == 1:
            rtensor.cols_of(x.t)
            p = rtensor.BC_R_COL
        elif self.axis == 0:
            p = rtensor.BC_R_ROW
        else:
            p = rtensor.BC_R_SCALAR
        return [Tensor(rtensor.tensor_mul(rtensor.ones_like(x.t), g.t, p))]


class MatmulNode(Node):
    def __init__(self, x, w, rows, cols, inner, tb):
        inputs = [x]
        inputs.append(w)
        Node.__init__(self, inputs)
        self.rows = rows
        self.cols = cols
        self.inner = inner
        self.tb = tb

    def apply(self, g):
        x = self.inputs[0]
        w = self.inputs[1]
        gx = None
        gw = None
        if self.tb:
            if x.requires_grad:
                gx = Tensor(rtensor.tensor_matmul(
                    g.t, w.t, self.rows, self.inner, self.cols, 0, 0))
            if w.requires_grad:
                gw = Tensor(rtensor.tensor_matmul(
                    g.t, x.t, self.cols, self.inner, self.rows, 1, 0))
        else:
            if x.requires_grad:
                gx = Tensor(rtensor.tensor_matmul(
                    g.t, w.t, self.rows, self.inner, self.cols, 0, 1))
            if w.requires_grad:
                gw = Tensor(rtensor.tensor_matmul(
                    x.t, g.t, self.inner, self.cols, self.rows, 1, 0))
        grads = [gx]
        grads.append(gw)
        return grads


class SubNode(Node):
    def __init__(self, a, b, p):
        inputs = [a]
        inputs.append(b)
        Node.__init__(self, inputs)
        self.p = p

    def apply(self, g):
        ga = None
        if self.inputs[0].requires_grad:
            ga = _unbroadcast(g, self.p, False)
        gb = None
        if self.inputs[1].requires_grad:
            gb = _unbroadcast(Tensor(_neg(g.t)), self.p, True)
        grads = [ga]
        grads.append(gb)
        return grads


class DivNode(Node):
    def __init__(self, a, b, p, y):
        inputs = [a]
        inputs.append(b)
        Node.__init__(self, inputs)
        self.p = p
        self.y = y

    def apply(self, g):
        a = self.inputs[0]
        b = self.inputs[1]
        ga = None
        if a.requires_grad:
            ga = _unbroadcast(
                Tensor(rtensor.tensor_div(g.t, b.t, _rp(self.p))),
                self.p, False)
        gb = None
        if b.requires_grad:
            t = rtensor.tensor_mul(self.y.t, g.t, rtensor.BC_NONE)
            t = rtensor.tensor_div(t, b.t, _rp(self.p))
            gb = _unbroadcast(Tensor(_neg(t)), self.p, True)
        grads = [ga]
        grads.append(gb)
        return grads


class ExpNode(Node):
    def __init__(self, x, y):
        inputs = [x]
        Node.__init__(self, inputs)
        self.y = y

    def apply(self, g):
        return [Tensor(rtensor.tensor_mul(g.t, self.y.t, rtensor.BC_NONE))]


class SqrtNode(Node):
    def __init__(self, x, y):
        inputs = [x]
        Node.__init__(self, inputs)
        self.y = y

    def apply(self, g):
        h = rtensor.tensor_mul(g.t, rtensor.scalar(0.5), rtensor.BC_R_SCALAR)
        return [Tensor(rtensor.tensor_div(h, self.y.t, rtensor.BC_NONE))]


class MaxNode(Node):
    def __init__(self, x, m):
        inputs = [x]
        Node.__init__(self, inputs)
        self.m = m

    def apply(self, g):
        x = self.inputs[0]
        rtensor.cols_of(x.t)
        mask = rtensor.tensor_eqmask(x.t, self.m.t, rtensor.BC_R_COL)
        return [Tensor(rtensor.tensor_mul(mask, g.t, rtensor.BC_R_COL))]


class BmmNode(Node):
    def __init__(self, a, b, batch, rows, cols, inner, tb):
        inputs = [a]
        inputs.append(b)
        Node.__init__(self, inputs)
        self.batch = batch
        self.rows = rows
        self.cols = cols
        self.inner = inner
        self.tb = tb

    def apply(self, g):
        a = self.inputs[0]
        b = self.inputs[1]
        ga = None
        gb = None
        if self.tb:
            if a.requires_grad:
                ga = Tensor(rtensor.tensor_bmm(
                    g.t, b.t, self.batch, self.rows, self.inner, self.cols,
                    0, 0))
            if b.requires_grad:
                gb = Tensor(rtensor.tensor_bmm(
                    g.t, a.t, self.batch, self.cols, self.inner, self.rows,
                    1, 0))
        else:
            if a.requires_grad:
                ga = Tensor(rtensor.tensor_bmm(
                    g.t, b.t, self.batch, self.rows, self.inner, self.cols,
                    0, 1))
            if b.requires_grad:
                gb = Tensor(rtensor.tensor_bmm(
                    a.t, g.t, self.batch, self.inner, self.cols, self.rows,
                    1, 0))
        grads = [ga]
        grads.append(gb)
        return grads


class HeadSplitNode(Node):
    def __init__(self, x, rows, dh, heads):
        inputs = [x]
        Node.__init__(self, inputs)
        self.rows = rows
        self.dh = dh
        self.heads = heads

    def apply(self, g):
        return [Tensor(rtensor.head_merge(g.t, self.rows, self.dh,
                                          self.heads))]


class HeadMergeNode(Node):
    def __init__(self, x, rows, dh, heads):
        inputs = [x]
        Node.__init__(self, inputs)
        self.rows = rows
        self.dh = dh
        self.heads = heads

    def apply(self, g):
        return [Tensor(rtensor.head_split(g.t, self.rows, self.dh,
                                          self.heads))]


class ReshapeNode(Node):
    def __init__(self, x, shape_list):
        inputs = [x]
        Node.__init__(self, inputs)
        self.shape_list = shape_list

    def apply(self, g):
        return [Tensor(rtensor.reshape(g.t, self.shape_list))]


@jit.unroll_safe
def _extend(params, more):
    for i in range(len(more)):
        params.append(more[i])


class Linear(object):
    def __init__(self, weight, bias):
        self.weight = weight
        self.bias = bias
        weight.requires_grad = True
        bias.requires_grad = True

    def forward(self, x):
        return x.matmul(self.weight).add(self.bias).relu()

    def parameters(self):
        params = [self.weight]
        params.append(self.bias)
        return params


class MLP(object):
    def __init__(self, layers):
        self.layers = layers

    @jit.unroll_safe
    def forward(self, x):
        for i in range(len(self.layers)):
            x = self.layers[i].forward(x)
        return x

    @jit.unroll_safe
    def parameters(self):
        params = []
        for i in range(len(self.layers)):
            params.append(self.layers[i].weight)
            params.append(self.layers[i].bias)
        return params


@jit.unroll_safe
def sgd_step(params, neg_lr):
    for i in range(len(params)):
        p = params[i]
        g = p.grad
        if g is not None:
            p.t = rtensor.add(p.t, rtensor.mul(g.t, neg_lr.t))
            p.grad = None


def softmax(x):
    rtensor.cols_of(x.t)
    m = x.max(1)
    e = x.sub(m, rtensor.BC_R_COL).exp()
    return e.div(e.sum(1), rtensor.BC_R_COL)


def layernorm(x, gamma, beta, eps):
    c = rtensor.cols_of(x.t)
    inv = Tensor(rtensor.scalar(1.0 / c))
    epst = Tensor(rtensor.scalar(eps))
    mean = x.sum(1).mul(inv, rtensor.BC_R_SCALAR)
    d = x.sub(mean, rtensor.BC_R_COL)
    var = d.mul(d, rtensor.BC_NONE).sum(1).mul(inv, rtensor.BC_R_SCALAR)
    denom = var.add(epst, rtensor.BC_R_SCALAR).sqrt()
    y = d.div(denom, rtensor.BC_R_COL)
    return y.mul(gamma, rtensor.BC_R_ROW).add(beta, rtensor.BC_R_ROW)


def mha(x, wq, wk, wv, wo, heads):
    rows = rtensor.tensor_shape(x.t, 0)
    d = rtensor.tensor_shape(x.t, 1)
    dh = d // heads
    q = x._matmul(wq, rows, d, d, 0).head_split(rows, dh, heads)
    k = x._matmul(wk, rows, d, d, 0).head_split(rows, dh, heads)
    v = x._matmul(wv, rows, d, d, 0).head_split(rows, dh, heads)
    scores = q.bmm(k, heads, rows, rows, dh, 1)
    scale = Tensor(rtensor.scalar(1.0 / math.sqrt(dh)))
    scaled = scores.mul(scale, rtensor.BC_R_SCALAR)
    ctx = softmax(scaled).bmm(v, heads, rows, dh, rows, 0)
    return ctx.head_merge(rows, dh, heads)._matmul(wo, rows, d, d, 0)


class MultiHead(object):
    def __init__(self, wq, wk, wv, wo, heads, requires_grad=False):
        self.wq = wq
        self.wk = wk
        self.wv = wv
        self.wo = wo
        self.heads = heads
        if requires_grad:
            wq.requires_grad = True
            wk.requires_grad = True
            wv.requires_grad = True
            wo.requires_grad = True

    def forward(self, x):
        return mha(x, self.wq, self.wk, self.wv, self.wo, self.heads)

    def parameters(self):
        params = [self.wq]
        params.append(self.wk)
        params.append(self.wv)
        params.append(self.wo)
        return params


class TransformerBlock(object):
    def __init__(self, attn, gamma1, beta1, gamma2, beta2, mlp, eps,
                 requires_grad=False):
        self.attn = attn
        self.gamma1 = gamma1
        self.beta1 = beta1
        self.gamma2 = gamma2
        self.beta2 = beta2
        self.mlp = mlp
        self.eps = eps
        if requires_grad:
            gamma1.requires_grad = True
            beta1.requires_grad = True
            gamma2.requires_grad = True
            beta2.requires_grad = True

    @jit.unroll_safe
    def forward(self, x):
        h = layernorm(x, self.gamma1, self.beta1, self.eps)
        x = x.add(self.attn.forward(h))
        h2 = layernorm(x, self.gamma2, self.beta2, self.eps)
        return x.add(self.mlp.forward(h2))

    @jit.unroll_safe
    def parameters(self):
        params = []
        _extend(params, self.attn.parameters())
        params.append(self.gamma1)
        params.append(self.beta1)
        params.append(self.gamma2)
        params.append(self.beta2)
        _extend(params, self.mlp.parameters())
        return params


class Transformer(object):
    def __init__(self, blocks, head):
        self.blocks = blocks
        self.head = head

    @jit.unroll_safe
    def forward(self, x):
        for i in range(len(self.blocks)):
            x = self.blocks[i].forward(x)
        return self.head.forward(x)

    @jit.unroll_safe
    def parameters(self):
        params = []
        for i in range(len(self.blocks)):
            _extend(params, self.blocks[i].parameters())
        _extend(params, self.head.parameters())
        return params


class Conv2d(object):
    def __init__(self, weight, bias, c, h, w, o):
        self.weight = weight
        self.bias = bias
        self.c = c
        self.h = h
        self.w = w
        self.o = o

    def forward(self, x):
        hw = self.h * self.w
        rows = rtensor.tensor_size(x.t) // (self.c * hw)
        cols = rtensor.im2col(x.t, self.c, self.h, self.w, 3, 1)
        y = rtensor.tensor_matmul(cols, self.weight.t, rows * hw, self.o,
                                  self.c * 9, 0, 0)
        y = rtensor.tensor_add(y, self.bias.t, rtensor.BC_R_ROW)
        return x._forward_only(rtensor.col2chw(y, rows, hw, self.o),
                               self.weight)


class BatchNorm2d(object):
    def __init__(self, c, eps=1e-05):
        self.c = c
        self.eps = eps
        self.gamma = [1.0] * c
        self.beta = [0.0] * c
        self.mean = [0.0] * c
        self.var = [1.0] * c
        self.rows = -1
        self.scale = rtensor.NULLTENSOR
        self.shift = rtensor.NULLTENSOR

    def set_stats(self, gamma, beta, mean, var):
        self.gamma = gamma
        self.beta = beta
        self.mean = mean
        self.var = var
        self.rows = -1

    def _prepare(self, rows):
        a = rtensor.column(rows)
        b = rtensor.column(rows)
        for i in range(rows):
            ci = i % self.c
            g = self.gamma[ci] / math.sqrt(self.var[ci] + self.eps)
            a.host[i] = g
            b.host[i] = self.beta[ci] - self.mean[ci] * g
        rtensor.dev(a)
        rtensor.dev(b)
        self.scale = a
        self.shift = b
        self.rows = rows

    def forward(self, x, hw):
        rows = rtensor.tensor_size(x.t) // hw
        if self.rows != rows:
            self._prepare(rows)
        v = rtensor.view2(x.t, rows, hw)
        y = rtensor.tensor_mul(v, self.scale, rtensor.BC_R_COL)
        return x._forward_only(
            rtensor.tensor_add(y, self.shift, rtensor.BC_R_COL), None)


class MaxPool2d(object):
    def __init__(self, c, h, w):
        self.c = c
        self.h = h
        self.w = w

    def forward(self, x):
        return x._forward_only(rtensor.maxpool2(x.t, self.c, self.h, self.w),
                               None)


class CNN(object):
    def __init__(self, conv, bn, pool, fc):
        self.conv = conv
        self.bn = bn
        self.pool = pool
        self.fc = fc

    @jit.unroll_safe
    def forward(self, x):
        y = self.conv.forward(x)
        y = self.bn.forward(y, self.pool.h * self.pool.w)
        y = self.pool.forward(y.relu())
        cols = self.pool.c * (self.pool.h // 2) * (self.pool.w // 2)
        shape = [rtensor.tensor_size(y.t) // cols]
        shape.append(cols)
        return self.fc.forward(y.reshape(shape))
