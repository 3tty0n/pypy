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

    def add(self, other):
        p = rtensor.bcast(self.t, other.t)
        needs = self.requires_grad or other.requires_grad
        node = None
        if needs:
            node = AddNode(self, other, p)
        return self._wrap(rtensor.tensor_add(self.t, other.t, p), node, needs)

    def mul(self, other):
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
        return self._wrap(rtensor.sum(self.t, axis), node, self.requires_grad)

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

    def sub(self, other):
        p = rtensor.bcast(self.t, other.t)
        return self._forward_only(rtensor.tensor_sub(self.t, other.t, p), other)

    def div(self, other):
        p = rtensor.bcast(self.t, other.t)
        return self._forward_only(rtensor.tensor_div(self.t, other.t, p), other)

    def exp(self):
        return self._forward_only(rtensor.tensor_exp(self.t), None)

    def sqrt(self):
        return self._forward_only(rtensor.tensor_sqrt(self.t), None)

    def max(self, axis=rtensor.AXIS_ALL):
        return self._forward_only(rtensor.tensor_maxr(self.t, axis), None)

    def matmul(self, other, transpose_b=False):
        if transpose_b:
            rows, cols, inner = rtensor.matmul_shape_t(self.t, other.t)
            return self._forward_only(
                rtensor.tensor_matmul(self.t, other.t, rows, cols, inner, 0, 1),
                other)
        rows, cols, inner = rtensor.matmul_shape(self.t, other.t)
        needs = self.requires_grad or other.requires_grad
        node = None
        if needs:
            node = MatmulNode(self, other, rows, cols, inner)
        return self._wrap(
            rtensor.tensor_matmul(self.t, other.t, rows, cols, inner, 0, 0),
            node, needs)

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
        row, scalar = rtensor.BC_R_ROW, rtensor.BC_R_SCALAR
    else:
        row, scalar = rtensor.BC_L_ROW, rtensor.BC_L_SCALAR
    if p == row:
        return Tensor(rtensor.sum(g.t, 0))
    if p == scalar:
        return Tensor(rtensor.sum(g.t, rtensor.AXIS_ALL))
    return g


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
            ga = _unbroadcast(Tensor(rtensor.mul(g.t, b.t)), self.p, False)
        gb = None
        if b.requires_grad:
            gb = _unbroadcast(Tensor(rtensor.mul(g.t, a.t)), self.p, True)
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
        if self.axis == 1:
            raise ValueError("no backward for sum(axis=1)")
        x = self.inputs[0]
        return [Tensor(rtensor.mul(rtensor.ones_like(x.t), g.t))]


class MatmulNode(Node):
    def __init__(self, x, w, rows, cols, inner):
        inputs = [x]
        inputs.append(w)
        Node.__init__(self, inputs)
        self.rows = rows
        self.cols = cols
        self.inner = inner

    def apply(self, g):
        x = self.inputs[0]
        w = self.inputs[1]
        gx = None
        if x.requires_grad:
            gx = Tensor(rtensor.tensor_matmul(g.t, w.t, self.rows, self.inner,
                                              self.cols, 0, 1))
        gw = None
        if w.requires_grad:
            gw = Tensor(rtensor.tensor_matmul(x.t, g.t, self.inner, self.cols,
                                              self.rows, 1, 0))
        grads = [gx]
        grads.append(gw)
        return grads


class ReshapeNode(Node):
    def __init__(self, x, shape_list):
        inputs = [x]
        Node.__init__(self, inputs)
        self.shape_list = shape_list

    def apply(self, g):
        return [Tensor(rtensor.reshape(g.t, self.shape_list))]


class Linear(object):
    def __init__(self, weight, bias):
        self.weight = weight
        self.bias = bias
        weight.requires_grad = True
        bias.requires_grad = True

    def forward(self, x):
        return x.matmul(self.weight).add(self.bias).relu()


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
    m = rtensor.tensor_maxr(x.t, 1)
    e = rtensor.tensor_exp(rtensor.tensor_sub(x.t, m, rtensor.BC_R_COL))
    s = rtensor.tensor_sum(e, 1)
    return x._forward_only(rtensor.tensor_div(e, s, rtensor.BC_R_COL), None)


def layernorm(x, gamma, beta, eps):
    c = rtensor.tensor_shape(x.t, 1)
    inv = rtensor.from_list([1.0 / c])
    epst = rtensor.from_list([eps])
    mean = rtensor.tensor_mul(rtensor.tensor_sum(x.t, 1), inv,
                              rtensor.BC_R_SCALAR)
    d = rtensor.tensor_sub(x.t, mean, rtensor.BC_R_COL)
    sq = rtensor.tensor_mul(d, d, rtensor.BC_NONE)
    var = rtensor.tensor_mul(rtensor.tensor_sum(sq, 1), inv,
                             rtensor.BC_R_SCALAR)
    denom = rtensor.tensor_sqrt(rtensor.tensor_add(var, epst,
                                                   rtensor.BC_R_SCALAR))
    y = rtensor.tensor_div(d, denom, rtensor.BC_R_COL)
    y = rtensor.tensor_mul(y, gamma.t, rtensor.BC_R_ROW)
    y = rtensor.tensor_add(y, beta.t, rtensor.BC_R_ROW)
    return x._forward_only(y, gamma)


def attention(q, k, v):
    d = rtensor.tensor_shape(q.t, 1)
    scale = rtensor.from_list([1.0 / math.sqrt(d)])
    scores = q.matmul(k, True)
    scaled = q._forward_only(
        rtensor.tensor_mul(scores.t, scale, rtensor.BC_R_SCALAR), None)
    return softmax(scaled).matmul(v)


class Head(object):
    def __init__(self, wq, wk, wv, wo):
        self.wq = wq
        self.wk = wk
        self.wv = wv
        self.wo = wo

    def forward(self, x):
        q = x.matmul(self.wq)
        k = x.matmul(self.wk)
        v = x.matmul(self.wv)
        return attention(q, k, v).matmul(self.wo)


class TransformerBlock(object):
    def __init__(self, heads, gamma1, beta1, gamma2, beta2, mlp, eps):
        self.heads = heads
        self.gamma1 = gamma1
        self.beta1 = beta1
        self.gamma2 = gamma2
        self.beta2 = beta2
        self.mlp = mlp
        self.eps = eps

    @jit.unroll_safe
    def forward(self, x):
        h = layernorm(x, self.gamma1, self.beta1, self.eps)
        a = self.heads[0].forward(h)
        for i in range(1, len(self.heads)):
            a = a.add(self.heads[i].forward(h))
        x = x.add(a)
        h2 = layernorm(x, self.gamma2, self.beta2, self.eps)
        return x.add(self.mlp.forward(h2))


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
