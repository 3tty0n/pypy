from . import asarray


def sgd_step(params, lr):
    neg_lr = asarray([-lr])
    for p in params:
        t = p.tensor
        g = t.grad
        if g is not None:
            p.tensor = t.add(g.mul(neg_lr)).detach()


class SGD(object):
    def __init__(self, params, lr):
        self.params = params
        self.lr = lr

    def step(self):
        sgd_step(self.params, self.lr)
