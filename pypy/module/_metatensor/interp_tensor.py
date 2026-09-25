import os
from rpython.metatensor import core, device, kernels, lazy, nn, ops, runtime
from rpython.metatensor.nn import _ready
from rpython.rlib import jit
from pypy.interpreter.baseobjspace import W_Root
from pypy.interpreter.typedef import TypeDef, GetSetProperty
from pypy.interpreter.gateway import interp2app, unwrap_spec
from pypy.interpreter.error import oefmt, wrap_oserror


def _floats_w(space, w_list):
    return [space.float_w(w_item) for w_item in space.listview(w_list)]


def _ints_w(space, w_list):
    return [space.int_w(w_item) for w_item in space.listview(w_list)]


def _shape_w(space, shape):
    return space.newtuple([space.newint(d) for d in shape])


def _mismatch(space):
    return oefmt(space.w_ValueError, "shape mismatch")


class _Iotas(object):
    def __init__(self):
        self.cache = {}
_iotas = _Iotas()

def _iota(like):
    n = ops.tensor_size(like)
    dtype = ops.tensor_dtype(like)
    key = n * core.NDTYPES + dtype
    t = _iotas.cache.get(key, core.NULLTENSOR)
    if not t:
        t = core.new_tensor(n, like.shape, dtype)
        for i in range(n):
            t.host[i] = float(i)
        device.dev(t)
        _iotas.cache[key] = t
    return t


def _wrap(t):
    return W_Tensor(nn.Tensor(t))


_UNARY_IDS = dict([(core.UNARY_NAMES[i], i)
                   for i in range(len(core.UNARY_NAMES))])
_BINARY_IDS = dict([(core.BINARY_NAMES[i], i)
                    for i in range(len(core.BINARY_NAMES))])


@jit.elidable
def _lookup(ids, name):
    return ids.get(name, -1)


def _fn_id(space, ids, name):
    fn = _lookup(ids, name)
    if fn < 0:
        raise oefmt(space.w_ValueError, "unknown function '%s'", name)
    return jit.promote(fn)


class W_Tensor(W_Root):
    _immutable_fields_ = ['tensor']

    def __init__(self, tensor):
        self.tensor = tensor

    def _other(self, space, w_other):
        other = space.interp_w(W_Tensor, w_other)
        return other.tensor

    def _operand(self, space, w_other):
        """A tensor, or a Python number as a cached scalar of this tensor's
        dtype, which the fusion pass folds into the kernel body."""
        if isinstance(w_other, W_Tensor):
            return w_other.tensor
        dtype = jit.promote(ops.tensor_dtype(self.tensor.t))
        return nn.Tensor(runtime.scalar_of(
            jit.promote(space.float_w(w_other)), dtype))

    def descr_add(self, space, w_other):
        try:
            return W_Tensor(self.tensor.add(self._operand(space, w_other)))
        except ValueError:
            raise _mismatch(space)

    def descr_mul(self, space, w_other):
        try:
            return W_Tensor(self.tensor.mul(self._operand(space, w_other)))
        except ValueError:
            raise _mismatch(space)

    def descr_add_(self, space, w_other):
        try:
            self.tensor.add_(self._other(space, w_other))
        except ValueError:
            raise oefmt(space.w_ValueError, "in-place op not allowed")
        return self

    def descr_mul_(self, space, w_other):
        try:
            self.tensor.mul_(self._other(space, w_other))
        except ValueError:
            raise oefmt(space.w_ValueError, "in-place op not allowed")
        return self

    def descr_sub(self, space, w_other):
        try:
            return W_Tensor(self.tensor.sub(self._operand(space, w_other)))
        except ValueError:
            raise _mismatch(space)

    def descr_div(self, space, w_other):
        try:
            return W_Tensor(self.tensor.div(self._operand(space, w_other)))
        except ValueError:
            raise _mismatch(space)

    def descr_exp(self, space):
        return W_Tensor(self.tensor.exp())

    def descr_sqrt(self, space):
        return W_Tensor(self.tensor.sqrt())

    @unwrap_spec(name='text')
    def descr_unary(self, space, name):
        """name is one of core.UNARY_NAMES."""
        fn = _fn_id(space, _UNARY_IDS, name)
        return W_Tensor(self.tensor.unary(fn))

    @unwrap_spec(name='text')
    def descr_binary(self, space, name, w_other):
        """name is one of core.BINARY_NAMES; comparisons give 1.0 or 0.0.
        other may be a Python number."""
        fn = _fn_id(space, _BINARY_IDS, name)
        try:
            return W_Tensor(self.tensor.binary(self._operand(space, w_other),
                                               fn))
        except ValueError:
            raise _mismatch(space)

    def descr_where(self, space, w_a, w_b):
        """a where this tensor is nonzero, b elsewhere."""
        try:
            return W_Tensor(self.tensor.where(self._operand(space, w_a),
                                              self._operand(space, w_b)))
        except ValueError:
            raise _mismatch(space)

    @unwrap_spec(axis=int)
    def descr_max(self, space, axis=-1):
        return W_Tensor(self.tensor.max(axis))

    def _rows_cols(self, space):
        t = self.tensor.t
        if ops.tensor_ndim(t) != 2:
            raise _mismatch(space)
        return ops.tensor_shape(t, 0), ops.tensor_shape(t, 1)

    def _rowwise(self, space):
        rows, cols = self._rows_cols(space)
        return cols

    def _weight(self, space, w_other, cols):
        other = self._other(space, w_other)
        if (ops.tensor_size(other.t) != cols or
                ops.tensor_dtype(other.t) != ops.tensor_dtype(self.tensor.t)):
            raise _mismatch(space)
        return other

    def descr_softmax(self, space):
        self._rowwise(space)
        return W_Tensor(nn.softmax(self.tensor))

    @unwrap_spec(eps=float)
    def descr_layer_norm(self, space, w_gamma, w_beta, eps=1e-5):
        cols = self._rowwise(space)
        gamma = self._weight(space, w_gamma, cols)
        beta = self._weight(space, w_beta, cols)
        return W_Tensor(nn.layernorm(self.tensor, gamma, beta, eps))

    @unwrap_spec(eps=float)
    def descr_rms_norm(self, space, w_gamma, eps=1e-5):
        cols = self._rowwise(space)
        gamma = self._weight(space, w_gamma, cols)
        return W_Tensor(nn.rmsnorm(self.tensor, gamma, eps))

    def descr_gelu(self, space):
        return W_Tensor(nn.gelu(self.tensor))

    def descr_gelu_erf(self, space):
        return W_Tensor(nn.gelu_erf(self.tensor))

    def descr_silu(self, space):
        return W_Tensor(nn.silu(self.tensor))

    def descr_relu(self, space):
        return W_Tensor(self.tensor.relu())

    @unwrap_spec(axis=int)
    def descr_sum(self, space, axis=-1):
        return W_Tensor(self.tensor.sum(axis))

    def descr_item(self, space):
        return space.newfloat(self.tensor.item())

    def descr_argmax(self, space):
        """Index of the largest element, as a host int: max, then the
        position of the element equal to it.  On a tie the indices add up,
        which a caller comparing token streams will see."""
        t = self.tensor.t
        m = ops.max(t)
        hit = ops.eqmask(t, m, ops.bcast(t, m))
        return space.newint(int(ops.item(ops.sum(ops.mul(hit, _iota(t))))))

    @unwrap_spec(transpose_b=bool, tf32=bool)
    def descr_matmul(self, space, w_other, transpose_b=False, tf32=False):
        try:
            return W_Tensor(self.tensor.matmul(self._other(space, w_other),
                                               transpose_b, 1 if tf32 else 0))
        except ValueError:
            raise _mismatch(space)

    def descr_reshape(self, space, w_shape):
        shape = _ints_w(space, w_shape)
        try:
            return W_Tensor(self.tensor.reshape(shape))
        except ValueError:
            raise _mismatch(space)

    @unwrap_spec(heads=int, dcols=int, off_a=int, off_b=int, seqs=int)
    def descr_attn_scores(self, space, w_other, heads, dcols=-1, off_a=0,
                          off_b=0, seqs=1):
        """seqs > 1 says the rows are a folded batch: seqs independent
        sequences stacked down the rows, each of shape[0] // seqs rows.
        Attention then runs per sequence and the result is
        [seqs*heads*rows, krows], sequence-major, where krows is other's
        rows per sequence: the same as rows, or another length for
        cross-attention (inference only)."""
        t = self.tensor.t
        other = self._other(space, w_other)
        if (heads <= 0 or seqs <= 0 or ops.tensor_ndim(t) != 2 or
                ops.tensor_ndim(other.t) != 2):
            raise _mismatch(space)
        total = ops.tensor_shape(t, 0)
        if total % seqs != 0:
            raise _mismatch(space)
        rows = total // seqs
        lda = ops.tensor_shape(t, 1)
        ldb = ops.tensor_shape(other.t, 1)
        ktotal = ops.tensor_shape(other.t, 0)
        d = lda if dcols <= 0 else dcols
        if (d % heads != 0 or off_a < 0 or off_b < 0 or
                off_a + d > lda or off_b + d > ldb or
                ktotal <= 0 or ktotal % seqs != 0 or
                ops.tensor_dtype(other.t) != ops.tensor_dtype(t)):
            raise _mismatch(space)
        return W_Tensor(self.tensor.attn_scores(
            other, heads, rows, d // heads, lda, ldb, off_a, off_b, seqs,
            ktotal // seqs))

    @unwrap_spec(heads=int, dcols=int, off_b=int, seqs=int)
    def descr_attn_context(self, space, w_other, heads, dcols=-1, off_b=0,
                           seqs=1):
        """The mirror of attn_scores: [seqs*heads*rows, krows]
        probabilities against [seqs*krows, ldb] values, back to
        [seqs*rows, d]."""
        t = self.tensor.t
        other = self._other(space, w_other)
        if (heads <= 0 or seqs <= 0 or ops.tensor_ndim(t) != 2 or
                ops.tensor_ndim(other.t) != 2):
            raise _mismatch(space)
        krows = ops.tensor_shape(t, 1)
        ldb = ops.tensor_shape(other.t, 1)
        d = ldb if dcols <= 0 else dcols
        qtotal = ops.tensor_shape(t, 0)
        if (qtotal % (seqs * heads) != 0 or d % heads != 0 or
                off_b < 0 or off_b + d > ldb or
                ops.tensor_shape(other.t, 0) != seqs * krows or
                ops.tensor_dtype(other.t) != ops.tensor_dtype(t)):
            raise _mismatch(space)
        return W_Tensor(self.tensor.attn_context(
            other, heads, qtotal // (seqs * heads), d // heads, ldb, off_b,
            seqs, krows))

    @unwrap_spec(heads=int, krows=int, dcols=int, off_a=int, off_b=int)
    def descr_decode_scores(self, space, w_cache, heads, krows, dcols,
                            off_a=0, off_b=0):
        """Scores of this tensor's rows against the first krows rows of a
        key/value cache: [heads*rows, krows].  dcols is the model width, the
        cache and the queries may be wider (whole qkv rows)."""
        qrows, lda = self._rows_cols(space)
        cache = self._other(space, w_cache)
        if ops.tensor_ndim(cache.t) != 2 or heads <= 0 or dcols % heads != 0:
            raise _mismatch(space)
        ldb = ops.tensor_shape(cache.t, 1)
        if (krows <= 0 or krows > ops.tensor_shape(cache.t, 0) or
                off_a < 0 or off_b < 0 or off_a + dcols > lda or
                off_b + dcols > ldb or
                ops.tensor_dtype(cache.t) != ops.tensor_dtype(self.tensor.t)):
            raise _mismatch(space)
        return W_Tensor(self.tensor.decode_scores(
            cache, heads, qrows, krows, dcols // heads, lda, ldb, off_a,
            off_b))

    @unwrap_spec(heads=int, dcols=int, off_b=int)
    def descr_decode_context(self, space, w_cache, heads, dcols, off_b=0):
        """[heads*rows, krows] probabilities against the first krows rows of
        the cache, back to [rows, dcols]."""
        hq, krows = self._rows_cols(space)
        cache = self._other(space, w_cache)
        if (ops.tensor_ndim(cache.t) != 2 or heads <= 0 or hq % heads != 0 or
                dcols % heads != 0):
            raise _mismatch(space)
        ldb = ops.tensor_shape(cache.t, 1)
        if (krows > ops.tensor_shape(cache.t, 0) or off_b < 0 or
                off_b + dcols > ldb or
                ops.tensor_dtype(cache.t) != ops.tensor_dtype(self.tensor.t)):
            raise _mismatch(space)
        return W_Tensor(self.tensor.decode_context(
            cache, heads, hq // heads, krows, dcols // heads, ldb, off_b))

    @unwrap_spec(row=int)
    def descr_write_rows(self, space, row, w_src):
        """self[row:row+len(src)] = src, in place; returns self."""
        rows, c = self._rows_cols(space)
        src = self._other(space, w_src)
        n = ops.tensor_size(src.t)
        if (row < 0 or c <= 0 or n % c != 0 or row + n // c > rows or
                ops.tensor_dtype(src.t) != ops.tensor_dtype(self.tensor.t)):
            raise _mismatch(space)
        self.tensor.write_rows(row, src)
        return self

    @unwrap_spec(dh=int)
    def descr_rot_half(self, space, dh):
        if dh <= 1 or dh % 2 != 0:
            raise _mismatch(space)
        rows, cols = self._rows_cols(space)
        if cols % dh != 0:
            raise _mismatch(space)
        return W_Tensor(self.tensor.rot_half(dh))

    @unwrap_spec(heads=int)
    def descr_head_split(self, space, heads):
        if heads <= 0:
            raise _mismatch(space)
        rows, d = self._rows_cols(space)
        if d % heads != 0:
            raise _mismatch(space)
        return W_Tensor(self.tensor.head_split(rows, d // heads, heads))

    @unwrap_spec(heads=int)
    def descr_head_merge(self, space, heads):
        if heads <= 0:
            raise _mismatch(space)
        hr, cols = self._rows_cols(space)
        if hr % heads != 0:
            raise _mismatch(space)
        return W_Tensor(self.tensor.head_merge(hr // heads, cols, heads))

    @unwrap_spec(batch=int, transpose_b=bool)
    def descr_bmm(self, space, w_other, batch, transpose_b=False):
        a = self.tensor.t
        b = self._other(space, w_other).t
        if (batch <= 0 or ops.tensor_ndim(a) != 2 or
                ops.tensor_ndim(b) != 2 or
                ops.tensor_shape(a, 0) % batch != 0 or
                ops.tensor_shape(b, 0) % batch != 0):
            raise _mismatch(space)
        rows = ops.tensor_shape(a, 0) // batch
        inner = ops.tensor_shape(a, 1)
        if transpose_b:
            cols = ops.tensor_shape(b, 0) // batch
            ok = ops.tensor_shape(b, 1) == inner
        else:
            cols = ops.tensor_shape(b, 1)
            ok = ops.tensor_shape(b, 0) // batch == inner
        if not ok:
            raise _mismatch(space)
        return W_Tensor(self.tensor.bmm(
            self._other(space, w_other), batch, rows, cols, inner,
            1 if transpose_b else 0))

    def descr_tolist(self, space):
        t = ops.tensor_force(self.tensor.t)
        h = device.host(t)
        n = ops.tensor_size(t)
        return space.newlist([space.newfloat(h[i]) for i in range(n)])

    @unwrap_spec(path='fsencode')
    def descr_tofile(self, space, path):
        """Write the elements to path as raw little-endian bytes of this
        tensor's dtype, as numpy's tofile does, without a host copy."""
        t = ops.tensor_force(self.tensor.t)
        try:
            fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0644)
        except OSError as e:
            raise wrap_oserror(space, e, path)
        ok = device.dump(t, fd)
        os.close(fd)
        if not ok:
            raise oefmt(space.w_IOError, "tofile: writing %s failed", path)

    def descr_take(self, space, w_idx):
        t = self.tensor.t
        idx = self._other(space, w_idx).t
        _ready(t)
        _ready(idx)
        trows, cols = self._rows_cols(space)
        if ops.tensor_dtype(idx) != ops.tensor_dtype(t):
            raise _mismatch(space)
        rows = ops.tensor_size(idx)
        return _wrap(runtime.rowgather(t, idx, rows, cols))

    @unwrap_spec(c=int, h=int, w=int, k=int, pad=int, stride=int)
    def descr_im2col(self, space, c, h, w, k=3, pad=1, stride=1):
        t = self.tensor.t
        _ready(t)
        self._conv_check(space, c, h, w, k, pad, stride)
        return _wrap(runtime.im2col(t, c, h, w, k, pad, stride))

    @unwrap_spec(c=int, h=int, w=int, k=int, pad=int, stride=int)
    def descr_im2col_nhwc(self, space, c, h, w, k=3, pad=1, stride=1):
        t = self.tensor.t
        _ready(t)
        self._conv_check(space, c, h, w, k, pad, stride)
        return _wrap(runtime.im2col_nhwc(t, c, h, w, k, pad, stride))

    def _conv_check(self, space, c, h, w, k, pad, stride):
        t = self.tensor.t
        if (c <= 0 or h <= 0 or w <= 0 or k <= 0 or pad < 0 or stride <= 0 or
                h + 2 * pad < k or w + 2 * pad < k or
                ops.tensor_size(t) % (c * h * w) != 0):
            raise _mismatch(space)

    @unwrap_spec(c=int, h=int, w=int, k=int, stride=int, pad=int)
    def descr_maxpool2(self, space, c, h, w, k=2, stride=2, pad=0):
        t = self.tensor.t
        _ready(t)
        self._conv_check(space, c, h, w, k, pad, stride)
        return _wrap(runtime.maxpool2(t, c, h, w, k, stride, pad))

    @unwrap_spec(c=int, h=int, w=int, k=int, stride=int, pad=int)
    def descr_maxpool2_nhwc(self, space, c, h, w, k=2, stride=2, pad=0):
        t = self.tensor.t
        _ready(t)
        self._conv_check(space, c, h, w, k, pad, stride)
        return _wrap(runtime.maxpool2_nhwc(t, c, h, w, k, stride,
                                           pad))

    @unwrap_spec(c=int, h=int, w=int, k=int, stride=int, pad=int,
                 groups=int, kw=int, stride_w=int, pad_w=int, dilation=int,
                 dilation_w=int)
    def descr_conv2d(self, space, w_weight, c, h, w, w_bias=None, k=3,
                     stride=1, pad=1, groups=1, kw=0, stride_w=0, pad_w=-1,
                     dilation=1, dilation_w=0):
        """NCHW convolution as [n, o*oh*ow].  k, stride, pad and dilation
        are the height's, and the width's too unless kw, stride_w, pad_w or
        dilation_w say otherwise.  weight is [c*k*kw, o // groups]: the rows
        of group g are its c // groups input channels by kernel row by kernel
        column, so torch's [o, c // groups, k, kw] weight w goes in as
        w.view(groups, o // groups, -1).transpose(1, 2).  groups == c
        (depthwise) runs as one direct kernel, other groups as im2col and a
        batched GEMM, one batch per group."""
        weight = self._other(space, w_weight).t
        t = self.tensor.t
        _ready(t)
        _ready(weight)
        kh, sh, ph, dh = k, stride, pad, dilation
        if kw <= 0:
            kw = k
        sw = stride_w if stride_w > 0 else stride
        pw = pad_w if pad_w >= 0 else pad
        dw = dilation_w if dilation_w > 0 else dilation
        if (c <= 0 or h <= 0 or w <= 0 or kh <= 0 or kw <= 0 or sh <= 0 or
                sw <= 0 or ph < 0 or pw < 0 or dh <= 0 or dw <= 0 or groups <= 0 or
                c % groups != 0 or ops.tensor_size(t) % (c * h * w) != 0 or
                core.conv_out(h, kh, sh, ph, dh) <= 0 or
                core.conv_out(w, kw, sw, pw, dw) <= 0 or
                ops.tensor_ndim(weight) != 2 or
                ops.tensor_shape(weight, 0) != c * kh * kw):
            raise _mismatch(space)
        og = ops.tensor_shape(weight, 1)
        o = og * groups
        ohw = (core.conv_out(h, kh, sh, ph, dh) *
               core.conv_out(w, kw, sw, pw, dw))
        rows = ops.tensor_size(t) // (c * h * w)
        bias = core.NULLTENSOR
        if w_bias is not None and not space.is_none(w_bias):
            bias = self._other(space, w_bias).t
            _ready(bias)
        if groups > 1 and groups == c:
            if bias and (ops.tensor_size(bias) != o or
                         ops.tensor_dtype(bias) != ops.tensor_dtype(t)):
                raise _mismatch(space)
            return _wrap(runtime.depthwise_conv2d(t, weight, bias, c, h, w,
                                                  kh, kw, sh, sw, ph, pw, dh,
                                                  dw, og))
        cols = runtime.im2col2(t, c, h, w, kh, kw, sh, sw, ph, pw, dh, dw)
        if groups == 1:
            y = runtime.tensor_matmul(cols, weight, rows * ohw, o,
                                      c * kh * kw, 0, 0, 1)
        else:
            inner = c // groups * kh * kw
            y = runtime.tensor_bmm(cols, weight, groups, rows * ohw, og, inner,
                                   0, 0, c * kh * kw, og, o, inner,
                                   inner * og, og, rows * ohw * o, rows * ohw)
        if bias:
            y = ops.add_p(y, bias, core.BC_R_ROW)
        return _wrap(runtime.col2chw(y, rows, ohw, o))

    @unwrap_spec(c=int, h=int, w=int, k=int, stride=int, pad=int,
                 output_padding=int, dilation=int, kw=int, stride_w=int,
                 pad_w=int, output_padding_w=int, dilation_w=int)
    def descr_conv_transpose2d(self, space, w_weight, c, h, w, w_bias=None,
                               k=3, stride=1, pad=0, output_padding=0,
                               dilation=1, kw=0, stride_w=0, pad_w=-1,
                               output_padding_w=-1, dilation_w=0):
        """NCHW transposed convolution as [n, o*oh*ow], the width's values
        defaulting to the height's as in conv2d.  weight is [c*k*kw, o]:
        torch's [c, o, k, kw] weight w as w.permute(0, 2, 3, 1).reshape(
        c*k*kw, o).  No groups."""
        weight = self._other(space, w_weight).t
        t = self.tensor.t
        _ready(t)
        _ready(weight)
        kh, sh, ph, dh, oph = k, stride, pad, dilation, output_padding
        if kw <= 0:
            kw = k
        sw = stride_w if stride_w > 0 else stride
        pw = pad_w if pad_w >= 0 else pad
        dw = dilation_w if dilation_w > 0 else dilation
        opw = output_padding_w if output_padding_w >= 0 else output_padding
        oh = (h - 1) * sh - 2 * ph + dh * (kh - 1) + oph + 1
        ow = (w - 1) * sw - 2 * pw + dw * (kw - 1) + opw + 1
        if (c <= 0 or h <= 0 or w <= 0 or kh <= 0 or kw <= 0 or sh <= 0 or
                sw <= 0 or ph < 0 or pw < 0 or dh <= 0 or dw <= 0 or
                oph < 0 or opw < 0 or (oph >= sh and oph >= dh) or
                (opw >= sw and opw >= dw) or oh <= 0 or ow <= 0 or
                ops.tensor_size(t) % (c * h * w) != 0 or
                ops.tensor_ndim(weight) != 2 or
                ops.tensor_shape(weight, 0) != c * kh * kw):
            raise _mismatch(space)
        o = ops.tensor_shape(weight, 1)
        rows = ops.tensor_size(t) // (c * h * w)
        cols = runtime.im2col_t(t, c, h, w, kh, kw, sh, sw, ph, pw, dh, dw,
                                oh, ow)
        y = runtime.tensor_matmul(cols, weight, rows * oh * ow, o,
                                  c * kh * kw, 0, 0, 1)
        if w_bias is not None and not space.is_none(w_bias):
            bias = self._other(space, w_bias).t
            _ready(bias)
            y = ops.add_p(y, bias, core.BC_R_ROW)
        return _wrap(runtime.col2chw(y, rows, oh * ow, o))

    def _pool_check(self, space, c, h, w):
        t = self.tensor.t
        _ready(t)
        if (c <= 0 or h <= 0 or w <= 0 or
                ops.tensor_size(t) % (c * h * w) != 0):
            raise _mismatch(space)
        return t

    @unwrap_spec(c=int, h=int, w=int, k=int, stride=int, pad=int,
                 count_include_pad=bool, ceil_mode=bool, kw=int, stride_w=int,
                 pad_w=int)
    def descr_avg_pool2d(self, space, c, h, w, k=2, stride=0, pad=0,
                         count_include_pad=True, ceil_mode=False, kw=0,
                         stride_w=0, pad_w=-1):
        """NCHW average pooling as [n, c*oh*ow].  stride defaults to the
        kernel; the width takes the height's values unless kw, stride_w or
        pad_w are given."""
        t = self._pool_check(space, c, h, w)
        if kw <= 0:
            kw = k
        sh = stride if stride > 0 else k
        sw = stride_w if stride_w > 0 else (stride if stride > 0 else kw)
        pw = pad_w if pad_w >= 0 else pad
        if (k <= 0 or pad < 0 or pw < 0 or 2 * pad > k or 2 * pw > kw or
                runtime.pool_out(h, k, sh, pad, ceil_mode) <= 0 or
                runtime.pool_out(w, kw, sw, pw, ceil_mode) <= 0):
            raise _mismatch(space)
        return _wrap(runtime.avg_pool2d(t, c, h, w, k, kw, sh, sw, pad, pw,
                                        count_include_pad, ceil_mode))

    @unwrap_spec(c=int, h=int, w=int, oh=int, ow=int)
    def descr_adaptive_avg_pool2d(self, space, c, h, w, oh, ow=0):
        """NCHW adaptive average pooling to oh x ow (ow defaults to oh) as
        [n, c*oh*ow]."""
        t = self._pool_check(space, c, h, w)
        if ow <= 0:
            ow = oh
        if oh <= 0:
            raise _mismatch(space)
        return _wrap(runtime.adaptive_avg_pool2d(t, c, h, w, oh, ow))

    @unwrap_spec(offset=int)
    def descr_strided(self, space, w_shape, w_strides, offset=0):
        """A new contiguous tensor of the given shape whose element
        (i0, i1, ..) is this tensor's flat element offset + sum(i_j *
        strides[j]).  Strides may be 0 (expand, repeat) or negative; this is
        permute, transpose, slice, select, narrow and expand, in one
        launch."""
        t = self.tensor.t
        _ready(t)
        shape = _ints_w(space, w_shape)
        strides = _ints_w(space, w_strides)
        if len(shape) != len(strides):
            raise _mismatch(space)
        lo = hi = offset
        n = 1
        for j in range(len(shape)):
            if shape[j] < 0:
                raise _mismatch(space)
            n *= shape[j]
            step = (shape[j] - 1) * strides[j]
            if step < 0:
                lo += step
            else:
                hi += step
        if n > 0 and (lo < 0 or hi >= ops.tensor_size(t)):
            raise _mismatch(space)
        return _wrap(runtime.strided(t, shape, strides, offset))

    @unwrap_spec(value=float)
    def descr_pad(self, space, w_shape, w_pads, value=0.0):
        """This tensor, contiguous with the given shape, padded with value:
        pads = [before_0, after_0, before_1, after_1, ..] in dimension order,
        one pair per dimension (negative pads crop)."""
        t = self.tensor.t
        _ready(t)
        shape = _ints_w(space, w_shape)
        pads = _ints_w(space, w_pads)
        if len(pads) != 2 * len(shape):
            raise _mismatch(space)
        n = 1
        for j in range(len(shape)):
            if (shape[j] < 0 or
                    shape[j] + pads[2 * j] + pads[2 * j + 1] < 0):
                raise _mismatch(space)
            n *= shape[j]
        if n != ops.tensor_size(t):
            raise _mismatch(space)
        return _wrap(runtime.pad(t, shape, pads, value))

    def _take(self, space, w_idx, outer, inner, per_row):
        t = self.tensor.t
        idx = self._other(space, w_idx).t
        _ready(t)
        _ready(idx)
        n = ops.tensor_size(t)
        m = ops.tensor_size(idx)
        if (outer <= 0 or inner <= 0 or n % (outer * inner) != 0 or
                ops.tensor_dtype(idx) != ops.tensor_dtype(t)):
            raise _mismatch(space)
        if not per_row:
            return _wrap(runtime.take(t, idx, outer, m, inner, 0, 1, 0))
        if m % (outer * inner) != 0:
            raise _mismatch(space)
        k = m // (outer * inner)
        return _wrap(runtime.take(t, idx, outer, k, inner, k * inner, inner,
                                  1))

    @unwrap_spec(outer=int, inner=int)
    def descr_gather(self, space, w_idx, outer, inner=1):
        """torch.gather along the middle axis of this tensor as
        [outer, d, inner]: idx is [outer, k, inner] (as floats of this dtype)
        and the result [outer*k, inner].  An index outside [0, d) gives 0."""
        return self._take(space, w_idx, outer, inner, True)

    @unwrap_spec(outer=int, inner=int)
    def descr_index_select(self, space, w_idx, outer=1, inner=1):
        """index_select along the middle axis of this tensor as
        [outer, d, inner] with idx a flat [k] (floats of this dtype); the
        result is [outer*k, inner]."""
        return self._take(space, w_idx, outer, inner, False)

    def descr_detach(self, space):
        return W_Tensor(nn.Tensor(self.tensor.t, self.tensor.requires_grad))

    def descr_force(self, space):
        return W_Tensor(nn.Tensor(ops.tensor_force(self.tensor.t),
                                  self.tensor.requires_grad))

    def descr_backward(self, space):
        self.tensor.backward()

    def descr_zero_grad(self, space):
        self.tensor.grad = None

    def descr_shape(self, space):
        t = self.tensor.t
        shape = [ops.tensor_shape(t, i) for i in range(ops.tensor_ndim(t))]
        return _shape_w(space, shape)

    def descr_size(self, space):
        return space.newint(ops.tensor_size(self.tensor.t))

    def descr_grad(self, space):
        g = self.tensor.grad
        if g is None:
            return space.w_None
        return W_Tensor(g)

    def descr_requires_grad(self, space):
        return space.newbool(self.tensor.requires_grad)

    def descr_dtype(self, space):
        dtype = ops.tensor_dtype(self.tensor.t)
        return space.newtext(core.DTYPE_NAMES[dtype])

    def descr_astype(self, space, w_dtype):
        dtype = _dtype_w(space, w_dtype)
        return W_Tensor(nn.Tensor(
            ops.astype(self.tensor.t, dtype), self.tensor.requires_grad))

    def descr_repr(self, space):
        t = self.tensor.t
        shape = [ops.tensor_shape(t, i) for i in range(ops.tensor_ndim(t))]
        parts = [str(d) for d in shape]
        return space.newtext("Tensor(shape=(%s))" % ", ".join(parts))


W_Tensor.typedef = TypeDef(
    'Tensor',
    add=interp2app(W_Tensor.descr_add),
    mul=interp2app(W_Tensor.descr_mul),
    add_=interp2app(W_Tensor.descr_add_),
    mul_=interp2app(W_Tensor.descr_mul_),
    sub=interp2app(W_Tensor.descr_sub),
    div=interp2app(W_Tensor.descr_div),
    exp=interp2app(W_Tensor.descr_exp),
    sqrt=interp2app(W_Tensor.descr_sqrt),
    unary=interp2app(W_Tensor.descr_unary),
    binary=interp2app(W_Tensor.descr_binary),
    where=interp2app(W_Tensor.descr_where),
    max=interp2app(W_Tensor.descr_max),
    relu=interp2app(W_Tensor.descr_relu),
    softmax=interp2app(W_Tensor.descr_softmax),
    layer_norm=interp2app(W_Tensor.descr_layer_norm),
    rms_norm=interp2app(W_Tensor.descr_rms_norm),
    gelu=interp2app(W_Tensor.descr_gelu),
    gelu_erf=interp2app(W_Tensor.descr_gelu_erf),
    silu=interp2app(W_Tensor.descr_silu),
    sum=interp2app(W_Tensor.descr_sum),
    item=interp2app(W_Tensor.descr_item),
    argmax=interp2app(W_Tensor.descr_argmax),
    matmul=interp2app(W_Tensor.descr_matmul),
    reshape=interp2app(W_Tensor.descr_reshape),
    attn_scores=interp2app(W_Tensor.descr_attn_scores),
    attn_context=interp2app(W_Tensor.descr_attn_context),
    decode_scores=interp2app(W_Tensor.descr_decode_scores),
    decode_context=interp2app(W_Tensor.descr_decode_context),
    write_rows=interp2app(W_Tensor.descr_write_rows),
    rot_half=interp2app(W_Tensor.descr_rot_half),
    head_split=interp2app(W_Tensor.descr_head_split),
    head_merge=interp2app(W_Tensor.descr_head_merge),
    bmm=interp2app(W_Tensor.descr_bmm),
    take=interp2app(W_Tensor.descr_take),
    tofile=interp2app(W_Tensor.descr_tofile),
    tolist=interp2app(W_Tensor.descr_tolist),
    im2col=interp2app(W_Tensor.descr_im2col),
    im2col_nhwc=interp2app(W_Tensor.descr_im2col_nhwc),
    maxpool2=interp2app(W_Tensor.descr_maxpool2),
    maxpool2_nhwc=interp2app(W_Tensor.descr_maxpool2_nhwc),
    conv2d=interp2app(W_Tensor.descr_conv2d),
    conv_transpose2d=interp2app(W_Tensor.descr_conv_transpose2d),
    avg_pool2d=interp2app(W_Tensor.descr_avg_pool2d),
    adaptive_avg_pool2d=interp2app(W_Tensor.descr_adaptive_avg_pool2d),
    strided=interp2app(W_Tensor.descr_strided),
    pad=interp2app(W_Tensor.descr_pad),
    gather=interp2app(W_Tensor.descr_gather),
    index_select=interp2app(W_Tensor.descr_index_select),
    detach=interp2app(W_Tensor.descr_detach),
    force=interp2app(W_Tensor.descr_force),
    backward=interp2app(W_Tensor.descr_backward),
    zero_grad=interp2app(W_Tensor.descr_zero_grad),
    __add__=interp2app(W_Tensor.descr_add),
    __mul__=interp2app(W_Tensor.descr_mul),
    __sub__=interp2app(W_Tensor.descr_sub),
    __div__=interp2app(W_Tensor.descr_div),
    __truediv__=interp2app(W_Tensor.descr_div),
    __repr__=interp2app(W_Tensor.descr_repr),
    shape=GetSetProperty(W_Tensor.descr_shape),
    size=GetSetProperty(W_Tensor.descr_size),
    grad=GetSetProperty(W_Tensor.descr_grad),
    requires_grad=GetSetProperty(W_Tensor.descr_requires_grad),
    dtype=GetSetProperty(W_Tensor.descr_dtype),
    astype=interp2app(W_Tensor.descr_astype),
)


def _dtype_w(space, w_dtype):
    try:
        dtype = core.dtype_of_name(space.text_w(w_dtype))
    except ValueError:
        raise oefmt(space.w_ValueError, "unknown dtype")
    ensure_dtype(dtype)
    return dtype


class DeviceState(object):
    ready = False
device_state = DeviceState()

def ensure_device():
    if not device_state.ready:
        device_state.ready = True
        kernels.init_device()

def ensure_dtype(dtype):
    ensure_device()
    kernels.init_dtype(dtype)

def scalar(space, w_value, w_dtype=None):
    """A cached 0-d tensor.  The value is promoted so the trace sees a
    constant pointer, which is what lets the fusion pass put the number into
    the kernel body instead of spending an input slot on it."""
    ensure_device()
    dtype = core.F64
    if w_dtype is not None and not space.is_none(w_dtype):
        dtype = _dtype_w(space, w_dtype)
    return W_Tensor(nn.Tensor(runtime.scalar_of(
        jit.promote(space.float_w(w_value)), jit.promote(dtype))))


@unwrap_spec(requires_grad=bool)
def tensor_flat(space, w_data, w_shape, requires_grad=False,
                w_dtype=None):
    ensure_device()
    dtype = core.F64
    if w_dtype is not None and not space.is_none(w_dtype):
        dtype = _dtype_w(space, w_dtype)
    values = _floats_w(space, w_data)
    shape = _ints_w(space, w_shape)
    n = 1
    for d in shape:
        n *= d
    if n != len(values):
        raise _mismatch(space)
    t = core.from_list(values, dtype)
    if len(shape) != 1 or shape[0] != n:
        try:
            t = ops.reshape(t, shape)
        except ValueError:
            raise _mismatch(space)
    return W_Tensor(nn.Tensor(t, requires_grad))


@unwrap_spec(requires_grad=bool)
def zeros(space, w_shape, requires_grad=False, w_dtype=None):
    ensure_device()
    dtype = core.F64
    if w_dtype is not None and not space.is_none(w_dtype):
        dtype = _dtype_w(space, w_dtype)
    shape = _ints_w(space, w_shape)
    t = core.zeros(shape, dtype)
    h = device.host(t)
    for i in range(t.size):
        h[i] = 0.0
    return W_Tensor(nn.Tensor(t, requires_grad))


@unwrap_spec(outer=int)
def cat(space, w_tensors, outer=1):
    """Each tensor viewed as [outer, size // outer], joined along the second
    axis into one [outer, sum of the widths]: torch.cat along dim d is
    outer = prod(shape[:d]).  Up to 8 tensors take one launch."""
    ts = []
    for w_t in space.listview(w_tensors):
        t = space.interp_w(W_Tensor, w_t).tensor.t
        _ready(t)
        ts.append(t)
    if not ts or outer <= 0:
        raise _mismatch(space)
    dtype = ops.tensor_dtype(ts[0])
    for t in ts:
        if (ops.tensor_size(t) % outer != 0 or
                ops.tensor_dtype(t) != dtype):
            raise _mismatch(space)
    return _wrap(runtime.cat(ts, outer))


def kernel_count(space):
    return space.newint(kernels.counter.n)


def kernel_compile_count(space):
    """Kernels actually compiled through the Triton subprocess (a disk-cache
    miss), as opposed to kernel_count() which also counts disk-cache hits."""
    return space.newint(kernels.miss_counter.n)


def launch_count(space):
    return space.newint(device.launch_count())


def lazy_enabled(space):
    return space.newbool(lazy.enabled())


def mark_step(space):
    """Materialize every deferred tensor the program still holds as a root.
    A deferred library needs this once per iteration; with the fusion pass on
    it does nothing, because the trace boundary already is one."""
    if lazy.enabled():
        lazy.mark_step()


def lazy_stats(space):
    """Deferred-execution counters: chains forced, nodes deferred, in-place
    barriers, live-set entries swept once the program dropped them, and forces
    that fell back to node-by-node evaluation.  All zero unless
    METATENSOR_LAZY is set."""
    s = lazy.stats
    return space.newtuple([space.newint(s.forces), space.newint(s.nodes),
                           space.newint(s.barriers), space.newint(s.pruned),
                           space.newint(s.fallbacks)])


def mem_total(space):
    return space.newint(device.mem_total())


def live_bytes(space):
    return space.newint(device.live_bytes())


def alloc_failed(space):
    return space.newbool(device.alloc_failed())


def cpu_fallbacks(space):
    """How many ops ran in host loops although the GPU is on: a failed
    compile, launch or allocation, or an op with no device kernel."""
    return space.newint(device.fallbacks.n)


def unfused_fallbacks(space):
    """How many times a fused kernel did not launch as one (a failed compile,
    launch or allocation, or a row or size it was not built for) and its ops
    ran one kernel each, while the GPU is on."""
    return space.newint(device.unfused_count())
