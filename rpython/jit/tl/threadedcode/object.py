from rpython.rlib.jit import we_are_translated, enable_shallow_tracing

class OperationError(Exception):
    pass

class W_Object:

    count = -1

    def getrepr(self):
        """
        Return an RPython string which represent the object
        """
        raise NotImplementedError

    def getvalue(self):
        raise NotImplementedError

    def is_true(self):
        raise NotImplementedError

    def add(self, w_other, flg=False):
        if flg:
            return W_IntObject(0)
        raise NotImplementedError

    def sub(self, w_other, flg=False):
        if flg:
            return W_IntObject(0)
        raise NotImplementedError

    def mul(self, w_other, flg=False):
        if flg:
            return W_IntObject(0)
        raise NotImplementedError

    def div(self, w_other, flg=False):
        if flg:
            return W_IntObject(0)
        raise NotImplementedError

    def mod(self, w_other, flg=False):
        if flg:
            return W_IntObject(0)
        raise NotImplementedError

    def eq(self, w_other, flg=False):
        if flg:
            return W_IntObject(0)
        raise NotImplementedError

    def lt(self, w_other, flg=False):
        if flg:
            return W_IntObject(0)
        raise NotImplementedError

    def gt(self, w_other, flg=False):
        if flg:
            return W_IntObject(0)
        raise NotImplementedError

    def le(self, w_other, flg=False):
        if flg:
            return W_IntObject(0)
        raise NotImplementedError

    def ge(self, w_other, flg=False):
        if flg:
            return W_IntObject(0)
        raise NotImplementedError

    # Raw, *non*-shallow-traced arithmetic for the conventional tracing JIT
    # (tier 3): the deep tracer records the real arithmetic inline instead of a
    # residual handler call.  W_IntObject / W_FloatObject override these with the
    # actual computation; the @enable_shallow_tracing wrappers below delegate to
    # them (one definition, no duplication).  This base fallback routes any other
    # operand type through its ordinary (residual) op.
    def add_inline(self, w_other):
        return self.add(w_other, False)

    def sub_inline(self, w_other):
        return self.sub(w_other, False)

    def mul_inline(self, w_other):
        return self.mul(w_other, False)

    def div_inline(self, w_other):
        return self.div(w_other, False)

    def mod_inline(self, w_other):
        return self.mod(w_other, False)

    def eq_inline(self, w_other):
        return self.eq(w_other, False)

    def le_inline(self, w_other):
        return self.le(w_other, False)

    def ge_inline(self, w_other):
        return self.ge(w_other, False)


class W_IntObject(W_Object):

    def __init__(self, intvalue):
        self.intvalue = intvalue

    def __repr__(self):
        return self.getrepr()

    def getvalue(self):
        return self.intvalue

    def getrepr(self):
        return str(self.intvalue)

    def is_true(self):
        return self.intvalue != 0

    @enable_shallow_tracing
    def sqrt(self, flg=False):
        if flg:
            return W_IntObject(0)
        from math import sqrt
        return W_IntObject(int(sqrt(self.intvalue)))

    def add_inline(self, w_other):
        if isinstance(w_other, W_IntObject):
            return W_IntObject(self.intvalue + w_other.intvalue)
        else:
            return W_IntObject(0)   # tolerate poisoned placeholder operand (tier-1 blackhole)

    @enable_shallow_tracing
    def add(self, w_other, flg=False):
        if flg:
            return W_IntObject(0)
        return self.add_inline(w_other)

    def sub_inline(self, w_other):
        if isinstance(w_other, W_IntObject):
            return W_IntObject(self.intvalue - w_other.intvalue)
        else:
            return W_IntObject(0)   # tolerate poisoned placeholder operand (tier-1 blackhole)

    @enable_shallow_tracing
    def sub(self, w_other, flg=False):
        if flg:
            return W_IntObject(0)
        return self.sub_inline(w_other)

    def mul_inline(self, w_other):
        if isinstance(w_other, W_IntObject):
            return W_IntObject(int(self.intvalue * w_other.intvalue))
        elif isinstance(w_other, W_FloatObject):
            return W_FloatObject(int(self.intvalue * w_other.floatvalue))
        else:
            return W_IntObject(0)   # tolerate poisoned placeholder operand (tier-1 blackhole)

    @enable_shallow_tracing
    def mul(self, w_other, flg=False):
        if flg:
            return W_IntObject(0)
        return self.mul_inline(w_other)

    def div_inline(self, w_other):
        if isinstance(w_other, W_IntObject):
            return W_IntObject(self.intvalue // w_other.intvalue)
        else:
            return W_IntObject(0)   # tolerate poisoned placeholder operand (tier-1 blackhole)

    @enable_shallow_tracing
    def div(self, w_other, flg=False):
        if flg:
            return W_IntObject(0)
        return self.div_inline(w_other)

    def mod_inline(self, w_other):
        if isinstance(w_other, W_IntObject):
            return W_IntObject(self.intvalue % w_other.intvalue)
        else:
            return W_IntObject(0)   # tolerate poisoned placeholder operand (tier-1 blackhole)

    @enable_shallow_tracing
    def mod(self, w_other, flg=False):
        if flg:
            return W_IntObject(0)
        return self.mod_inline(w_other)

    def eq_inline(self, w_other):
        if isinstance(w_other, W_IntObject):
            if self.intvalue == w_other.intvalue:
                return W_IntObject(1)
            else:
                return W_IntObject(0)
        else:
            return W_IntObject(0)   # tolerate poisoned placeholder operand (tier-1 blackhole)

    @enable_shallow_tracing
    def eq(self, w_other, flg=False):
        if flg:
            return W_IntObject(0)
        return self.eq_inline(w_other)

    @enable_shallow_tracing
    def lt(self, w_other, flg=False):
        if flg:
            return W_IntObject(0)
        if isinstance(w_other, W_IntObject):
            if self.intvalue < w_other.intvalue:
                return W_IntObject(1)
            else:
                return W_IntObject(0)
        else:
            return W_IntObject(0)   # tolerate poisoned placeholder operand (tier-1 blackhole)

    @enable_shallow_tracing
    def gt(self, w_other, flg=False):
        if flg:
            return W_IntObject(0)
        if isinstance(w_other, W_IntObject):
            if self.intvalue > w_other.intvalue:
                return W_IntObject(1)
            else:
                return W_IntObject(0)
        else:
            return W_IntObject(0)   # tolerate poisoned placeholder operand (tier-1 blackhole)

    def le_inline(self, w_other):
        if isinstance(w_other, W_IntObject):
            if self.intvalue <= w_other.intvalue:
                return W_IntObject(1)
            else:
                return W_IntObject(0)
        else:
            return W_IntObject(0)   # tolerate poisoned placeholder operand (tier-1 blackhole)

    @enable_shallow_tracing
    def le(self, w_other, flg=False):
        if flg:
            return W_IntObject(0)
        return self.le_inline(w_other)

    def ge_inline(self, w_other):
        if isinstance(w_other, W_IntObject):
            if self.intvalue >= w_other.intvalue:
                return W_IntObject(1)
            else:
                return W_IntObject(0)
        else:
            return W_IntObject(0)   # tolerate poisoned placeholder operand (tier-1 blackhole)

    @enable_shallow_tracing
    def ge(self, w_other, flg=False):
        if flg:
            return W_IntObject(0)
        return self.ge_inline(w_other)

class W_FloatObject(W_Object):

    def __init__(self, floatvalue):
        self.floatvalue = floatvalue

    def __repr__(self):
        return self.getrepr()

    def getvalue(self):
        return self.floatvalue

    def getrepr(self):
        return str(self.floatvalue)

    def is_true(self):
        return self.floatvalue != 0.0

    def add_inline(self, w_other):
        if isinstance(w_other, W_FloatObject):
            return W_FloatObject(self.floatvalue + w_other.floatvalue)
        else:
            return W_IntObject(0)   # tolerate poisoned placeholder operand (tier-1 blackhole)

    def sub_inline(self, w_other):
        if isinstance(w_other, W_FloatObject):
            return W_FloatObject(self.floatvalue - w_other.floatvalue)
        else:
            return W_IntObject(0)   # tolerate poisoned placeholder operand (tier-1 blackhole)

    def mul_inline(self, w_other):
        if isinstance(w_other, W_FloatObject):
            return W_FloatObject(self.floatvalue * w_other.floatvalue)
        else:
            return W_IntObject(0)   # tolerate poisoned placeholder operand (tier-1 blackhole)

    def div_inline(self, w_other):
        if isinstance(w_other, W_FloatObject):
            return W_FloatObject(self.floatvalue / w_other.floatvalue)
        else:
            return W_IntObject(0)   # tolerate poisoned placeholder operand (tier-1 blackhole)

    def mod_inline(self, w_other):
        return W_IntObject(0)   # float mod unsupported / poisoned placeholder (tier-1 blackhole)

    def eq_inline(self, w_other):
        if isinstance(w_other, W_FloatObject):
            if self.floatvalue == w_other.floatvalue:
                return W_IntObject(1)
            else:
                return W_IntObject(0)
        else:
            return W_IntObject(0)   # tolerate poisoned placeholder operand (tier-1 blackhole)

    def le_inline(self, w_other):
        if isinstance(w_other, W_FloatObject):
            if self.floatvalue <= w_other.floatvalue:
                return W_IntObject(1)
            else:
                return W_IntObject(0)
        else:
            return W_IntObject(0)   # tolerate poisoned placeholder operand (tier-1 blackhole)

    def ge_inline(self, w_other):
        if isinstance(w_other, W_FloatObject):
            if self.floatvalue >= w_other.floatvalue:
                return W_IntObject(1)
            else:
                return W_IntObject(0)
        else:
            return W_IntObject(0)   # tolerate poisoned placeholder operand (tier-1 blackhole)

    @enable_shallow_tracing
    def sqrt(self, flg=False):
        if flg:
            return W_IntObject(0)
        from math import sqrt
        return W_FloatObject(sqrt(self.floatvalue))

    @enable_shallow_tracing
    def add(self, w_other, flg=False):
        if flg:
            return W_IntObject(0)
        if isinstance(w_other, W_FloatObject):
            sum = self.floatvalue + w_other.floatvalue
            return W_FloatObject(sum)
        else:
            return W_IntObject(0)   # tolerate poisoned placeholder operand (tier-1 blackhole)

    @enable_shallow_tracing
    def sub(self, w_other, flg=False):
        if flg:
            return W_IntObject(0)
        if isinstance(w_other, W_FloatObject):
            sum = self.floatvalue - w_other.floatvalue
            return W_FloatObject(sum)
        else:
            return W_IntObject(0)   # tolerate poisoned placeholder operand (tier-1 blackhole)

    @enable_shallow_tracing
    def mul(self, w_other, flg=False):
        if flg:
            return W_IntObject(0)
        if isinstance(w_other, W_FloatObject):
            sum = self.floatvalue * w_other.floatvalue
            return W_FloatObject(sum)
        else:
            return W_IntObject(0)   # tolerate poisoned placeholder operand (tier-1 blackhole)

    @enable_shallow_tracing
    def div(self, w_other, flg=False):
        if flg:
            return W_IntObject(0)
        if isinstance(w_other, W_FloatObject):
            sum = self.floatvalue / w_other.floatvalue
            return W_FloatObject(sum)
        else:
            return W_IntObject(0)   # tolerate poisoned placeholder operand (tier-1 blackhole)

    @enable_shallow_tracing
    def mod(self, w_other, flg=False):
        if flg:
            return W_IntObject(0)
        return W_IntObject(0)   # float mod unsupported / poisoned placeholder (tier-1 blackhole)

    @enable_shallow_tracing
    def eq(self, w_other, flg=False):
        if flg:
            return W_IntObject(0)
        if isinstance(w_other, W_FloatObject):
            if self.floatvalue == w_other.floatvalue:
                return W_IntObject(1)
            else:
                return W_IntObject(0)
        else:
            return W_IntObject(0)   # tolerate poisoned placeholder operand (tier-1 blackhole)

    @enable_shallow_tracing
    def lt(self, w_other, flg=False):
        if flg:
            return W_IntObject(0)
        if isinstance(w_other, W_FloatObject):
            if self.floatvalue < w_other.floatvalue:
                return W_IntObject(1)
            else:
                return W_IntObject(0)
        else:
            return W_IntObject(0)   # tolerate poisoned placeholder operand (tier-1 blackhole)

    @enable_shallow_tracing
    def gt(self, w_other, flg=False):
        if flg:
            return W_IntObject(0)
        if isinstance(w_other, W_FloatObject):
            if self.floatvalue > w_other.floatvalue:
                return W_IntObject(1)
            else:
                return W_IntObject(0)
        else:
            return W_IntObject(0)   # tolerate poisoned placeholder operand (tier-1 blackhole)

    @enable_shallow_tracing
    def le(self, w_other, flg=False):
        if flg:
            return W_IntObject(0)
        if isinstance(w_other, W_FloatObject):
            if self.floatvalue <= w_other.floatvalue:
                return W_IntObject(1)
            else:
                return W_IntObject(0)
        else:
            return W_IntObject(0)   # tolerate poisoned placeholder operand (tier-1 blackhole)

    @enable_shallow_tracing
    def ge(self, w_other, flg=False):
        if flg:
            return W_IntObject(0)
        if isinstance(w_other, W_FloatObject):
            if self.floatvalue >= w_other.floatvalue:
                return W_IntObject(1)
            else:
                return W_IntObject(0)
        else:
            return W_IntObject(0)   # tolerate poisoned placeholder operand (tier-1 blackhole)

class W_StringObject(W_Object):

    def __init__(self, strvalue):
        self.strvalue = strvalue

    def getvalue(self):
        return self.strvalue

    def getrepr(self):
        return self.strvalue

    def is_true(self):
        return len(self.strvalue) != 0


class W_ListObject(W_Object):

    def __init__(self, listvalue):
        self.listvalue = listvalue

    def getvalue(self):
        return self.listvalue

    def getrepr(self):
        if we_are_translated():
            self.count += 1
            str = "<List %d>" % self.count
            return str
        return "<List %s>" % (id(self.listvalue))

    def is_true(self):
        return True


class W_RetAddrObject(W_Object):
    def __init__(self, addrvalue):
        self.addrvalue = addrvalue

    def getrepr(self):
        return str(self.addrvalue)
