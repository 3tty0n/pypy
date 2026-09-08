"""top is a ref threaded as its own boundary name, not on a virtualizable."""

from rpython.rlib.pe import PEDriver
from rpython.rlib.objectmodel import always_inline
from rpython.rlib.jit import JitDriver


class W_IntObject(object):

    def __init__(self, intvalue):
        self.intvalue = intvalue

    def is_true(self):
        return self.intvalue != 0

    def sub(self, w_other):
        return W_IntObject(self.intvalue - w_other.intvalue)


OPNAMES = []
HASARG = []


def define_op(name, has_arg=False):
    globals()[name] = len(OPNAMES)
    OPNAMES.append(name)
    HASARG.append(has_arg)


define_op("CONST_INT", True)
define_op("SUB", True)
define_op("JUMP_IF", True)
define_op("RETURN")


def get_printable_location(pc, bytecode):
    op = ord(bytecode[pc])
    name = OPNAMES[op]
    if HASARG[op]:
        arg = str(ord(bytecode[pc + 1]))
    else:
        arg = ''
    return "%s: %s %s" % (pc, name, arg)


jitdriver = JitDriver(greens=['pc', 'bytecode'],
                      reds=['self', 'top'],
                      virtualizables=['self'],
                      get_printable_location=get_printable_location)

pedriver = PEDriver(static="opcode", split="pc")


class Frame(object):
    """_unused keeps Frame virtualizable; no field here holds real state."""
    _virtualizable_ = ['_unused']

    def __init__(self, bytecode):
        self.bytecode = bytecode
        self._unused = 0

    def interp(self, top):
        bytecode = self.bytecode
        pc = 0
        while pc < len(bytecode):
            jitdriver.jit_merge_point(bytecode=bytecode, pc=pc, self=self,
                                      top=top)
            opcode = ord(bytecode[pc])
            pc += 1
            if HASARG[opcode]:
                oparg = ord(bytecode[pc])
                pc += 1
            else:
                oparg = 0

            pc, w_result, frame, top = self.interp_step(
                bytecode, opcode, oparg, pc, top)
            assert frame is self
            if w_result is not None:
                return w_result

    @always_inline
    def interp_step(self, bytecode, opcode, oparg, pc, top):
        pedriver.pe_merge_point(self=self, bytecode=bytecode, opcode=opcode,
                                oparg=oparg, pc=pc, top=top)
        if opcode == CONST_INT:
            top = W_IntObject(oparg)

        elif opcode == SUB:
            top = top.sub(W_IntObject(oparg))

        elif opcode == JUMP_IF:
            if top.is_true():
                jitdriver.can_enter_jit(
                    bytecode=bytecode, pc=oparg, self=self, top=top)
                return oparg, None, self, top
            return pc, None, self, top

        elif opcode == RETURN:
            return -1, top, self, top

        else:
            assert False, 'Unknown opcode: %d' % opcode

        return pc, None, self, top


def run(bytecode, intvalue):
    frame = Frame(bytecode)
    return frame.interp(W_IntObject(intvalue))
