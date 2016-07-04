import sys
from rpython.rlib import revdb
from rpython.rlib.debug import make_sure_not_resized
from rpython.rlib.objectmodel import specialize, we_are_translated
from rpython.rtyper.annlowlevel import cast_gcref_to_instance
from pypy.interpreter.error import OperationError, oefmt
from pypy.interpreter.baseobjspace import W_Root
from pypy.interpreter import gateway, typedef, pycode, pytraceback, pyframe
from pypy.module.marshal import interp_marshal
from pypy.interpreter.executioncontext import AbstractActionFlag


class DBState:
    standard_code = True
    breakpoint_stack_id = 0
    breakpoint_funcnames = None
    printed_objects = {}
    metavars = []
    watch_progs = []
    watch_futures = {}

dbstate = DBState()


pycode.PyCode.co_revdb_linestarts = None   # or a string: see below

# invariant: "f_revdb_nextline_instr" is the bytecode offset of
# the start of the line that follows "last_instr".
pyframe.PyFrame.f_revdb_nextline_instr = -1


# ____________________________________________________________


def setup_revdb(space):
    """Called at run-time, before the space is set up.

    The various register_debug_command() lines attach functions
    to some commands that 'revdb.py' can call, if we are running
    in replay mode.
    """
    assert space.config.translation.reverse_debugger
    dbstate.space = space
    dbstate.w_future = space.w_Ellipsis    # a random prebuilt object

    make_sure_not_resized(dbstate.watch_progs)
    make_sure_not_resized(dbstate.metavars)

    revdb.register_debug_command(revdb.CMD_PRINT, lambda_print)
    revdb.register_debug_command(revdb.CMD_BACKTRACE, lambda_backtrace)
    revdb.register_debug_command(revdb.CMD_LOCALS, lambda_locals)
    revdb.register_debug_command(revdb.CMD_BREAKPOINTS, lambda_breakpoints)
    revdb.register_debug_command(revdb.CMD_STACKID, lambda_stackid)
    revdb.register_debug_command("ALLOCATING", lambda_allocating)
    revdb.register_debug_command(revdb.CMD_ATTACHID, lambda_attachid)
    revdb.register_debug_command(revdb.CMD_COMPILEWATCH, lambda_compilewatch)
    revdb.register_debug_command(revdb.CMD_CHECKWATCH, lambda_checkwatch)
    revdb.register_debug_command(revdb.CMD_WATCHVALUES, lambda_watchvalues)


# ____________________________________________________________


def enter_call(caller_frame, callee_frame):
    if dbstate.breakpoint_funcnames is not None:
        name = callee_frame.getcode().co_name
        if name in dbstate.breakpoint_funcnames:
            revdb.breakpoint(dbstate.breakpoint_funcnames[name])
    if dbstate.breakpoint_stack_id != 0 and caller_frame is not None:
        if dbstate.breakpoint_stack_id == revdb.get_unique_id(caller_frame):
            revdb.breakpoint(-1)
    #
    code = callee_frame.pycode
    if code.co_revdb_linestarts is None:
        build_co_revdb_linestarts(code)

def leave_call(caller_frame, callee_frame):
    if dbstate.breakpoint_stack_id != 0 and caller_frame is not None:
        if dbstate.breakpoint_stack_id == revdb.get_unique_id(caller_frame):
            revdb.breakpoint(-2)


def jump_backward(frame, jumpto):
    # When we see a jump backward, we set 'f_revdb_nextline_instr' in
    # such a way that the next instruction, at 'jumpto', will trigger
    # stop_point_at_start_of_line().  We have to trigger it even if
    # 'jumpto' is not actually a start of line.  For example, in a
    # 'while foo: body', the body ends with a JUMP_ABSOLUTE which
    # jumps back to the *second* opcode of the while.
    frame.f_revdb_nextline_instr = jumpto


def potential_stop_point(frame):
    if not we_are_translated():
        return
    #
    # We only record a stop_point at every line, not every bytecode.
    # Uses roughly the same algo as ExecutionContext.run_trace_func()
    # to know where the line starts are, but tweaked for speed,
    # avoiding the quadratic complexity when run N times with a large
    # code object.
    #
    cur = frame.last_instr
    if cur < frame.f_revdb_nextline_instr:
        return    # fast path: we're still inside the same line as before
    #
    call_stop_point_at_line = True
    co_revdb_linestarts = frame.pycode.co_revdb_linestarts
    if cur > frame.f_revdb_nextline_instr:
        #
        # We jumped forward over the start of the next line.  We're
        # inside a different line, but we will only trigger a stop
        # point if we're at the starting bytecode of that line.  Fetch
        # from co_revdb_linestarts the start of the line that is at or
        # follows 'cur'.
        ch = ord(co_revdb_linestarts[cur])
        if ch == 0:
            pass   # we are at the start of a line now
        else:
            # We are not, so don't call stop_point_at_start_of_line().
            # We still have to fill f_revdb_nextline_instr.
            call_stop_point_at_line = False
    #
    if call_stop_point_at_line:
        stop_point_at_start_of_line()
        cur += 1
        ch = ord(co_revdb_linestarts[cur])
    #
    # Update f_revdb_nextline_instr.  Check if 'ch' was greater than
    # 255, in which case it was rounded down to 255 and we have to
    # continue looking
    nextline_instr = cur + ch
    while ch == 255:
        ch = ord(co_revdb_linestarts[nextline_instr])
        nextline_instr += ch
    frame.f_revdb_nextline_instr = nextline_instr


def build_co_revdb_linestarts(code):
    # Inspired by findlinestarts() in the 'dis' standard module.
    # Set up 'bits' so that it contains \x00 at line starts and \xff
    # in-between.
    bits = ['\xff'] * (len(code.co_code) + 1)
    if not code.hidden_applevel:
        lnotab = code.co_lnotab
        addr = 0
        p = 0
        newline = 1
        while p + 1 < len(lnotab):
            byte_incr = ord(lnotab[p])
            line_incr = ord(lnotab[p+1])
            if byte_incr:
                if newline != 0:
                    bits[addr] = '\x00'
                    newline = 0
                addr += byte_incr
            newline |= line_incr
            p += 2
        if newline:
            bits[addr] = '\x00'
    bits[len(code.co_code)] = '\x00'
    #
    # Change 'bits' so that the character at 'i', if not \x00, measures
    # how far the next \x00 is
    next_null = len(code.co_code)
    p = next_null - 1
    while p >= 0:
        if bits[p] == '\x00':
            next_null = p
        else:
            ch = next_null - p
            if ch > 255: ch = 255
            bits[p] = chr(ch)
        p -= 1
    lstart = ''.join(bits)
    code.co_revdb_linestarts = lstart
    return lstart

def get_final_lineno(code):
    lineno = code.co_firstlineno
    lnotab = code.co_lnotab
    p = 1
    while p < len(lnotab):
        line_incr = ord(lnotab[p])
        lineno += line_incr
        p += 2
    return lineno


class NonStandardCode(object):
    def __enter__(self):
        dbstate.standard_code = False
        self.t = dbstate.space.actionflag._ticker
        self.c = dbstate.space.actionflag._ticker_count
    def __exit__(self, *args):
        dbstate.space.actionflag._ticker = self.t
        dbstate.space.actionflag._ticker_count = self.c
        dbstate.standard_code = True
non_standard_code = NonStandardCode()


def stop_point_at_start_of_line():
    if revdb.watch_save_state():
        any_watch_point = False
        space = dbstate.space
        with non_standard_code:
            for prog, watch_id, expected in dbstate.watch_progs:
                any_watch_point = True
                try:
                    got = _run_watch(space, prog)
                except OperationError as e:
                    got = e.errorstr(space)
                except Exception:
                    break
                if got != expected:
                    break
            else:
                watch_id = -1
        revdb.watch_restore_state(any_watch_point)
        if watch_id != -1:
            revdb.breakpoint(watch_id)
    revdb.stop_point()


def load_metavar(index):
    assert index >= 0
    space = dbstate.space
    metavars = dbstate.metavars
    w_var = metavars[index] if index < len(metavars) else None
    if w_var is None:
        raise oefmt(space.w_NameError, "no constant object '$%d'",
                    index)
    if w_var is dbstate.w_future:
        raise oefmt(space.w_RuntimeError,
                    "'$%d' refers to an object created later in time",
                    index)
    return w_var

def set_metavar(index, w_obj):
    assert index >= 0
    if index >= len(dbstate.metavars):
        missing = index + 1 - len(dbstate.metavars)
        dbstate.metavars = dbstate.metavars + [None] * missing
    dbstate.metavars[index] = w_obj


# ____________________________________________________________


def fetch_cur_frame():
    ec = dbstate.space.getexecutioncontext()
    frame = ec.topframeref()
    if frame is None:
        revdb.send_output("No stack.\n")
    return frame

def compile(source, mode):
    space = dbstate.space
    compiler = space.createcompiler()
    code = compiler.compile(source, '<revdb>', mode, 0,
                            hidden_applevel=True)
    return code


class W_RevDBOutput(W_Root):
    softspace = 0

    def __init__(self, space):
        self.space = space

    def descr_write(self, w_buffer):
        space = self.space
        if space.isinstance_w(w_buffer, space.w_unicode):
            w_buffer = space.call_method(w_buffer, 'encode',
                                         space.wrap('utf-8'))   # safe?
        revdb.send_output(space.str_w(w_buffer))

def descr_get_softspace(space, revdb):
    return space.wrap(revdb.softspace)
def descr_set_softspace(space, revdb, w_newvalue):
    revdb.softspace = space.int_w(w_newvalue)

W_RevDBOutput.typedef = typedef.TypeDef(
    "revdb_output",
    write = gateway.interp2app(W_RevDBOutput.descr_write),
    softspace = typedef.GetSetProperty(descr_get_softspace,
                                       descr_set_softspace,
                                       cls=W_RevDBOutput),
    )

def revdb_displayhook(space, w_obj):
    """Modified sys.displayhook() that also outputs '$NUM = ',
    for non-prebuilt objects.  Such objects are then recorded in
    'printed_objects'.
    """
    if space.is_w(w_obj, space.w_None):
        return
    uid = revdb.get_unique_id(w_obj)
    if uid > 0:
        dbstate.printed_objects[uid] = w_obj
        revdb.send_nextnid(uid)   # outputs '$NUM = '
    space.setitem(space.builtin.w_dict, space.wrap('_'), w_obj)
    # do str_w(repr()) only now: if w_obj was produced successfully,
    # but its repr crashes because it tries to do I/O, then we already
    # have it recorded in '_' and in '$NUM ='.
    s = space.str_w(space.repr(w_obj))
    revdb.send_output(s)
    revdb.send_output("\n")

@specialize.memo()
def get_revdb_displayhook(space):
    return space.wrap(gateway.interp2app(revdb_displayhook))


def prepare_print_environment(space):
    w_revdb_output = space.wrap(W_RevDBOutput(space))
    w_displayhook = get_revdb_displayhook(space)
    space.sys.setdictvalue(space, 'stdout', w_revdb_output)
    space.sys.setdictvalue(space, 'stderr', w_revdb_output)
    space.sys.setdictvalue(space, 'displayhook', w_displayhook)

def command_print(cmd, expression):
    frame = fetch_cur_frame()
    if frame is None:
        return
    space = dbstate.space
    with non_standard_code:
        try:
            prepare_print_environment(space)
            code = compile(expression, 'single')
            try:
                code.exec_code(space,
                               frame.get_w_globals(),
                               frame.getdictscope())

            except OperationError as operationerr:
                # can't use sys.excepthook: it will likely try to do 'import
                # traceback', which might not be doable without using I/O
                tb = operationerr.get_traceback()
                if tb is not None:
                    revdb.send_output("Traceback (most recent call last):\n")
                    while tb is not None:
                        if not isinstance(tb, pytraceback.PyTraceback):
                            revdb.send_output("  ??? %s\n" % tb)
                            break
                        show_frame(tb.frame, tb.get_lineno(), indent='  ')
                        tb = tb.next
                revdb.send_output('%s\n' % operationerr.errorstr(space))

                # set the sys.last_xxx attributes
                w_type = operationerr.w_type
                w_value = operationerr.get_w_value(space)
                w_tb = space.wrap(operationerr.get_traceback())
                w_dict = space.sys.w_dict
                space.setitem(w_dict, space.wrap('last_type'), w_type)
                space.setitem(w_dict, space.wrap('last_value'), w_value)
                space.setitem(w_dict, space.wrap('last_traceback'), w_tb)

        except OperationError as e:
            revdb.send_output('%s\n' % e.errorstr(space, use_repr=True))
lambda_print = lambda: command_print


def file_and_lineno(frame, lineno):
    code = frame.getcode()
    return 'File "%s", line %d in %s' % (
        code.co_filename, lineno, code.co_name)

def show_frame(frame, lineno=0, indent=''):
    if lineno == 0:
        lineno = frame.get_last_lineno()
    revdb.send_output("%s%s\n%s  " % (
        indent,
        file_and_lineno(frame, lineno),
        indent))
    revdb.send_linecache(frame.getcode().co_filename, lineno)

def display_function_part(frame, max_lines_before, max_lines_after):
    code = frame.getcode()
    if code.co_filename.startswith('<builtin>'):
        return
    first_lineno = code.co_firstlineno
    current_lineno = frame.get_last_lineno()
    final_lineno = get_final_lineno(code)
    #
    ellipsis_after = False
    if first_lineno < current_lineno - max_lines_before - 1:
        first_lineno = current_lineno - max_lines_before
        revdb.send_output("...\n")
    if final_lineno > current_lineno + max_lines_after + 1:
        final_lineno = current_lineno + max_lines_after
        ellipsis_after = True
    #
    for i in range(first_lineno, final_lineno + 1):
        if i == current_lineno:
            revdb.send_output("> ")
        else:
            revdb.send_output("  ")
        revdb.send_linecache(code.co_filename, i, strip=False)
    #
    if ellipsis_after:
        revdb.send_output("...\n")

def command_backtrace(cmd, extra):
    frame = fetch_cur_frame()
    if frame is None:
        return
    if cmd.c_arg1 == 0:
        revdb.send_output("%s:\n" % (
            file_and_lineno(frame, frame.get_last_lineno()),))
        display_function_part(frame, max_lines_before=8, max_lines_after=5)
    elif cmd.c_arg1 == 2:
        display_function_part(frame, max_lines_before=1000,max_lines_after=1000)
    else:
        revdb.send_output("Current call stack (most recent call last):\n")
        frames = []
        while frame is not None:
            frames.append(frame)
            if len(frames) == 200:
                revdb.send_output("  ...\n")
                break
            frame = frame.get_f_back()
        while len(frames) > 0:
            show_frame(frames.pop(), indent='  ')
lambda_backtrace = lambda: command_backtrace


def command_locals(cmd, extra):
    frame = fetch_cur_frame()
    if frame is None:
        return
    space = dbstate.space
    try:
        prepare_print_environment(space)
        space.appexec([space.wrap(space.sys),
                       frame.getdictscope()], """(sys, locals):
            lst = locals.keys()
            lst.sort()
            print 'Locals:'
            for key in lst:
                try:
                    print '    %s =' % key,
                    s = '%r' % locals[key]
                    if len(s) > 140:
                        s = s[:100] + '...' + s[-30:]
                    print s
                except:
                    exc, val, tb = sys.exc_info()
                    print '!<%s: %r>' % (exc, val)
        """)
    except OperationError as e:
        revdb.send_output('%s\n' % e.errorstr(space, use_repr=True))
lambda_locals = lambda: command_locals


def command_breakpoints(cmd, extra):
    space = dbstate.space
    dbstate.breakpoint_stack_id = cmd.c_arg1
    funcnames = None
    watch_progs = []
    for i, kind, name in revdb.split_breakpoints_arg(extra):
        if kind == 'B':
            if funcnames is None:
                funcnames = {}
            funcnames[name] = i
        elif kind == 'W':
            code = interp_marshal.loads(space, space.wrap(name))
            watch_progs.append((code, i, ''))
    dbstate.breakpoint_funcnames = funcnames
    dbstate.watch_progs = watch_progs[:]
lambda_breakpoints = lambda: command_breakpoints


def command_watchvalues(cmd, extra):
    expected = extra.split('\x00')
    for j in range(len(dbstate.watch_progs)):
        prog, i, _ = dbstate.watch_progs[j]
        if i >= len(expected):
            raise IndexError
        dbstate.watch_progs[j] = prog, i, expected[i]
lambda_watchvalues = lambda: command_watchvalues


def command_stackid(cmd, extra):
    frame = fetch_cur_frame()
    if frame is not None and cmd.c_arg1 != 0:     # parent_flag
        frame = dbstate.space.getexecutioncontext().getnextframe_nohidden(frame)
    if frame is None:
        uid = 0
    else:
        uid = revdb.get_unique_id(frame)
    revdb.send_answer(revdb.ANSWER_STACKID, uid)
lambda_stackid = lambda: command_stackid


def command_allocating(uid, gcref):
    w_obj = cast_gcref_to_instance(W_Root, gcref)
    dbstate.printed_objects[uid] = w_obj
    try:
        index_metavar = dbstate.watch_futures.pop(uid)
    except KeyError:
        pass
    else:
        set_metavar(index_metavar, w_obj)
lambda_allocating = lambda: command_allocating


def command_attachid(cmd, extra):
    space = dbstate.space
    index_metavar = cmd.c_arg1
    uid = cmd.c_arg2
    try:
        w_obj = dbstate.printed_objects[uid]
    except KeyError:
        # uid not found, probably a future object
        dbstate.watch_futures[uid] = index_metavar
        w_obj = dbstate.w_future
    set_metavar(index_metavar, w_obj)
lambda_attachid = lambda: command_attachid


def command_compilewatch(cmd, expression):
    space = dbstate.space
    with non_standard_code:
        try:
            code = compile(expression, 'eval')
            marshalled_code = space.str_w(interp_marshal.dumps(
                space, space.wrap(code),
                space.wrap(interp_marshal.Py_MARSHAL_VERSION)))
        except OperationError as e:
            revdb.send_watch(e.errorstr(space), ok_flag=0)
        else:
            revdb.send_watch(marshalled_code, ok_flag=1)
lambda_compilewatch = lambda: command_compilewatch

def command_checkwatch(cmd, marshalled_code):
    space = dbstate.space
    with non_standard_code:
        try:
            code = interp_marshal.loads(space, space.wrap(marshalled_code))
            text = _run_watch(space, code)
        except OperationError as e:
            revdb.send_watch(e.errorstr(space), ok_flag=0)
        else:
            revdb.send_watch(text, ok_flag=1)
lambda_checkwatch = lambda: command_checkwatch


def _run_watch(space, prog):
    w_dict = space.builtin.w_dict
    w_res = prog.exec_code(space, w_dict, w_dict)
    return space.str_w(space.repr(w_res))


# ____________________________________________________________


class RDBSignalActionFlag(AbstractActionFlag):
    # Used instead of pypy.module.signal.interp_signal.SignalActionFlag
    # when we have reverse-debugging.  That other class would work too,
    # but inefficiently: it would generate two words of data per bytecode.
    # This class is tweaked to generate one byte per _SIG_TICKER_COUNT
    # bytecodes, at the expense of not reacting to signals instantly.

    _SIG_TICKER_COUNT = 100
    _ticker = 0
    _ticker_count = _SIG_TICKER_COUNT * 10

    def get_ticker(self):
        return self._ticker

    def reset_ticker(self, value):
        self._ticker = value

    def rearm_ticker(self):
        self._ticker = -1

    def decrement_ticker(self, by):
        if we_are_translated():
            c = self._ticker_count - 1
            if c < 0:
                c = self._update_ticker_from_signals()
            self._ticker_count = c
        if self.has_bytecode_counter:    # this 'if' is constant-folded
            print ("RDBSignalActionFlag: has_bytecode_counter: "
                   "not supported for now")
            raise NotImplementedError
        return self._ticker

    def _update_ticker_from_signals(self):
        from rpython.rlib import rsignal
        if dbstate.standard_code:
            if rsignal.pypysig_check_and_reset():
                self.rearm_ticker()
        return self._SIG_TICKER_COUNT
    _update_ticker_from_signals._dont_inline_ = True
