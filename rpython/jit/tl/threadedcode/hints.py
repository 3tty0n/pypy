from rlib.objectmodel import always_inline
from rlib.jit import dont_look_inside, we_are_jitted

def enable_shallow_tracing(func):
    "A decorator to enable an actual handler to do shallow tracing"
    always_inline(func)  # tell RPython to inline

    @dont_look_inside
    def shallow_hanlder(*args):
        dummy = args[-1]
        args = args[:-2]
        if dummy:
            return
        return func(*args)

    shallow_hanlder.func_name = "handler_" + func.func_name

    @always_inline
    def call_handler(*args):
        """Add dummy flag, which is placed at the last argument, to shallow_handler.
        When we_are_jitted returns True, add True to the dummy flag. Otherwise,
        pass False to the flag.
        """
        if we_are_jitted():
            shallow_hanlder(*args + (func, True,))
        else:
            shallow_hanlder(*args + (None, False,))

    return call_handler


def enable_shallow_tracing_argn(argn):
    def enable_shallow_tracing(func):
        """
        A decorator to enable an actual handler to do shallow tracing.
        Use this decorator for a function that returns a value, which is
        at `argn' of args.
        """
        always_inline(func)  # tell RPython to inline

        @dont_look_inside
        def shallow_hanlder(*args):
            dummy = args[-1]
            args = args[:-2]
            if dummy:
                return args[argn]
            return func(*args)

        shallow_hanlder.func_name = "handler_" + func.func_name

        @always_inline
        def call_handler(*args):
            if we_are_jitted():
                return shallow_hanlder(*args + (func, True,))
            else:
                return shallow_hanlder(*args + (None, False,))

        return call_handler

    return enable_shallow_tracing


def enable_shallow_tracing_with_value(value):
    def enable_shallow_tracing(func):
        """
        A decorator to enable an actual handler to do shallow tracing.
        Use this decorator for a function that returns a value, which is
        at `argn' of args.
        """
        always_inline(func)  # tell RPython to inline

        @dont_look_inside
        def shallow_hanlder(*args):
            dummy = args[-1]
            args = args[:-2]
            if dummy:
                return value
            return func(*args)

        shallow_hanlder.func_name = "handler_" + func.func_name

        @always_inline
        def call_handler(*args):
            if we_are_jitted():
                return shallow_hanlder(*args + (func, True,))
            else:
                return shallow_hanlder(*args + (None, False,))

        return call_handler

    return enable_shallow_tracing
