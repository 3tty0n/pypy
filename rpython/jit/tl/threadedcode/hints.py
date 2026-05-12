"""Tracing-policy decorators for threaded-code handlers.

Two roles are distinguished:

* **Inner primitives** (``pop`` / ``push`` / ``take`` / ``drop`` / ``is_true``):
  always appear as residual ``call_n(handler_<name>, ...)`` ops in the trace.
  Decorate them with ``@enable_shallow_tracing`` (or one of its value-returning
  variants).

* **Outer bytecode handlers** (``ADD`` / ``SUB`` / ``CONST_INT`` / ...): when
  ``frame.jit_inline_budget > 0`` the JIT inlines the body into the trace, so
  the body's calls to inner primitives become residual ops. When the budget
  is exhausted the handler reverts to a single ``handler_<name>`` residual
  call, identical in shape to the shallow case. Decorate them with
  ``@enable_deep_tracing``.

The ``handler_<name>`` shim in every variant has the same ``(..., func,
dummy)`` tail-argument convention so ``OptTraceSplit._handle_dummy_flag``
can rewrite the residual call to a direct call to ``func`` after splitting.
"""
from rpython.rlib.objectmodel import always_inline
from rpython.rlib.jit import dont_look_inside, we_are_jitted

def enable_shallow_tracing(func):
    """Decorator for an "inner primitive" handler.

    The body of the handler appears as a single residual ``handler_<name>``
    call in the JIT trace. After ``OptTraceSplit`` runs, that residual call
    is rewritten to ``call(real_<name>, ...)`` by ``_handle_dummy_flag``
    (the ``(func, dummy)`` tail-arg pair is what it keys on).

    Implementation note: the shim deliberately uses the ``func`` argument
    passed at the call site (rather than the closure-captured ``func``).
    Otherwise RPython elides the unused parameter from the specialised
    shim's signature, which silently breaks
    ``CallDescr.get_calldescr_without_flag`` (it expects the second-to-last
    arg of the shim to be the ``Ptr`` func pointer to strip; if that's
    absent the heuristic strips a real handler arg by mistake and the
    rewritten call has the W_Object* in the function-pointer slot).
    """
    always_inline(func)  # tell RPython to inline

    @dont_look_inside
    def shallow_hanlder(*args):
        dummy = args[-1]
        f = args[-2]
        args = args[:-2]
        if dummy:
            return
        return f(*args)

    shallow_hanlder.func_name = "handler_" + func.func_name

    @always_inline
    def call_handler(*args):
        """Add dummy flag, which is placed at the last argument, to shallow_handler.
        When we_are_jitted and the frame has no inline budget, use shallow (dummy True).
        Otherwise run the full handler (one-step-deeper inlining when budget > 0).
        """
        frame = args[0]
        if we_are_jitted() and frame.jit_inline_budget <= 0:
            shallow_hanlder(*args + (func, True,))
        else:
            shallow_hanlder(*args + (func, False,))

    return call_handler


def enable_shallow_tracing_argn(argn):
    def enable_shallow_tracing(func):
        """Variant of ``enable_shallow_tracing`` that returns a value
        taken from the call's arguments when the dummy flag is true."""
        always_inline(func)  # tell RPython to inline

        @dont_look_inside
        def shallow_hanlder(*args):
            dummy = args[-1]
            f = args[-2]
            args = args[:-2]
            if dummy:
                return args[argn]
            return f(*args)

        shallow_hanlder.func_name = "handler_" + func.func_name

        @always_inline
        def call_handler(*args):
            frame = args[0]
            if we_are_jitted() and frame.jit_inline_budget <= 0:
                return shallow_hanlder(*args + (func, True,))
            else:
                return shallow_hanlder(*args + (func, False,))

        return call_handler

    return enable_shallow_tracing


def enable_shallow_tracing_with_value(value):
    def enable_shallow_tracing(func):
        """Variant of ``enable_shallow_tracing`` that returns a fixed
        default value when the dummy flag is true."""
        always_inline(func)  # tell RPython to inline

        @dont_look_inside
        def shallow_hanlder(*args):
            dummy = args[-1]
            f = args[-2]
            args = args[:-2]
            if dummy:
                return value
            return f(*args)

        shallow_hanlder.func_name = "handler_" + func.func_name

        @always_inline
        def call_handler(*args):
            frame = args[0]
            if we_are_jitted() and frame.jit_inline_budget <= 0:
                return shallow_hanlder(*args + (func, True,))
            else:
                return shallow_hanlder(*args + (func, False,))

        return call_handler

    return enable_shallow_tracing


def enable_deep_tracing(func):
    """Outer bytecode handler decorator: one level deeper than shallow.

    Behaviour while the JIT is tracing:

    * ``frame.jit_inline_budget > 0`` — call ``func`` directly. The JIT
      inlines the body into the trace. Inner ``@enable_shallow_tracing``
      calls inside ``func`` (e.g. ``pop`` / ``push``) keep producing
      residual ``handler_<inner>`` ops, which the trace splitter rewrites
      to direct calls. End result: a trace shape like

          v0 = call(real_pop, frame)
          v1 = call(real_pop, frame)
          v2 = int_add(v0, v1)
          call_n(real_push, frame, v2)

    * ``frame.jit_inline_budget <= 0`` — call through a ``handler_<name>``
      shim that runs the body once at trace time (so subsequent ops see
      the post-handler frame state). The trace records a single residual
      ``call_n(handler_<name>, ..., func, dummy=False)`` which the trace
      splitter rewrites into ``call(real_<name>, ...)``.

    Outside the JIT the body just runs.

    See ``enable_shallow_tracing`` for the reason the shim references the
    ``func`` argument passed at the call site rather than the closure
    capture.
    """
    always_inline(func)  # tell RPython to inline

    @dont_look_inside
    def shallow_hanlder(*args):
        # Same shape as ``enable_shallow_tracing``'s shim — the trailing
        # ``(func, dummy)`` pair is what ``_handle_dummy_flag`` keys on.
        # We never pass ``dummy=True`` here: the residual path must still
        # mutate the frame at trace time so the rest of the recording
        # picks up correct stack state.
        dummy = args[-1]
        f = args[-2]
        args = args[:-2]
        if dummy:
            return
        return f(*args)

    shallow_hanlder.func_name = "handler_" + func.func_name

    @always_inline
    def call_handler(*args):
        frame = args[0]
        if we_are_jitted() and frame.jit_inline_budget <= 0:
            # Residual: single call op, body runs in the shim at trace time.
            shallow_hanlder(*args + (func, False,))
        else:
            # Deep: JIT traces directly into the body. Any inner
            # @enable_shallow_tracing call there stays residual via its
            # own shim.
            func(*args)

    return call_handler


def enable_deep_tracing_with_value(value):
    """Value-returning counterpart of ``enable_deep_tracing`` — kept for
    parity with the shallow family. The ``value`` argument is the
    default returned by the residual shim when its ``dummy`` arg is
    True; with the deep decorator we never pass ``dummy=True``, so it is
    only used to match the shallow signature."""
    def deco(func):
        always_inline(func)

        @dont_look_inside
        def shallow_hanlder(*args):
            dummy = args[-1]
            f = args[-2]
            args = args[:-2]
            if dummy:
                return value
            return f(*args)

        shallow_hanlder.func_name = "handler_" + func.func_name

        @always_inline
        def call_handler(*args):
            frame = args[0]
            if we_are_jitted() and frame.jit_inline_budget <= 0:
                return shallow_hanlder(*args + (func, False,))
            else:
                return func(*args)

        return call_handler

    return deco
