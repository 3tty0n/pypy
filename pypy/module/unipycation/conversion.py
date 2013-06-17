import prolog.interpreter.term as pterm

def _type_check(space, inst, typ):
    # XXX do the right thing with the exception
    if not space.is_true(space.isinstance(inst, typ)):
        raise TypeError("%s is not of type %s" % (inst, typ))

# -----------------------------
# Convert from Python to Prolog
# -----------------------------

def p_int_of_w_int(space, w_int):
    _type_check(space, w_int, space.w_int)

    val = space.int_w(w_int)
    return pterm.Number(val)

def p_float_of_w_float(space, w_float):
    _type_check(space, w_float, space.w_float)

    val = space.float_w(w_float)
    return pterm.Float(val)

def p_bigint_of_w_long(space, w_long):
    _type_check(space, w_long, space.w_long)

    val = space.bigint_w(w_long)
    return pterm.BigInt(val)

def p_atom_of_w_str(space, w_str):
    _type_check(space, w_str, space.w_str)

    val = space.str_w(w_str)
    return pterm.Atom(val)

# -----------------------------
# Convert from Prolog to Python
# -----------------------------

def w_int_of_p_int(space, p_int):
    # XXX type check
    return space.newint(p_int.num)

def w_float_of_p_float(space, p_float):
    # XXX type check
    return space.newfloat(p_float.floatval)

def w_long_of_p_bigint(space, p_bigint):
    # XXX type check
    return space.newlong_from_rbigint(p_bigint.value)

def w_str_of_p_atom(space, p_atom):
    # XXX type check
    return space.wrap(p_atom._signature.name)
