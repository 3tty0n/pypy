import os

BENCHMARKS_OWN_MACRO = [
    "bm_dulwich_log",
    "sqlalchemy_declarative",
    "sqlalchemy_imperative",
    "bm_gzip",
    "bm_krakatau",
    "bm_mdp",
    "bm_gzip",
    "bm_sympy",
    "go",
    "pyxl_bench",
    "pypy_interp",
    "eparse",
    "bm_icbd",
]

BENCHMARKS_OWN_MICRO = [
    "bm_chameleon",
    "bm_genshi",
    "crypto_pyaes",
    "deltablue",
    "fannkuch",
    "fib",
    "meteor-contest",
    "nbody_modified",
    "raytrace-simple",
    "spectral-norm",
    "spitfire",
    "sqlitesynth",
    "hexiom2",
    "json_bench",
]

BENCHMARKS_UNLADEN_SWALLOW = [
    "bm_django",
    "bm_html5lib",
    "bm_richards",
    "bm_spambayes",
    "bm_unpack_sequence",
]


def setup_env_own():
    env = os.environ.copy()
    env["PYTHONPATH"] = ":".join(
        [
            "benchmarks/lib/" + x
            for x in [
                "chameleon/src",
                "dulwich-0.19.13",
                "jinja2",
                "pyxl",
                "monte",
                "pytz",
                "mako",
                "sqlalchemy/lib",
                "sympy",
                "genshi",
                "twisted-trunk/twisted",
                "pypy",
            ]
        ]
    )
    return env


def setup_env_unladen():
    own_env = setup_env_own()
    env = os.environ.copy()
    env["PYTHONPATH"] = ":".join(
        [
            "benchmarks/unladen_swallow/lib/" + x
            for x in ["django", "html5lib", "spambayes", "spitfire", "lockfile"]
        ]
    )
    env["PYTHONPATH"] = "benchmarks/lib:benchmarks/lib/pytz:" + own_env["PYTHONPATH"]
    env["PYTHONHASHSEED"] = "0"
    env.setdefault("LC_ALL", "C")
    return env


def setup_env(typ):
    if typ in ("own", "own-macro", "own_micro"):
        return setup_env_own()
    elif typ == "unladen_swallow":
        return setup_env_unladen()
    else:
        raise Exception("unrachable path")


def setup_bm_path(typ):
    if typ == "own":
        return "benchmarks/own/"
    elif typ == "own-macro":
        return "benchmarks/own/"
    elif typ == "own-micro":
        return "benchmarks/own/"
    elif typ == "unladen_swallow":
        return "benchmarks/unladen_swallow/performance/"
    else:
        raise Exception("unreachable path")


def setup_bms(typ):
    if typ == "own":
        return BENCHMARKS_OWN_MICRO + BENCHMARKS_OWN_MACRO
    elif typ == "own-macro":
        return BENCHMARKS_OWN_MACRO
    elif typ == "own-micro":
        return BENCHMARKS_OWN_MICRO
    else:
        raise Exception("unreachable path")


COMMANDS = [
    ("pypy-c", "./pypy/goal/pypy-c"),
    ("pypy-jit-ext-c", "./pypy/goal/pypy-jit-ext-c"),
]
