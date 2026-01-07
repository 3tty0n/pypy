import os

BENCHMARKS_OWN_MACRO = [
    "bm_chameleon",
    "bm_genshi",
    "bm_dulwich_log",
    "sqlalchemy_declarative",
    "sqlalchemy_imperative",
    "sqlitesynth",
    "bm_gzip",
    "bm_krakatau",
    "bm_mdp",
    "bm_sympy",
    "go",
    "pyxl_bench",
    "pypy_interp",
    "eparse",
    "spitfire",
    "bm_icbd",
]

BENCHMARKS_OWN_MICRO = [
    "crypto_pyaes",
    "deltablue",
    "fannkuch",
    "fib",
    "meteor-contest",
    "nbody_modified",
    "raytrace-simple",
    "spectral-norm",
    #"hexiom2",
    "json_bench",
]

BENCHMARKS_OWN_ALL = BENCHMARKS_OWN_MACRO + BENCHMARKS_OWN_MICRO

BENCHMARKS_UNLADEN_SWALLOW = [
    "bm_django",
    "bm_html5lib",
    "bm_richards",
    "bm_spambayes",
]

BENCHMARKS_ALL_MACRO = BENCHMARKS_OWN_MACRO + BENCHMARKS_UNLADEN_SWALLOW

BENCHMARKS_ALL = BENCHMARKS_OWN_MICRO + BENCHMARKS_OWN_MACRO + BENCHMARKS_UNLADEN_SWALLOW

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
    path = ":".join([
            "benchmarks/unladen_swallow/lib/" + x
            for x in ["django", "html5lib", "spambayes", "spitfire", "lockfile"]
        ])
    env["PYTHONPATH"] = "benchmarks/lib:benchmarks/lib/pytz:" + path
    env["PYTHONHASHSEED"] = "0"
    env.setdefault("LC_ALL", "C")
    return env


def setup_env(typ):
    if typ in ("own", "own-macro", "own-micro"):
        return setup_env_own()
    elif typ == "unladen":
        return setup_env_unladen()
    else:
        raise Exception("unrachable path")


def setup_bm_path(typ):
    if typ in ("own", "own-macro", "own-micro"):
        return "benchmarks/own/"
    elif typ == "unladen":
        return "benchmarks/unladen_swallow/performance/"
    else:
        raise Exception("unreachable path")


def setup_bms(typ):
    if typ == "own":
        return BENCHMARKS_OWN_ALL
    elif typ == "own-macro":
        return BENCHMARKS_OWN_MACRO
    elif typ == "own-micro":
        return BENCHMARKS_OWN_MICRO
    elif typ == "unladen":
        return BENCHMARKS_UNLADEN_SWALLOW
    else:
        raise Exception("unreachable path")


def setup_bms_plot(typ):
    if typ == "own":
        return BENCHMARKS_OWN_ALL
    elif typ == "own-macro":
        return BENCHMARKS_OWN_MACRO
    elif typ == "own-micro":
        return BENCHMARKS_OWN_MICRO
    elif typ == "unladen":
        return BENCHMARKS_UNLADEN_SWALLOW
    elif typ == "macro-all":
        return BENCHMARKS_ALL_MACRO
    elif typ == "all":
        return BENCHMARKS_ALL
    else:
        raise Exception("unreachable path")


COMMANDS = [
    ("pypy-c", "./pypy/goal/pypy-c"),
    ("pypy-jit-ext-c", "./pypy/goal/pypy-jit-ext-c"),
]
