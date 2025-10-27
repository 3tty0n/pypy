BENCHMARKS = [
    'bm_chameleon', 'bm_dulwich_log', 'bm_icbd', 'bm_mako',
    'raytrace-simple', 'scimark', 'spectral-norm', 'spitfire',
    'telco', 'bm_gzip', 'bm_krakatau', 'bm_mdp', 'pyxl_bench',
    'hexiom2', 'eparse', 'json_bench', 'pypy_interp', 'pyflate-fast'
]


BENCHMARKS_UNLADEN_SWALLOW = [
    'bm_django', 'bm_html5lib',
    'bm_richards', 'bm_spambayes',
    'bm_unpack_sequence',
]


COMMANDS = [
    ("pypy-c", "./pypy/goal/pypy-c"),
    ("pypy-jit-ext-c", "./pypy/goal/pypy-jit-ext-c")
]
