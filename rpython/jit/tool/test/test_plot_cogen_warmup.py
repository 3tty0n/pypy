from rpython.jit.tool.plot_cogen_warmup import ratios


def test_ratios_warm_and_steady():
    results = {"b": {"base_times": [4.0, 2.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                     "changed_times": [2.0, 1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0]}}
    [(name, warm, steady)] = ratios(results)
    assert name == "b"
    assert abs(warm - 6.0 / 9.0) < 1e-12
    assert steady == 1.75
