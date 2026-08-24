from rpython.jit.tool.plot_cogen_comparison import compare, suite_summary


def raw(base, changed):
    return {"bench": {"base_times": base, "changed_times": changed}}


def test_compare_balances_forward_and_reverse_order():
    forward = raw([10.0, 5.0, 4.0, 4.0], [8.0, 4.0, 4.0, 4.0])
    reverse = raw([9.0, 4.5, 4.0, 4.0], [10.0, 5.0, 4.0, 4.0])
    [row] = compare(forward, reverse)
    assert row["first"] < 0
    assert abs(row["stable"]) < 1e-12
    assert row["total"] < 0


def test_suite_summary_uses_equal_benchmark_weights():
    rows = [{"first": -10.0, "early": -10.0, "stable": -10.0,
             "total": -10.0},
            {"first": 10.0, "early": 10.0, "stable": 10.0,
             "total": 10.0}]
    summary = suite_summary(rows)
    assert summary["first"] < 0
    assert summary["first"] > -1
