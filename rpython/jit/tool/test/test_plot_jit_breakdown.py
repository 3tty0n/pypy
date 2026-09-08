from rpython.jit.tool.plot_jit_breakdown import parse_summary


def test_parse_summary(tmpdir):
    log = tmpdir.join("s.log")
    log.write("[1a] Tracing:      \t10\t0.5\n[1b] Optimizing:   \t9\t0.25\n"
              "Backend:      \t8\t0.125\nBlackhole:    \t7\t0.0625\n"
              "PE cogen scan:\t2\t0.01\nPE cogen install:\t2\t0.02\n"
              "TOTAL:      \t\t3.0\nabort: trace too long:\t3\n"
              "abort: force quasi-immut:\t4\nTotal # of loops:\t5\n"
              "Total # of bridges:\t6\n")
    row = parse_summary(str(log))
    assert row["tracing"] == 0.5 and row["optimizing"] == 0.25
    assert row["backend"] == 0.125 and row["blackhole"] == 0.0625
    assert abs(row["cogen"] - 0.03) < 1e-12
    assert (row["loops"], row["bridges"], row["aborts"]) == (5, 6, 7)
    assert row["total"] == 3.0
