import csv, json, os, statistics, sys, collections


def read_tsv(path):
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return list(csv.DictReader(f, delimiter="\t"))


def med(xs):
    xs = [float(x) for x in xs if x not in (None, "")]
    return statistics.median(xs) if xs else None


def fmt(v, spec="%.1f"):
    return spec % v if v is not None else "n/a"


def read_jsonl(path):
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def known(v):
    """compile_ms / first_run_ms are -1 when the system cannot separate them."""
    return v if isinstance(v, (int, float)) and v >= 0 else None


def compile_table(records, lines):
    """Compile cost next to the steady state, per workload and system, from
    results.jsonl, the one place those fields are named.  Break-even is the
    iteration count after which a compiled system has repaid its first run
    relative to torch eager."""
    g = collections.defaultdict(lambda: collections.defaultdict(
        lambda: collections.defaultdict(list)))
    for r in records:
        if r.get("kind") == "micro" and r.get("dtype", "float64") == "float64":
            key = "micro v%s k%s n%s" % (r.get("variant"), r.get("k"), r.get("n"))
            system = r.get("mode")
        elif r.get("kind") == "models":
            key, system = r.get("model"), r.get("system")
        else:
            continue
        for field in ("compile_ms", "first_run_ms", "steady_us"):
            if r.get(field) not in (None, ""):
                g[key][system][field].append(float(r[field]))
    rows = []
    for key in sorted(g):
        eager = g[key].get("torch-eager", {})
        e_first = known(med(eager.get("first_run_ms", [])))
        e_steady = med(eager.get("steady_us", []))
        for system in ("torch-compile", "torch-compile-ro", "torch-compile-mat",
                       "torch-tensorrt", "jax", "iree", "triton", "fused", "ours"):
            if system not in g[key]:
                continue
            d = g[key][system]
            compile_ms = known(med(d.get("compile_ms", [])))
            first = known(med(d.get("first_run_ms", [])))
            steady = med(d.get("steady_us", []))
            if compile_ms is None and first is None:
                continue
            # Systems that report compile_ms separately pay it before the first
            # run; torch.compile folds it into first_run_ms already.
            paid = (first + (compile_ms or 0.0)) if first is not None else None
            even = None
            if (paid is not None and e_first is not None and steady
                    and e_steady and e_steady > steady):
                even = (paid - e_first) * 1000.0 / (e_steady - steady)
            rows.append("| %s | %s | %s | %s | %s | %s |" % (
                key, system, fmt(compile_ms), fmt(first), fmt(steady),
                fmt(even, "%.0f")))
    if rows:
        lines.append("## Compilation overhead (median; break-even vs torch eager, iterations)\n")
        lines.append("| workload | system | compile_ms | first_run_ms | steady_us | break-even |")
        lines.append("|---|---|---|---|---|---|")
        lines.extend(rows)
        lines.append("")


def warmup_table(records, lines):
    """Per-forward warm-up trace (kind=warmup, run_warmup.sh): first forward,
    steady-state onset, and the crossover against torch-eager, per model,
    system and cache state, median over the rounds."""
    g = collections.defaultdict(lambda: collections.defaultdict(list))
    for r in records:
        if r.get("kind") != "warmup":
            continue
        key = (r.get("model"), r.get("system"), r.get("cache"))
        for field in ("first_us", "steady_us", "steady_at", "crossover",
                      "cold_compiles", "first_forward_compiles"):
            if r.get(field) not in (None, ""):
                g[key][field].append(r[field])
    if not g:
        return
    lines.append("## Warm-up (median over rounds; steady_at/crossover in "
                  "iterations, 'none' if never reached within N)\n")
    lines.append("| model | system | cache | first (ms) | steady_at | "
                  "steady_us | crossover vs eager |")
    lines.append("|---|---|---|---|---|---|---|")
    def numeric_med(xs):
        nums = [x for x in xs if isinstance(x, (int, float))]
        return med(nums) if nums else None
    for key in sorted(g):
        d = g[key]
        first = numeric_med(d.get("first_us", []))
        steady = numeric_med(d.get("steady_us", []))
        sa = numeric_med(d.get("steady_at", []))
        sa_str = "%.0f" % sa if sa is not None else "none"
        co = numeric_med(d.get("crossover", []))
        co_str = "%.0f" % co if co is not None else "none"
        lines.append("| %s | %s | %s | %s | %s | %s | %s |" % (
            key[0], key[1], key[2],
            "%.2f" % (first / 1000.0) if first is not None else "n/a",
            sa_str, fmt(steady), co_str))
    lines.append("")

    ours_compiles = {k: v for k, v in g.items()
                     if k[1] == "ours" and v.get("cold_compiles")}
    if ours_compiles:
        lines.append("## Warm-up: kernel compiles (ours, median over rounds; "
                      "cold_compiles = Triton subprocess compiles over the "
                      "whole trace, first_forward_compiles = at iter 0)\n")
        lines.append("| model | cache | cold_compiles | first_forward_compiles |")
        lines.append("|---|---|---|---|")
        for key in sorted(ours_compiles):
            d = ours_compiles[key]
            cc = numeric_med(d.get("cold_compiles", []))
            ffc = numeric_med(d.get("first_forward_compiles", []))
            lines.append("| %s | %s | %s | %s |" % (
                key[0], key[2],
                "%.0f" % cc if cc is not None else "n/a",
                "%.0f" % ffc if ffc is not None else "n/a"))
        lines.append("")


def main(out_dir):
    lines = []
    lines.append("# Paper benchmark summary\n")

    micro = read_tsv(os.path.join(out_dir, "micro.tsv"))
    if micro:
        lines.append("## Microbenchmarks (median steady_us over rounds)\n")
        lines.append("| variant | k | n | fused (ours) | app-level (ours) | eager (ours) | nojit (ours) | torch.compile | compile-ro | compile-mat | torch eager | JAX/XLA | IREE | Triton | speedup vs compile | interp tax | ours/Triton |")
        lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
        groups = collections.defaultdict(lambda: collections.defaultdict(list))
        for r in micro:
            dtype = r.get("breaks") or r.get("graphs") or "float64"
            key = (dtype, int(r["variant"]), int(r["k"]), int(r["n"]))
            groups[key][r["mode"]].append(float(r["steady_us"]))
        def row(key, g):
            fused = med(g.get("fused", []))
            app = med(g.get("app", []))
            compiled = med(g.get("torch-compile", []))
            compiled_ro = med(g.get("torch-compile-ro", []))
            compiled_mat = med(g.get("torch-compile-mat", []))
            speedup = compiled / fused if fused and compiled else None
            # What the PyPy interpreter costs on top of the mechanism: the same
            # model run app-level over the same model run as a translated
            # RPython program.
            tax = app / fused if app and fused else None
            triton = med(g.get("triton", []))
            kernel_gap = fused / triton if fused and triton else None
            return "| %d | %d | %d | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s |" % (
                key[1], key[2], key[3], fmt(fused), fmt(app),
                fmt(med(g.get("eager", []))),
                fmt(med(g.get("nojit", []))), fmt(compiled),
                fmt(compiled_ro), fmt(compiled_mat),
                fmt(med(g.get("torch-eager", []))),
                fmt(med(g.get("jax", []))), fmt(med(g.get("iree", []))),
                fmt(triton),
                fmt(speedup, "%.2fx") if speedup else "n/a",
                fmt(tax, "%.2fx") if tax else "n/a",
                fmt(kernel_gap, "%.2fx") if kernel_gap else "n/a")
        for key in sorted(groups):
            if key[0] == "float64":
                lines.append(row(key, groups[key]))
        lines.append("")
        others = [k for k in sorted(groups) if k[0] != "float64"]
        if others:
            lines.append("## Precision sweep (median steady_us)\n")
            lines.append("| dtype | variant | k | n | fused (ours) | app-level (ours) | eager (ours) | nojit (ours) | torch.compile | compile-ro | compile-mat | torch eager | JAX/XLA | IREE | Triton | speedup vs compile | interp tax | ours/Triton |")
            lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
            for key in others:
                lines.append("| %s %s" % (key[0], row(key, groups[key])))
            lines.append("")

        lines.append("## Guards / graph breaks (launches per iter, torch graph breaks)\n")
        lines.append("| variant | n | launches/iter (fused) | torch graphs | torch breaks |")
        lines.append("|---|---|---|---|---|")
        gb = collections.defaultdict(lambda: collections.defaultdict(list))
        for r in micro:
            v = int(r["variant"])
            if v < 1 or v > 5:
                continue
            key = (v, int(r["n"]))
            gb[key][r["mode"]].append(r)
        for key in sorted(gb):
            fused_rows = gb[key].get("fused", [])
            torch_rows = gb[key].get("torch-compile", [])
            launches = med([x["launches_per_iter"] for x in fused_rows
                           if x.get("launches_per_iter")])
            graphs = torch_rows[0].get("launches_per_iter", "") if torch_rows else ""
            breaks = torch_rows[0].get("graphs", "") if torch_rows else ""
            lines.append("| %d | %d | %s | %s | %s |" % (
                key[0], key[1], fmt(launches, "%.1f"), graphs, breaks))
        lines.append("")

    models = read_tsv(os.path.join(out_dir, "models.tsv"))
    if models:
        lines.append("## End-to-end models (median steady_us, ratio to torch.compile)\n")
        lines.append("| model | ours | torch.compile | compile-ro | compile-mat | torch eager | JAX/XLA | IREE | TensorRT | ratio ours/compile | ratio jax/compile | launches/iter (ours) | correctness | tol | pass |")
        lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
        g = collections.defaultdict(lambda: collections.defaultdict(list))
        corr = collections.defaultdict(list)
        tols = {}
        verdict = collections.defaultdict(list)
        launches = collections.defaultdict(list)
        for r in models:
            g[r["model"]][r["system"]].append(float(r["steady_us"]))
            if r.get("maxabsdiff"):
                corr[r["model"]].append(r["maxabsdiff"])
            if r.get("tolerance"):
                tols[r["model"]] = r["tolerance"]
            if r.get("pass") not in (None, ""):
                verdict[r["model"]].append(r["pass"])
            if r["system"] == "ours" and r.get("launches_per_iter"):
                launches[r["model"]].append(float(r["launches_per_iter"]))
        for model in sorted(g):
            ours = med(g[model].get("ours", []))
            compiled = med(g[model].get("torch-compile", []))
            compiled_ro = med(g[model].get("torch-compile-ro", []))
            compiled_mat = med(g[model].get("torch-compile-mat", []))
            eager = med(g[model].get("torch-eager", []))
            ratio = ours / compiled if ours and compiled else None
            jax_ = med(g[model].get("jax", []))
            iree = med(g[model].get("iree", []))
            jratio = jax_ / compiled if jax_ and compiled else None
            c = corr.get(model, [])
            cstr = c[0] if c else ""
            v = verdict.get(model, [])
            # One failing system fails the model: the point of the column is
            # that nothing slipped through, not that something passed.
            vstr = "" if not v else ("pass" if all(x == "1" for x in v)
                                     else "FAIL")
            lines.append("| %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s |" % (
                model, fmt(ours), fmt(compiled), fmt(compiled_ro), fmt(compiled_mat),
                fmt(eager), fmt(jax_),
                fmt(iree), fmt(med(g[model].get("torch-tensorrt", []))),
                fmt(ratio, "%.2fx") if ratio else "n/a",
                fmt(jratio, "%.2fx") if jratio else "n/a",
                fmt(med(launches.get(model, [])), "%.1f"),
                cstr, tols.get(model, ""), vstr))
        lines.append("")

    batch = read_tsv(os.path.join(out_dir, "batch.tsv"))
    if batch:
        # steady_us is per forward (the whole batch); per_seq_us is steady/B,
        # which is what converges as the GEMMs grow.  "rows identical" is the
        # check on the batching itself: B copies of one sequence must give B
        # identical output row blocks.
        lines.append("## Batch-size sweep "
                     "(median steady_us per forward; per_seq = steady/B)\n")
        lines.append("| model | batch | ours | torch.compile | compile-ro "
                     "| torch eager | JAX/XLA | ratio ours/compile "
                     "| ratio ours/jax | per_seq ours | per_seq compile "
                     "| per_seq jax | rows identical | failed |")
        lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
        st = collections.defaultdict(list)
        ps = collections.defaultdict(list)
        ident = collections.defaultdict(list)
        failed = collections.defaultdict(list)
        for r in batch:
            key = (r["model"], int(r["batch"]), r["system"])
            if (r.get("status") or "ok") != "ok" or not r.get("steady_us"):
                failed[(r["model"], int(r["batch"]))].append(r["system"])
                continue
            st[key].append(float(r["steady_us"]))
            if r.get("per_seq_us"):
                ps[key].append(float(r["per_seq_us"]))
            if r.get("batch_rows_identical") not in (None, ""):
                ident[(r["model"], int(r["batch"]))].append(
                    r["batch_rows_identical"])
        for model, b in sorted({(k[0], k[1]) for k in st}
                               | set(failed)):
            def v(system, table=st):
                return med(table.get((model, b, system), []))
            ours, comp = v("ours"), v("torch-compile")
            jx = v("jax")
            marks = ident.get((model, b), [])
            miss = sorted(set(failed.get((model, b), [])))
            lines.append("| %s | %d | %s | %s | %s | %s | %s | %s | %s | %s "
                         "| %s | %s | %s | %s |" % (
                model, b, fmt(ours), fmt(comp), fmt(v("torch-compile-ro")),
                fmt(v("torch-eager")), fmt(jx),
                fmt(ours / comp, "%.2fx") if ours and comp else "n/a",
                fmt(ours / jx, "%.2fx") if ours and jx else "n/a",
                fmt(v("ours", ps)), fmt(v("torch-compile", ps)),
                fmt(v("jax", ps)),
                "" if not marks else ("yes" if all(x == "1" for x in marks)
                                      else "NO"),
                ",".join(miss)))
        lines.append("")

    ablation = read_tsv(os.path.join(out_dir, "ablation.tsv"))
    if ablation:
        lines.append("## Ablations (median steady_us)\n")
        lines.append("| experiment | variant | model | steady_us | launches/iter | note |")
        lines.append("|---|---|---|---|---|---|")
        g = collections.defaultdict(list)
        launches = collections.defaultdict(list)
        notes = {}
        for r in ablation:
            key = (r["experiment"], r["variant"], r["model"])
            if r.get("steady_us"):
                g[key].append(float(r["steady_us"]))
            if r.get("launches_per_iter"):
                launches[key].append(float(r["launches_per_iter"]))
            if r.get("note"):
                notes[key] = r["note"]
        for key in sorted(g) if g else sorted(notes):
            steady = med(g.get(key, []))
            lines.append("| %s | %s | %s | %s | %s | %s |" % (
                key[0], key[1], key[2], fmt(steady),
                fmt(med(launches.get(key, [])), "%.1f"), notes.get(key, "")))
        for key in sorted(notes):
            if key not in g:
                lines.append("| %s | %s | %s | %s | %s | %s |" % (
                    key[0], key[1], key[2], "n/a", "", notes[key]))
        lines.append("")

    deopt = read_tsv(os.path.join(out_dir, "deopt.tsv"))
    if deopt:
        # first_fail/after_fail/peak are single iterations, not medians of a
        # loop: peak is the deoptimization itself (bridge compile here, dynamo
        # recompilation there), steady is what the iteration costs once the
        # system has settled on the new path.
        lines.append("## Deoptimization cost "
                     "(median over rounds, us per iteration)\n")
        lines.append("| system | pattern | steady_us | first_fail_us "
                     "| after_fail_us | peak_us | cold_us | launches/it "
                     "| loops | bridges |")
        lines.append("|---|---|---|---|---|---|---|---|---|---|")
        g = collections.defaultdict(lambda: collections.defaultdict(list))
        order = ["never", "alternate", "both-hot", "fresh", "probe-a",
                 "probe-e"]
        for r in deopt:
            for col in ("steady_us", "first_fail_us", "after_fail_us",
                        "peak_us", "cold_us", "launches_per_iter", "loops",
                        "bridges"):
                if r.get(col) not in (None, ""):
                    g[(r["pattern"], r["system"])][col].append(float(r[col]))

        def cell(d, col, spec="%.0f"):
            v = med(d.get(col, []))
            return fmt(v, spec) if v is not None and v >= 0 else "n/a"

        for key in sorted(g, key=lambda k: (order.index(k[0])
                                            if k[0] in order else 99, k[1])):
            d = g[key]
            lines.append("| %s | %s | %s | %s | %s | %s | %s | %s | %s | %s |"
                         % (key[1], key[0], cell(d, "steady_us", "%.1f"),
                            cell(d, "first_fail_us"), cell(d, "after_fail_us"),
                            cell(d, "peak_us"), cell(d, "cold_us"),
                            cell(d, "launches_per_iter", "%.2f"),
                            cell(d, "loops"), cell(d, "bridges")))
        lines.append("")

    records = read_jsonl(os.path.join(out_dir, "results.jsonl"))
    compile_table(records, lines)
    warmup_table(records, lines)

    dsum = read_tsv(os.path.join(out_dir, "dynamic_summary.tsv"))
    if dsum:
        # pass 1 is the first visit of each length, pass 2 the revisit.  The
        # counter columns are the delta inside that length's window, so a
        # non-zero value on pass 2 means a revisit was not free.
        lines.append("## Dynamic sequence length "
                     "(median steady_us per length, counters per window)\n")
        lines.append("| system | pass | length | median_us | total_us | loops "
                     "| bridges | kernels | cache_hits | recompiles |")
        lines.append("|---|---|---|---|---|---|---|---|---|---|")
        g = collections.defaultdict(list)
        for r in dsum:
            g[(r["system"], r.get("pass", "1"), r["length"])].append(r)

        def d(rows, col):
            vs = [float(x[col]) for x in rows if x.get(col) not in (None, "")]
            return "%.0f" % med(vs) if vs else ""

        for key in sorted(g, key=lambda k: (k[0], k[1], int(k[2]))):
            rows = g[key]
            lines.append("| %s | %s | %s | %s | %s | %s | %s | %s | %s | %s |" % (
                key[0], key[1], key[2],
                fmt(med([x["median_us"] for x in rows])),
                fmt(med([x["total_us"] for x in rows])) if rows[0].get("total_us") else "",
                d(rows, "loops"), d(rows, "bridges"), d(rows, "kernels"),
                d(rows, "cache_hits"), d(rows, "recompiles")))
        lines.append("")

    text = "\n".join(lines) + "\n"
    out_path = os.path.join(out_dir, "summary.md")
    with open(out_path, "w") as f:
        f.write(text)
    print(text)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else ".")
