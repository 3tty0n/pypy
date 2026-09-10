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
        for system in ("torch-compile", "torch-tensorrt", "jax", "iree", "triton", "fused", "ours"):
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


def main(out_dir):
    lines = []
    lines.append("# Paper benchmark summary\n")

    micro = read_tsv(os.path.join(out_dir, "micro.tsv"))
    if micro:
        lines.append("## Microbenchmarks (median steady_us over rounds)\n")
        lines.append("| variant | k | n | fused (ours) | app-level (ours) | eager (ours) | nojit (ours) | torch.compile | torch eager | JAX/XLA | IREE | Triton | speedup vs compile | interp tax | ours/Triton |")
        lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
        groups = collections.defaultdict(lambda: collections.defaultdict(list))
        for r in micro:
            dtype = r.get("breaks") or r.get("graphs") or "float64"
            key = (dtype, int(r["variant"]), int(r["k"]), int(r["n"]))
            groups[key][r["mode"]].append(float(r["steady_us"]))
        def row(key, g):
            fused = med(g.get("fused", []))
            app = med(g.get("app", []))
            compiled = med(g.get("torch-compile", []))
            speedup = compiled / fused if fused and compiled else None
            # What the PyPy interpreter costs on top of the mechanism: the same
            # model run app-level over the same model run as a translated
            # RPython program.
            tax = app / fused if app and fused else None
            triton = med(g.get("triton", []))
            kernel_gap = fused / triton if fused and triton else None
            return "| %d | %d | %d | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s |" % (
                key[1], key[2], key[3], fmt(fused), fmt(app),
                fmt(med(g.get("eager", []))),
                fmt(med(g.get("nojit", []))), fmt(compiled),
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
            lines.append("| dtype | variant | k | n | fused (ours) | app-level (ours) | eager (ours) | nojit (ours) | torch.compile | torch eager | JAX/XLA | IREE | Triton | speedup vs compile | interp tax | ours/Triton |")
            lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
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
        lines.append("| model | ours | torch.compile | torch eager | JAX/XLA | IREE | TensorRT | ratio ours/compile | ratio jax/compile | correctness |")
        lines.append("|---|---|---|---|---|---|---|---|---|---|")
        g = collections.defaultdict(lambda: collections.defaultdict(list))
        corr = collections.defaultdict(list)
        for r in models:
            g[r["model"]][r["system"]].append(float(r["steady_us"]))
            if r.get("maxabsdiff"):
                corr[r["model"]].append(r["maxabsdiff"])
        for model in sorted(g):
            ours = med(g[model].get("ours", []))
            compiled = med(g[model].get("torch-compile", []))
            eager = med(g[model].get("torch-eager", []))
            ratio = ours / compiled if ours and compiled else None
            jax_ = med(g[model].get("jax", []))
            iree = med(g[model].get("iree", []))
            jratio = jax_ / compiled if jax_ and compiled else None
            c = corr.get(model, [])
            cstr = c[0] if c else ""
            lines.append("| %s | %s | %s | %s | %s | %s | %s | %s | %s | %s |" % (
                model, fmt(ours), fmt(compiled), fmt(eager), fmt(jax_),
                fmt(iree), fmt(med(g[model].get("torch-tensorrt", []))),
                fmt(ratio, "%.2fx") if ratio else "n/a",
                fmt(jratio, "%.2fx") if jratio else "n/a", cstr))
        lines.append("")

    ablation = read_tsv(os.path.join(out_dir, "ablation.tsv"))
    if ablation:
        lines.append("## Ablations (median steady_us)\n")
        lines.append("| experiment | variant | model | steady_us | note |")
        lines.append("|---|---|---|---|---|")
        g = collections.defaultdict(list)
        notes = {}
        for r in ablation:
            key = (r["experiment"], r["variant"], r["model"])
            if r.get("steady_us"):
                g[key].append(float(r["steady_us"]))
            if r.get("note"):
                notes[key] = r["note"]
        for key in sorted(g) if g else sorted(notes):
            steady = med(g.get(key, []))
            lines.append("| %s | %s | %s | %s | %s |" % (
                key[0], key[1], key[2], fmt(steady), notes.get(key, "")))
        for key in sorted(notes):
            if key not in g:
                lines.append("| %s | %s | %s | %s | %s |" % (
                    key[0], key[1], key[2], "n/a", notes[key]))
        lines.append("")

    compile_table(read_jsonl(os.path.join(out_dir, "results.jsonl")), lines)

    dsum = read_tsv(os.path.join(out_dir, "dynamic_summary.tsv"))
    if dsum:
        lines.append("## Dynamic sequence length (median steady_us per length)\n")
        lines.append("| system | length | median_us | loops | bridges | recompiles |")
        lines.append("|---|---|---|---|---|---|")
        g = collections.defaultdict(list)
        for r in dsum:
            key = (r["system"], r["length"])
            g[key].append(r)
        for key in sorted(g, key=lambda k: (k[0], int(k[1]))):
            rows = g[key]
            m = med([x["median_us"] for x in rows])
            loops = rows[0].get("loops", "")
            bridges = rows[0].get("bridges", "")
            recompiles = rows[0].get("recompiles", "")
            lines.append("| %s | %s | %s | %s | %s | %s |" % (
                key[0], key[1], fmt(m), loops, bridges, recompiles))
        lines.append("")

    text = "\n".join(lines) + "\n"
    out_path = os.path.join(out_dir, "summary.md")
    with open(out_path, "w") as f:
        f.write(text)
    print(text)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else ".")
