import csv, os, statistics, sys, collections


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


def main(out_dir):
    lines = []
    lines.append("# Paper benchmark summary\n")

    micro = read_tsv(os.path.join(out_dir, "micro.tsv"))
    if micro:
        lines.append("## Microbenchmarks (median steady_us over rounds)\n")
        lines.append("| variant | k | n | fused (ours) | torch.compile | torch eager | speedup vs compile |")
        lines.append("|---|---|---|---|---|---|---|")
        groups = collections.defaultdict(lambda: collections.defaultdict(list))
        for r in micro:
            key = (int(r["variant"]), int(r["k"]), int(r["n"]))
            groups[key][r["mode"]].append(float(r["steady_us"]))
        for key in sorted(groups):
            g = groups[key]
            fused = med(g.get("fused", []))
            compiled = med(g.get("torch-compile", []))
            eager = med(g.get("torch-eager", []))
            speedup = compiled / fused if fused and compiled else None
            lines.append("| %d | %d | %d | %s | %s | %s | %s |" % (
                key[0], key[1], key[2], fmt(fused), fmt(compiled),
                fmt(eager), fmt(speedup, "%.2fx") if speedup else "n/a"))
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
        lines.append("| model | ours | torch.compile | torch eager | ratio ours/compile | correctness |")
        lines.append("|---|---|---|---|---|---|")
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
            c = corr.get(model, [])
            cstr = c[0] if c else ""
            lines.append("| %s | %s | %s | %s | %s | %s |" % (
                model, fmt(ours), fmt(compiled), fmt(eager),
                fmt(ratio, "%.2fx") if ratio else "n/a", cstr))
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
