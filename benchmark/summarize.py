import sys, csv, statistics, collections
rows = collections.defaultdict(list)
for path in sys.argv[1:]:
    with open(path) as f:
        for r in csv.DictReader(f, delimiter="\t"):
            key = (r["mode"], int(r["variant"]), int(r["k"]), int(r["n"]))
            rows[key].append((float(r["steady_us"]), float(r["warm_s"]), r.get("compiled_in_timed", "")))
def med(xs): return statistics.median(xs)
print("%-14s %3s %3s %10s %10s %8s %s" % ("mode", "var", "k", "n", "steady_us", "warm_s", "compiled_in_timed"))
for key in sorted(rows, key=lambda k: (k[1], k[2], k[3], k[0])):
    v = rows[key]
    print("%-14s %3d %3d %10d %10.1f %8.3f %s" % (key[0], key[1], key[2], key[3], med([x[0] for x in v]), med([x[1] for x in v]), ",".join(sorted(set(x[2] for x in v)))))
