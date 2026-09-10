"""Append one measurement to $OUT/results.jsonl.

    record.py micro mode=fused variant=8 k=1 n=256000 steady_us=16217.9 ...

The tsv files are positional and a row's meaning depends on which system wrote
it: micro.tsv carries the dtype in the `breaks` column for our rows and in
`graphs` for torch's, because the two emitters have different field counts.
A record here names its own fields.
"""

import json
import math
import os
import sys


def coerce(value):
    if value == "":
        return None
    try:
        return int(value)
    except ValueError:
        pass
    try:
        f = float(value)
    except ValueError:
        return value
    # inf and nan are not JSON; a reader that is not Python would reject them.
    return f if math.isfinite(f) else str(value)


def parse(argv):
    if len(argv) < 2:
        raise SystemExit("usage: record.py KIND key=value ...")
    record = {"kind": argv[1]}
    for arg in argv[2:]:
        key, sep, value = arg.partition("=")
        if not sep:
            raise SystemExit("record.py: %r is not key=value" % arg)
        parsed = coerce(value)
        if parsed is not None:
            record[key] = parsed
    return record


def main(argv):
    out = os.environ.get("OUT")
    if not out:
        raise SystemExit("record.py: $OUT is unset")
    record = parse(argv)
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "results.jsonl"), "a") as f:
        f.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
