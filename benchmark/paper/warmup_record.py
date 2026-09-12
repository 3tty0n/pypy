"""Parse one warm-up trace (stdin, the `iter=I us=U` lines plus the final
`total_us=.. steady_us=.. steady_at=..` line that WARMUP_TRACE mode prints)
into rows appended to $OUT/warmup.tsv, plus a one-line summary that
run_warmup.sh picks up with the existing field_of() shell helper.

    warmup_record.py MODEL SYSTEM CACHE ROUND TSV_PATH SERIES_OUT [EAGER_PATH] < trace

SERIES_OUT always gets this run's per-iteration series (one float per line),
so a later system in the same (model, cache, round) can pass it back in as
EAGER_PATH to get a crossover point against torch-eager.
"""
import sys


def parse_kv(line):
    return dict(kv.split('=', 1) for kv in line.split())


def main(argv):
    if len(argv) < 7:
        raise SystemExit('usage: warmup_record.py MODEL SYSTEM CACHE ROUND '
                          'TSV_PATH SERIES_OUT [EAGER_PATH]')
    model, system, cache, round_, tsv_path, series_out = argv[1:7]
    eager_path = argv[7] if len(argv) > 7 else None

    us = []
    total_us = steady_us = None
    steady_at = 'none'
    cold_compiles = first_forward_compiles = ''
    with open(tsv_path, 'a') as tsv:
        for line in sys.stdin:
            line = line.strip()
            if line.startswith('iter='):
                kv = parse_kv(line)
                i, u = int(kv['iter']), float(kv['us'])
                us.append(u)
                # compiled/launches: the "ours" driver only (kernel-cache
                # instrumentation); blank for every other system.
                compiled = kv.get('compiled', '')
                launches = kv.get('launches', '')
                tsv.write('%s\t%s\t%s\t%s\t%d\t%.1f\t%s\t%s\n' %
                          (model, system, cache, round_, i, u, compiled, launches))
            elif line.startswith('total_us='):
                kv = parse_kv(line)
                total_us = float(kv['total_us'])
                steady_us = float(kv['steady_us'])
                steady_at = kv['steady_at']
                cold_compiles = kv.get('cold_compiles', '')
                first_forward_compiles = kv.get('first_forward_compiles', '')

    if not us:
        print('error=no_trace_output', file=sys.stderr)
        return 1

    with open(series_out, 'w') as f:
        f.write('\n'.join('%.1f' % u for u in us))

    fields = ['first_us=%.1f' % us[0],
              'total_us=%.1f' % (total_us if total_us is not None else sum(us)),
              'steady_us=%.1f' % (steady_us if steady_us is not None else 0.0),
              'steady_at=%s' % steady_at]
    if cold_compiles != '':
        fields.append('cold_compiles=%s' % cold_compiles)
    if first_forward_compiles != '':
        fields.append('first_forward_compiles=%s' % first_forward_compiles)

    if eager_path:
        try:
            eager_us = [float(x) for x in open(eager_path).read().split()]
        except IOError:
            eager_us = []
        crossover = 'none'
        ce = cs = 0.0
        for k in range(min(len(us), len(eager_us))):
            ce += eager_us[k]
            cs += us[k]
            if cs < ce:
                crossover = k + 1
                break
        fields.append('crossover=%s' % crossover)

    print(' '.join(fields))
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv))
