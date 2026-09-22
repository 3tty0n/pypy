#!/bin/bash
# Pack a result run's row data into one archive, or unpack it again.
#
# A run writes about 3MB of tsv and jsonl, and results.jsonl is rewritten
# whole on every stage, so committing the rows loose puts a few megabytes of
# barely-diffable text into the history per run.  Packed, the same rows are
# under 300KB and change as one object.
#
#   archive.sh pack   RESULT_DIR   rows -> RESULT_DIR/data.tar.gz
#   archive.sh unpack RESULT_DIR   RESULT_DIR/data.tar.gz -> rows
#   archive.sh check  RESULT_DIR   every row file is in the archive, unchanged
#
# What stays loose, and why: machine.txt, so the provenance of a run is
# readable in a diff and on the web without downloading anything; and
# figures/, because the paper copies those pdf and tex files directly.
#
# The archive is reproducible - same rows in, same bytes out - so re-packing
# a run that has not changed produces no diff.  That needs tar to sort its
# entries and drop the filesystem's timestamps and ownership, and gzip to
# leave its own timestamp out.
set -eu

# The per-forward warm-up traces are one small file per (model, cache,
# round, system) - a few hundred of them, all raw rows.  leave/ is the MOTION
# leave check: every run's saved result and output, the checker and audit logs.
ROWS=("*.tsv" "results.jsonl" ".warmup_series" "leave")
NAME=data.tar.gz

usage() { sed -n '2,/^set -eu/p' "$0" | sed 's/^# \?//;$d'; exit 2; }

list_rows() {
  # No array: an empty one printed a blank line, which became a tar member
  # with an empty name in a run that happened to have no tsv at all.
  local dir=$1 f pat
  for pat in "${ROWS[@]}"; do
    for f in "$dir"/$pat; do
      [ -e "$f" ] && basename "$f"   # a directory goes in whole
    done
  done | LC_ALL=C sort
}

pack() {
  local dir=${1%/} rows
  [ -d "$dir" ] || { echo "archive.sh: no such directory: $dir" >&2; exit 1; }
  mapfile -t rows < <(list_rows "$dir")
  if [ "${#rows[@]}" -eq 0 ]; then
    echo "archive.sh: $dir holds no rows to pack" >&2
    return 0
  fi
  # --sort, a fixed mtime and numeric 0:0 ownership make the tar depend on the
  # file contents alone; gzip -n keeps its own name and timestamp out.
  tar --sort=name --mtime='2026-01-01 00:00:00 UTC' \
      --owner=0 --group=0 --numeric-owner \
      -cf - -C "$dir" "${rows[@]}" | gzip -9n > "$dir/$NAME.tmp"
  mv "$dir/$NAME.tmp" "$dir/$NAME"
  echo "$dir/$NAME: ${#rows[@]} files, $(du -h "$dir/$NAME" | cut -f1)"
}

unpack() {
  local dir=${1%/}
  [ -f "$dir/$NAME" ] || { echo "archive.sh: no $dir/$NAME" >&2; exit 1; }
  tar -xzf "$dir/$NAME" -C "$dir"
  echo "$dir: unpacked $(tar -tzf "$dir/$NAME" | wc -l) files"
}

check() {
  local dir=${1%/} tmp rc=0
  [ -f "$dir/$NAME" ] || { echo "archive.sh: no $dir/$NAME" >&2; exit 1; }
  tmp=$(mktemp -d)
  tar -xzf "$dir/$NAME" -C "$tmp"
  local f
  while IFS= read -r f; do
    if ! [ -e "$tmp/$f" ]; then
      echo "$dir/$f: not in the archive" >&2; rc=1
    elif [ -d "$dir/$f" ]; then
      diff -rq "$dir/$f" "$tmp/$f" >/dev/null 2>&1 ||
        { echo "$dir/$f: differs from the archive" >&2; rc=1; }
    elif ! cmp -s "$dir/$f" "$tmp/$f"; then
      echo "$dir/$f: differs from the archive" >&2; rc=1
    fi
  done < <(list_rows "$dir")
  rm -rf "$tmp"
  [ "$rc" = 0 ] && echo "$dir: archive matches the rows on disk"
  return $rc
}

[ $# -ge 2 ] || usage
cmd=$1; shift
case "$cmd" in
  pack)   for d in "$@"; do pack "$d"; done ;;
  unpack) for d in "$@"; do unpack "$d"; done ;;
  check)  for d in "$@"; do check "$d"; done ;;
  *)      usage ;;
esac
