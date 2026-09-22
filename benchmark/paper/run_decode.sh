#!/bin/bash
# Next-token latency: greedy decode with a key/value cache.
#
# The model sweep times one forward over a fixed-length prompt.  A deployed
# language model spends its time elsewhere, generating one token per step
# against a cache of everything before it, and each step waits for the last
# one's argmax.  This stage measures that step.
#
#   ours            applevel/gpt2_decode.py
#   torch-eager     applevel/gpt2_decode_torch.py, the cache sliced per step
#   torch-compile   the same, static cache + mask so shapes do not change
#   torch-compile-ro   and CUDA graphs on top
#
# All four run the same prompt on the same weights and must produce the same
# token stream; ours also checks itself against one full causal forward.
#
#   DECODE_MODELS  default "distilgpt2 gpt2 gpt2-medium" (the GPT-2 ladder)
#   DECODE_NEW     tokens generated per round (64)
#   DECODE_ROUNDS  generations per process; the first is cold (5)
#   DECODE_RUNS    processes per system and model (10)
#   DECODE_PYPY    interpreter, default $PYPY
set -e
HERE=$(cd "$(dirname "$0")" && pwd)
source "$HERE/config.sh"
APP="$HERE/../applevel"

PROBE_PYPY=${DECODE_PYPY:-$PYPY}
[ -x "$PROBE_PYPY" ] || { echo "run_decode.sh: no interpreter at $PROBE_PYPY" >&2; exit 1; }
MODELS=${DECODE_MODELS:-"distilgpt2 gpt2 gpt2-medium"}
NEW=${DECODE_NEW:-64}
GEN_ROUNDS=${DECODE_ROUNDS:-5}
RUNS=${DECODE_RUNS:-10}
SYSTEMS=${DECODE_SYSTEMS:-"torch-eager ours torch-compile torch-compile-ro"}

TSV="$OUT/decode.tsv"
tsv_init "$TSV" "model\tsystem\trun\tnew\ttoken_us\tp90_us\tcold_token_us\tprefill_ms\tlaunches_per_token\tretained_bytes\trounds_agree\tteacher_forced\ttokens_match\tbinary"
PYPY_SHA=$(binary_sha "$PROBE_PYPY")
TORCH_VER=$(pkg_version "$TORCH_PYTHON" torch torch)
STATUS=0

progress_init decode $(( $(echo $MODELS | wc -w) * RUNS * $(echo $SYSTEMS | wc -w) ))
for model in $MODELS; do
  W="$WEIGHTS/$model"
  [ -f "$W/index.json" ] || { echo "run_decode.sh: no weights for $model" >&2; STATUS=1; continue; }
  REF="$OUT/.decode_tokens_$model"
  rm -f "$REF"
  for run in $(seq "$RUNS"); do
    for system in $SYSTEMS; do
      progress_step "$model $system run $run/$RUNS"
      err=$(mktemp)
      if [ "$system" = ours ]; then
        row=$("$PROBE_PYPY" $JIT_FLAGS "$APP/gpt2_decode.py" "$W" "$NEW" \
              "$GEN_ROUNDS" 2>"$err" | grep '^decode ' | tail -1) || true
        binary=$PYPY_SHA
      else
        row=$("$TORCH_PYTHON" "$APP/gpt2_decode_torch.py" "${system#torch-}" \
              "$W" "$NEW" "$GEN_ROUNDS" 2>"$err" | grep '^decode ' | tail -1) || true
        binary=$TORCH_VER
      fi
      if [ -z "$row" ]; then
        echo "run_decode.sh: $model $system run $run produced no row:" \
             "$(grep -m1 -E 'Error|error|Exception' "$err" || tail -1 "$err")" >&2
        STATUS=1; rm -f "$err"; continue
      fi
      rm -f "$err"
      toks=$(echo "$row" | grep -o 'tokens=[0-9,]*' | cut -d= -f2)
      [ -f "$REF" ] || echo "$toks" > "$REF"
      match=0; [ "$toks" = "$(cat "$REF")" ] && match=1
      [ "$match" = 1 ] || { echo "run_decode.sh: $model $system run $run generated different tokens" >&2; STATUS=1; }
      forced=$(field_of "$row" teacher_forced)
      echo -e "$model\t$system\t$run\t$NEW\t$(field_of "$row" token_us)\t$(field_of "$row" p90_us)\t$(field_of "$row" cold_token_us)\t$(field_of "$row" prefill_ms)\t$(field_of "$row" launches_per_token)\t$(field_of "$row" retained_bytes)\t$(field_of "$row" rounds_agree)\t${forced}\t$match\t$binary" >> "$TSV"
      bench_record decode model="$model" system="$system" run="$run" new="$NEW" \
        token_us="$(field_of "$row" token_us)" p90_us="$(field_of "$row" p90_us)" \
        cold_token_us="$(field_of "$row" cold_token_us)" \
        prefill_ms="$(field_of "$row" prefill_ms)" \
        launches_per_token="$(field_of "$row" launches_per_token)" \
        retained_bytes="$(field_of "$row" retained_bytes)" \
        rounds_agree="$(field_of "$row" rounds_agree)" teacher_forced="$forced" \
        tokens_match="$match" binary="$binary"
    done
  done
done
progress_done
echo "wrote $TSV"
exit $STATUS
