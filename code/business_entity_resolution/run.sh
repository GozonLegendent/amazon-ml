#!/usr/bin/env bash
# Stage runner. Usage:  bash run.sh <stage> [extra args passed to the stage]
#   prepare | bienc | retrieve | features | prune | xenc | final | decide | validate
#   all1  = prepare -> bienc -> retrieve -> refold
#   all2  = features -> prune -> feats2 -> xenc -> final -> decide -> validate
#   smoke = both chains on a 1% subset (catches bugs in a few minutes)
# Paths default to <ROOT>/student_resource/dataset and <ROOT>/submission/...
# Override with env vars:  DATA=/path/to/dataset WORK=/path/to/work bash run.sh prepare
set -eo pipefail
cd "$(dirname "$0")"
ROOT=${ROOT:-$(cd ../../.. && pwd)}
DATA=${DATA:-$ROOT/student_resource/dataset}
WORK=${WORK:-$ROOT/work}
OUT=${OUT:-$ROOT/submission/output}
VALIDATOR=${VALIDATOR:-$ROOT/student_resource/utils/validate_submission.py}
LOG=$ROOT/logs
mkdir -p "$LOG"
stage=${1:-}; shift || true
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false

run() { local name=$1; shift; echo ">>> $*"; python -m "$@" 2>&1 | grep --line-buffered -v DeprecationWarning | tee "$LOG/$name.log"; }

chain1() {  # $1=data $2=work $3=log prefix, rest = extra bienc args
  local d=$1 w=$2 p=$3; shift 3
  run ${p}prepare  src.prepare   --data-dir "$d" --work-dir "$w"
  run ${p}bienc    src.biencoder --data-dir "$d" --work-dir "$w" "$@"
  run ${p}retrieve src.retrieve  --data-dir "$d" --work-dir "$w"
  run ${p}refold   src.refold    --data-dir "$d" --work-dir "$w"
}
chain2() {  # $1=data $2=work $3=out $4=log prefix, rest = extra xenc args
  local d=$1 w=$2 o=$3 p=$4; shift 4
  run ${p}features src.features     --data-dir "$d" --work-dir "$w"
  run ${p}prune    src.ranker       --data-dir "$d" --work-dir "$w" --stage prune
  run ${p}feats2   src.features2    --data-dir "$d" --work-dir "$w"
  run ${p}xenc     src.crossencoder --data-dir "$d" --work-dir "$w" "$@"
  run ${p}final    src.ranker       --data-dir "$d" --work-dir "$w" --stage final
  run ${p}decide   src.decide       --data-dir "$d" --work-dir "$w" --out-dir "$o"
  python3 "$VALIDATOR" --matching "$o/matching_results.tsv" --candidate "$o/candidate_pairs.tsv" \
      --test-dir "$d/test" 2>&1 | tee "$LOG/${p}validate.log"
}

case "$stage" in
  smoke)
    run smoke_subset src.make_dev_subset --data-dir "$DATA" --out-dir "$ROOT/dev_data" --frac 0.01
    chain1 "$ROOT/dev_data" "$ROOT/dev_work" smoke_ --max-pairs 30000 --bs 256 --overwrite
    chain2 "$ROOT/dev_data" "$ROOT/dev_work" "$ROOT/dev_output" smoke_ --max-queries 20000 --overwrite
    ;;
  prepare)  run prepare  src.prepare      --data-dir "$DATA" --work-dir "$WORK" "$@" ;;
  bienc)    run bienc    src.biencoder    --data-dir "$DATA" --work-dir "$WORK" "$@" ;;
  retrieve) run retrieve src.retrieve     --data-dir "$DATA" --work-dir "$WORK" "$@" ;;
  refold)   run refold   src.refold       --data-dir "$DATA" --work-dir "$WORK" "$@" ;;
  features) run features src.features     --data-dir "$DATA" --work-dir "$WORK" "$@" ;;
  prune)    run prune    src.ranker       --data-dir "$DATA" --work-dir "$WORK" --stage prune "$@" ;;
  feats2)   run feats2   src.features2    --data-dir "$DATA" --work-dir "$WORK" "$@" ;;
  analyze)  run analyze  src.analyze      --data-dir "$DATA" --work-dir "$WORK" "$@" ;;
  xenc)     run xenc     src.crossencoder --data-dir "$DATA" --work-dir "$WORK" "$@" ;;
  final)    run final    src.ranker       --data-dir "$DATA" --work-dir "$WORK" --stage final "$@" ;;
  decide)   run decide   src.decide       --data-dir "$DATA" --work-dir "$WORK" --out-dir "$OUT" "$@" ;;
  validate)
    python3 "$VALIDATOR" --matching "$OUT/matching_results.tsv" --candidate "$OUT/candidate_pairs.tsv" \
        --test-dir "$DATA/test" 2>&1 | tee "$LOG/validate.log" ;;
  all1) chain1 "$DATA" "$WORK" "" ;;
  all2) chain2 "$DATA" "$WORK" "$OUT" "" ;;
  *) echo "unknown stage '$stage'"; exit 1 ;;
esac
