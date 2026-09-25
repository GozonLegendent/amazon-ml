#!/usr/bin/env bash
# Stage runner. Usage:  bash run.sh <stage> [extra args passed to the stage]
# Stages: smoke | prepare | bienc | retrieve | all1
# Paths default to the layout  <ROOT>/student_resource/dataset  and  <ROOT>/submission/code/...
# Override with env vars:  DATA=/path/to/dataset WORK=/path/to/work bash run.sh prepare
set -eo pipefail
cd "$(dirname "$0")"
ROOT=${ROOT:-$(cd ../../.. && pwd)}
DATA=${DATA:-$ROOT/student_resource/dataset}
WORK=${WORK:-$ROOT/work}
LOG=$ROOT/logs
mkdir -p "$LOG"
stage=${1:-}; shift || true
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false

run() { local name=$1; shift; echo ">>> $*"; python -m "$@" 2>&1 | grep -v DeprecationWarning | tee "$LOG/$name.log"; }

case "$stage" in
  smoke)
    # 1% subset end-to-end, catches bugs in ~2 minutes before the full run
    run smoke_subset src.make_dev_subset --data-dir "$DATA" --out-dir "$ROOT/dev_data" --frac 0.01
    run smoke_prepare src.prepare --data-dir "$ROOT/dev_data" --work-dir "$ROOT/dev_work"
    run smoke_bienc src.biencoder --data-dir "$ROOT/dev_data" --work-dir "$ROOT/dev_work" --max-pairs 30000 --bs 256 --overwrite "$@"
    run smoke_retrieve src.retrieve --data-dir "$ROOT/dev_data" --work-dir "$ROOT/dev_work"
    ;;
  prepare)  run prepare  src.prepare   --data-dir "$DATA" --work-dir "$WORK" "$@" ;;
  bienc)    run bienc    src.biencoder --data-dir "$DATA" --work-dir "$WORK" "$@" ;;
  retrieve) run retrieve src.retrieve  --data-dir "$DATA" --work-dir "$WORK" "$@" ;;
  all1)
    run prepare  src.prepare   --data-dir "$DATA" --work-dir "$WORK"
    run bienc    src.biencoder --data-dir "$DATA" --work-dir "$WORK"
    run retrieve src.retrieve  --data-dir "$DATA" --work-dir "$WORK"
    ;;
  *) echo "unknown stage '$stage'"; exit 1 ;;
esac
