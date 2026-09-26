#!/usr/bin/env bash
# Experiment v6: a third cross-encoder, BAAI/bge-reranker-v2-m3 (Apache-2.0, 568M params, XLM-R large already
# trained as a multilingual cross-encoder), added to the final ranker next to the e5-small and e5-base ones.
# Waits until overnight.sh (v5) has finished, then runs. Like overnight.sh it leaves the box in the v3 state;
# v6 lives in scored_v6_*, output_v6, output_v6pf.
#   nohup bash overnight2.sh > ../../../logs/v6.out 2>&1 &
set -eo pipefail
cd "$(dirname "$0")"
ROOT=$(cd ../../.. && pwd); D=${DATA:-$ROOT/student_resource/dataset}; W=${WORK:-$ROOT/work}; L=$ROOT/logs/exp; mkdir -p "$L"
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
REPO=BAAI/bge-reranker-v2-m3; REV=953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e
MODEL=${MODEL:-$ROOT/models/bge-reranker-v2-m3}; T=${T:-0.75}; V=$ROOT/student_resource/utils/validate_submission.py
OR=${OUTROOT:-$ROOT}
quiet() { grep --line-buffered -v -E "Deprecation|explode|empty_as_null|warn"; }

echo "waiting for overnight.sh (v5) to finish ($(date +%H:%M))"
# grep without -q reads all of ps output: with -q, ps can die of SIGPIPE and pipefail ends the wait early
while ps aux | grep "[o]vernight[.]sh" >/dev/null; do sleep 60; done
echo "v5 finished ($(date +%H:%M)): $(tail -1 "$ROOT/logs/v5.out" 2>/dev/null)"
EXTRA=3; [ -f "$W/xenc2/test.npy" ] && [ -f "$W/xenc2/train.npy" ] && EXTRA=2,3
free_gb=$(df -Pk "$W" | awk 'NR==2 {print int($4 / 1048576)}')
[ "$free_gb" -ge "${MINFREE:-6}" ] || { echo "only ${free_gb} GB free under $W, need ${MINFREE:-6}"; exit 1; }

if [ ! -f "$MODEL/model.safetensors" ]; then
  HF_HUB_OFFLINE=0 python -c "
from huggingface_hub import snapshot_download
snapshot_download('$REPO', revision='$REV', local_dir='$MODEL', allow_patterns=['config.json', 'model.safetensors',
  'tokenizer.json', 'tokenizer_config.json', 'sentencepiece.bpe.model', 'special_tokens_map.json'])"
fi
ls -l "$MODEL"

for s in train test; do [ -f "$W/scored_v3_$s.parquet" ] || cp "$W/scored/$s.parquet" "$W/scored_v3_$s.parquet"; done
for x in lgb_final.txt lgb_final_importance.json; do [ -f "$W/models/v3_$x" ] || cp "$W/models/$x" "$W/models/v3_$x"; done
restore_v3() {
  for s in train test; do cp "$W/scored_v3_$s.parquet" "$W/scored/$s.parquet"; done
  for x in lgb_final.txt lgb_final_importance.json; do cp "$W/models/v3_$x" "$W/models/$x"; done
}
trap restore_v3 EXIT  # even if a step fails, the box is left in the submitted v3 state

echo "=== 1. cross-encoder 3: $REPO ($(date +%H:%M)) ==="
python -m src.crossencoder --data-dir "$D" --work-dir "$W" --tag 3 --model "$MODEL" --used-only \
  --max-queries "${MAXQ:-800000}" --lr "${LR:-2e-5}" ${XARGS:-} 2>&1 | quiet | tee "$L/v6_xenc.log"

echo "=== 2. final ranker, extra cross-encoders: $EXTRA ($(date +%H:%M)) ==="
python -m src.ranker --data-dir "$D" --work-dir "$W" --stage final --extra-xenc "$EXTRA" 2>&1 | quiet | tee "$L/v6_final.log" \
  | grep -E "train rows|top features|Early|Did not|Error|Traceback" | cut -c1-600
for s in train test; do cp "$W/scored/$s.parquet" "$W/scored_v6_$s.parquet"; done
for x in lgb_final.txt lgb_final_importance.json; do cp "$W/models/$x" "$W/models/v6_$x"; done

echo "=== 3. validation, same fold-0 protocol: v3, v5, v6 ==="
for v in v3 v5 v6; do
  [ -f "$W/scored_${v}_train.parquet" ] || continue
  echo "--- $v"
  python -m src.decide --data-dir "$D" --work-dir "$W" --out-dir /tmp/unused --scored-file "$W/scored_${v}_train.parquet" --eval-only 2>&1 \
    | grep -E "threshold sweep|VALIDATION|singletons|micro|country" || true
done

echo "=== 4. v6 test files (t=$T) + removal-only postfix ==="
O=$OR/output_v6; OP=$OR/output_v6pf; mkdir -p "$O" "$OP"
python -m src.decide --data-dir "$D" --work-dir "$W" --out-dir "$O" --threshold "$T" 2>&1 | grep -E "wrote /.*matching"
cp "$O/candidate_pairs.tsv" "$OP/"
python -m src.postfix --data-dir "$D" --work-dir "$W" --pred "$O/matching_results.tsv" --out "$OP/matching_results.tsv" 2>&1 | grep -E "removes|links "
python3 "$V" --matching "$OP/matching_results.tsv" --candidate "$OP/candidate_pairs.tsv" --test-dir "$D/test" | tail -1

echo "=== 5. label-free test diagnostics (v6) ==="
python -m src.bucket_diag --data-dir "$D" --work-dir "$W" --strict 2>&1 | quiet || true
python -m src.decoy_diag --data-dir "$D" --work-dir "$W" --t "$T" 2>&1 | grep -E "kept_frac|share of" || true
restore_v3
python -m src.silver_check --data-dir "$D" --work-dir "$W" --pred "$OR/output_v3pf/matching_results.tsv" --pred "$OR/output_v5pf/matching_results.tsv" \
  --pred "$OP/matching_results.tsv" 2>&1 | quiet || true
echo "done $(date +%H:%M); scored/ and the final model restored to v3"
