#!/usr/bin/env bash
# Experiment v5: a second, larger cross-encoder (intfloat/multilingual-e5-base, MIT, 278M params) trained
# on 2x more queries and added to the final ranker next to the e5-small one. Every v3 file stays usable:
# scored/ and the final model are restored to v3 at the end; v5 lives in scored_v5_*, output_v5, output_v5pf.
#   MODEL=/workspace/AWS/models/multilingual-e5-base nohup bash overnight.sh > ../../../logs/v5.out 2>&1 &
set -eo pipefail
cd "$(dirname "$0")"
ROOT=$(cd ../../.. && pwd); D=${DATA:-$ROOT/student_resource/dataset}; W=${WORK:-$ROOT/work}; L=$ROOT/logs/exp; mkdir -p "$L"
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
MODEL=${MODEL:-intfloat/multilingual-e5-base}; T=${T:-0.75}; V=$ROOT/student_resource/utils/validate_submission.py
quiet() { grep --line-buffered -v -E "Deprecation|explode|empty_as_null|warn"; }
OR=${OUTROOT:-$ROOT}   # where output_v5 / output_v5pf go (output_v3pf is read from here too)
free_gb=$(df -Pk "$W" | awk 'NR==2 {print int($4 / 1048576)}')
[ "$free_gb" -ge 4 ] || { echo "only ${free_gb} GB free under $W, need 4"; exit 1; }

# the submitted v3 model must be backed up before scored/ and the final model are replaced
for s in train test; do [ -f "$W/scored_v3_$s.parquet" ] || cp "$W/scored/$s.parquet" "$W/scored_v3_$s.parquet"; done
for x in lgb_final.txt lgb_final_importance.json; do [ -f "$W/models/v3_$x" ] || cp "$W/models/$x" "$W/models/v3_$x"; done
restore_v3() {
  for s in train test; do cp "$W/scored_v3_$s.parquet" "$W/scored/$s.parquet"; done
  for x in lgb_final.txt lgb_final_importance.json; do cp "$W/models/v3_$x" "$W/models/$x"; done
}
trap restore_v3 EXIT  # even if a step fails, the box is left in the submitted v3 state

echo "=== 1. cross-encoder 2: $MODEL ($(date +%H:%M)) ==="
python -m src.crossencoder --data-dir "$D" --work-dir "$W" --tag 2 --model "$MODEL" --used-only \
  --max-queries "${MAXQ:-1500000}" --lr 3e-5 ${XARGS:-} 2>&1 | quiet | tee "$L/v5_xenc.log"

echo "=== 2. final ranker with both cross-encoders ($(date +%H:%M)) ==="
python -m src.ranker --data-dir "$D" --work-dir "$W" --stage final --extra-xenc 2 2>&1 | quiet | tee "$L/v5_final.log" \
  | grep -E "train rows|top features|Early|Did not|Error|Traceback" | cut -c1-600
for s in train test; do cp "$W/scored/$s.parquet" "$W/scored_v5_$s.parquet"; done
for x in lgb_final.txt lgb_final_importance.json; do cp "$W/models/$x" "$W/models/v5_$x"; done

echo "=== 3. validation, same fold-0 protocol: v3 then v5 ==="
for v in v3 v5; do
  echo "--- $v"
  python -m src.decide --data-dir "$D" --work-dir "$W" --out-dir /tmp/unused --scored-file "$W/scored_${v}_train.parquet" --eval-only 2>&1 \
    | grep -E "threshold sweep|VALIDATION|singletons|micro|country" || true
done

echo "=== 4. v5 test files (t=$T) + removal-only postfix ==="
O=$OR/output_v5; OP=$OR/output_v5pf; mkdir -p "$O" "$OP"
python -m src.decide --data-dir "$D" --work-dir "$W" --out-dir "$O" --threshold "$T" 2>&1 | grep -E "wrote /.*matching"
cp "$O/candidate_pairs.tsv" "$OP/"
python -m src.postfix --data-dir "$D" --work-dir "$W" --pred "$O/matching_results.tsv" --out "$OP/matching_results.tsv" 2>&1 | grep -E "removes|links "
python3 "$V" --matching "$OP/matching_results.tsv" --candidate "$OP/candidate_pairs.tsv" --test-dir "$D/test" | tail -1

echo "=== 5. label-free test diagnostics: v5 then v3 ==="
report() {
  python -m src.bucket_diag --data-dir "$D" --work-dir "$W" --strict 2>&1 | quiet || true
  python -m src.decoy_diag --data-dir "$D" --work-dir "$W" --t "$T" 2>&1 | grep -E "kept_frac|share of" || true
}
echo "--- v5"; report
restore_v3
echo "--- v3"; report
python -m src.silver_check --data-dir "$D" --work-dir "$W" --pred "$OR/output_v3pf/matching_results.tsv" --pred "$OP/matching_results.tsv" 2>&1 | quiet || true
echo "done $(date +%H:%M); scored/ and the final model restored to v3"
