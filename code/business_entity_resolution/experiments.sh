#!/usr/bin/env bash
# Label-free model-selection experiments (diagnostics only, not part of the submission path).
# For each variant of the final ranker: validation F0.5, per-country/per-band test precision
# estimates, test decoy acceptance, silver-pair recovery, links per entity.
#   bash experiments.sh          # v3 report + variants v4a, v4b   (~40 min)
set -eo pipefail
cd "$(dirname "$0")"
ROOT=$(cd ../../.. && pwd); D=$ROOT/student_resource/dataset; W=$ROOT/work; L=$ROOT/logs/exp; mkdir -p "$L"
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
T=${T:-0.75}
BASE="cl_same_qnum,cl_same_snum,cl_n,pr_s_n"                                         # removed in v3
S1SIDE="s_n,rs,s_rank,gap_s_best,xe_srank,xe_sgap,xe_s_npos,pa_srank,pa_sgap"       # counts / ranks over the entity's other candidates
NAMEFREQ="s_core_freq,s_fullname_freq"                                               # name frequency in the S1 table

report() {  # scored/ must hold the variant; output_$1 must exist
  local O=$ROOT/output_$1
  echo "=============== $1 ==============="
  python -m src.bucket_diag --data-dir "$D" --work-dir "$W" 2>&1 | grep -v -E "Deprecation|explode"
  python -m src.decoy_diag --data-dir "$D" --work-dir "$W" --t "$T" 2>&1 | grep -E "kept_frac|share of"
  python -m src.silver_check --data-dir "$D" --work-dir "$W" --pred "$O/matching_results.tsv" 2>&1 | grep -v -E "Deprecation|explode"
  python - "$O" "$W" <<'EOF'
import sys, polars as pl
O, W = sys.argv[1], sys.argv[2]
s1 = pl.read_parquet(f"{W}/test/s1.parquet", columns=["entity_id", "country"]).rename({"entity_id": "source1_entity_id"})
d = pl.read_csv(f"{O}/matching_results.tsv", separator="\t", quote_char=None, infer_schema=False)
d = d.with_columns(pl.col("matched_entity_ids").fill_null("").str.split(",").list.eval(pl.element().filter(pl.element() != "")).list.len().alias("k")).join(s1, on="source1_entity_id")
print("links per S1:", dict(d.group_by("country").agg(pl.col("k").mean().round(4)).sort("country").iter_rows()), "| total", d["k"].sum())
EOF
}

variant() {  # $1 tag, $2 comma list of final-stage features to drop
  echo "=============== train $1 (drop: $2) ==============="
  python -m src.ranker --data-dir "$D" --work-dir "$W" --stage final --final-drop "$2" 2>&1 | grep -E "top features|Error|Traceback" | cut -c1-300
  python -m src.decide --data-dir "$D" --work-dir "$W" --out-dir "$ROOT/output_$1" 2>&1 | grep -E "best threshold|VALIDATION|singletons|micro|country"
  python -m src.decide --data-dir "$D" --work-dir "$W" --out-dir "$ROOT/output_$1" --threshold "$T" 2>&1 | grep -E "wrote /.*matching"
  for s in train test; do cp "$W/scored/$s.parquet" "$W/scored_${1}_$s.parquet"; done
  report "$1"
}

for s in train test; do [ -f "$W/scored_v3_$s.parquet" ] || cp "$W/scored/$s.parquet" "$W/scored_v3_$s.parquet"; done
report v3_t075 2>&1 | tee "$L/v3.log"
variant v4a "$BASE,$S1SIDE" 2>&1 | tee "$L/v4a.log"
variant v4b "$BASE,$S1SIDE,$NAMEFREQ" 2>&1 | tee "$L/v4b.log"
# leave the box in the v3 state (the submitted model)
for s in train test; do cp "$W/scored_v3_$s.parquet" "$W/scored/$s.parquet"; done
echo "done; scored/ restored to v3"
