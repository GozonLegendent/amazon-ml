"""Error analysis on the validation fold: prints sampled false merges and missed
matches with their texts and scores, plus where the misses were lost
(blocking, pruning, or the final decision).
"""
import argparse
import json

import polars as pl

from .common import Paths, log


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--work-dir", required=True)
    ap.add_argument("--n", type=int, default=15)
    ap.add_argument("--country", default=None)
    args = ap.parse_args()
    P = Paths(args.data_dir, args.work_dir)
    pl.Config.set_fmt_str_lengths(70)
    pl.Config.set_tbl_width_chars(260)
    pl.Config.set_tbl_rows(args.n + 5)
    t = json.load(open(P.w("decision.json")))["threshold"]
    s1 = pl.read_parquet(P.w("train", "s1.parquet"), columns=["idx", "fold", "country", "business_name", "business_address"])
    q = pl.read_parquet(P.w("train", "q.parquet"), columns=["idx", "business_name", "business_address", "true_s1"])
    val = s1.filter(pl.col("fold") == 0)
    if args.country:
        val = val.filter(pl.col("country") == args.country)
    vset = val.select(pl.col("idx").alias("s1_idx"))
    gt = pl.read_parquet(P.w("train", "gt.parquet")).join(vset, on="s1_idx")
    cands = pl.read_parquet(P.w("cands", "train.parquet"), columns=["q_idx", "s1_idx"]).with_columns(pl.lit(True).alias("in_cands"))
    pruned = pl.read_parquet(P.w("pruned", "train.parquet"), columns=["q_idx", "s1_idx"]).with_columns(pl.lit(True).alias("in_pruned"))
    scored = pl.read_parquet(P.w("scored", "train.parquet"))
    best = (scored.sort("q_idx", "p", descending=[False, True]).group_by("q_idx", maintain_order=True)
                  .agg(pl.col("s1_idx").first().alias("best_s1"), pl.col("p").first().alias("best_p")))
    pred = best.filter(pl.col("best_p") >= t).select("q_idx", pl.col("best_s1").alias("s1_idx")).join(vset, on="s1_idx")

    # misses: where were they lost?
    m = (gt.join(pred.with_columns(pl.lit(True).alias("hit")), on=["q_idx", "s1_idx"], how="left")
           .filter(pl.col("hit").is_null())
           .join(cands, on=["q_idx", "s1_idx"], how="left").join(pruned, on=["q_idx", "s1_idx"], how="left")
           .join(scored.select("q_idx", "s1_idx", pl.col("p").alias("p_true")), on=["q_idx", "s1_idx"], how="left")
           .join(best, on="q_idx", how="left"))
    stage = (pl.when(pl.col("in_cands").is_null()).then(pl.lit("lost_in_blocking"))
               .when(pl.col("in_pruned").is_null()).then(pl.lit("lost_in_pruning"))
               .when(pl.col("best_s1") != pl.col("s1_idx")).then(pl.lit("other_s1_won"))
               .otherwise(pl.lit("below_threshold")))
    m = m.with_columns(stage.alias("stage"))
    log.info(f"validation true pairs {gt.height}, missed {m.height}: {dict(m['stage'].value_counts().iter_rows())}")

    def show(df, title, s1col, qcol):
        df = (df.join(s1.select(pl.col("idx").alias(s1col), pl.col("business_name").alias("s1_name"),
                               pl.col("business_address").alias("s1_addr")), on=s1col)
                .join(q.select(pl.col("idx").alias(qcol), pl.col("business_name").alias("q_name"),
                               pl.col("business_address").alias("q_addr")), on=qcol))
        print(f"\n===== {title} =====")
        for r in df.head(args.n).iter_rows(named=True):
            extra = {k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()
                     if k in ("stage", "p_true", "best_p", "p")}
            print(f"{extra}\n   S1: {r['s1_name']} | {r['s1_addr']}\n   Q : {r['q_name']} | {r['q_addr']}")

    for st in ("lost_in_blocking", "lost_in_pruning", "other_s1_won", "below_threshold"):
        show(m.filter(pl.col("stage") == st).sample(min(args.n, m.filter(pl.col("stage") == st).height), seed=0),
             f"MISSED ({st})", "s1_idx", "q_idx")

    fp = (pred.join(gt.with_columns(pl.lit(True).alias("ok")), on=["q_idx", "s1_idx"], how="left")
              .filter(pl.col("ok").is_null())
              .join(q.select(pl.col("idx").alias("q_idx"), "true_s1"), on="q_idx")
              .join(best.select("q_idx", pl.col("best_p").alias("p")), on="q_idx"))
    log.info(f"false merges into validation entities: {fp.height} "
             f"(record matches nothing: {fp['true_s1'].is_null().sum()}, "
             f"record belongs to another entity: {fp['true_s1'].is_not_null().sum()})")
    show(fp.sample(min(args.n, fp.height), seed=0), "FALSE MERGES", "s1_idx", "q_idx")


if __name__ == "__main__":
    main()
