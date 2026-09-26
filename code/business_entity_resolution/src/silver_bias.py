"""Diagnostic: can silver precision stand in for true precision?

On validation (fold-0 entities and fold-0 queries, US / India, labels known) it reports per band:
  prec_all       - true precision of all argmax links in the band
  silver_agree   - share of silver-covered links that point to the silver entity (the number
                   bucket_diag measures on test, where no labels exist)
  prec_nonsilver - true precision of the links that silver does not cover
plus the accuracy of the silver labels themselves. If silver_agree tracks prec_all, the French
silver numbers can be read as French precision; if silver_agree stays near 1 while prec_all
falls, silver only certifies easy links and says little about a band.
Usage: python -m src.silver_bias --data-dir D --work-dir W
"""
import argparse

import polars as pl

from .bucket_diag import EDGES, argmax, band
from .common import Paths, log
from .silver_check import silver


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--work-dir", required=True)
    args = ap.parse_args()
    P = Paths(args.data_dir, args.work_dir)
    pl.Config.set_tbl_rows(-1); pl.Config.set_tbl_cols(-1); pl.Config.set_tbl_width_chars(250)

    s1 = pl.read_parquet(P.w("train", "s1.parquet"), columns=["idx", "entity_id", "fold", "country"])
    q = pl.read_parquet(P.w("train", "q.parquet"), columns=["idx", "entity_id", "fold", "true_s1"])
    q = q.join(s1.select(pl.col("idx").alias("true_s1"), pl.col("entity_id").alias("true_id")), on="true_s1", how="left")
    va = (argmax(P, "train").filter(pl.col("p") >= EDGES[0])
            .join(s1.select(pl.col("idx").alias("s1_idx"), pl.col("entity_id").alias("pid"), pl.col("fold").alias("sf"), "country"),
                  on="s1_idx")
            .join(q.select(pl.col("idx").alias("q_idx"), pl.col("entity_id").alias("qid"), pl.col("fold").alias("qf"), "true_id"),
                  on="q_idx")
            .filter((pl.col("sf") == 0) & (pl.col("qf") == 0))
            .with_columns((pl.col("pid") == pl.col("true_id")).fill_null(False).alias("y"), band()))

    for strict in (False, True):
        mode = "strict" if strict else "loose"
        sv = silver(P, strict=strict, split="train").select("qid", "sid")
        acc = (sv.join(q.select(pl.col("entity_id").alias("qid"), "true_id", "fold"), on="qid").filter(pl.col("fold") == 0))
        la = (acc["sid"] == acc["true_id"]).fill_null(False).mean() if acc.height else float("nan")
        log.info(f"{mode} silver, fold-0 queries: {acc.height} pairs, silver label accuracy {la:.4f}")
        t = va.join(sv, on="qid", how="left")
        out = (t.group_by("country", "band").agg(
                    pl.len().alias("links"), pl.col("y").mean().round(3).alias("prec_all"),
                    pl.col("sid").is_not_null().sum().alias("silver_links"),
                    (pl.col("pid") == pl.col("sid")).filter(pl.col("sid").is_not_null()).mean().round(3).alias("silver_agree"),
                    pl.col("y").filter(pl.col("sid").is_null()).mean().round(3).alias("prec_nonsilver"))
                 .sort("country", "band"))
        print(f"--- {mode} silver on validation ---")
        print(out)


if __name__ == "__main__":
    main()
