"""Step 3b: give every unmatched training query the fold of the Source 1 entity it
competes with (its top-1 retrieved candidate).

Matched queries already inherit their true entity's fold. Unmatched ones (decoys)
were hashed into a random fold, so the decoys competing with validation entities
were mostly queries the rankers had been trained on, which made false merges look
rarer on validation than on test. After this step a decoy lives in the same fold
as the entity it would be wrongly merged into, so validation stays out-of-sample.
Idempotent; rewrites train/q.parquet.
"""
import argparse

import polars as pl

from .common import Paths, log


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--work-dir", required=True)
    args = ap.parse_args()
    P = Paths(args.data_dir, args.work_dir)
    q = pl.read_parquet(P.w("train", "q.parquet"))
    s1 = pl.read_parquet(P.w("train", "s1.parquet"), columns=["idx", "fold"])
    top = (pl.read_parquet(P.w("cands", "train.parquet"), columns=["q_idx", "s1_idx", "cos"])
             .sort("q_idx", "cos", descending=[False, True]).group_by("q_idx", maintain_order=True).first()
             .join(s1.select(pl.col("idx").alias("s1_idx"), pl.col("fold").alias("nf")), on="s1_idx"))
    before = q["fold"].to_numpy().copy()
    q = (q.join(top.select(pl.col("q_idx").alias("idx"), "nf"), on="idx", how="left", maintain_order="left")
          .with_columns(pl.when(pl.col("true_s1").is_null() & pl.col("nf").is_not_null())
                          .then(pl.col("nf")).otherwise(pl.col("fold")).cast(pl.Int8).alias("fold"))
          .drop("nf").sort("idx"))
    changed = int((q["fold"].to_numpy() != before).sum())
    q.write_parquet(P.w("train", "q.parquet"))
    unm = q.filter(pl.col("true_s1").is_null())
    log.info(f"refold: {changed} unmatched queries moved to their top-1 entity's fold "
             f"({unm.height} unmatched in total); fold sizes: {dict(q['fold'].value_counts().sort('fold').iter_rows())}")


if __name__ == "__main__":
    main()
