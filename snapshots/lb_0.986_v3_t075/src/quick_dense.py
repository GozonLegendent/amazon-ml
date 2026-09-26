"""Quick baseline straight from blocking: assign each query to its bi-encoder top-1
Source 1 entity when the cosine and the margin over the runner-up clear
thresholds tuned for macro F0.5 on the validation fold."""
import argparse
import os

import numpy as np
import polars as pl

from .common import Paths, log, timer
from .decide import f05_macro


def top2(P, split):
    c = (pl.scan_parquet(P.w("cands", f"{split}.parquet")).filter(pl.col("rq") <= 1)
           .select("q_idx", "s1_idx", "cos", "rq").collect())
    b = c.filter(pl.col("rq") == 0).select("q_idx", "s1_idx", "cos")
    s = c.filter(pl.col("rq") == 1).select("q_idx", pl.col("cos").alias("cos2"))
    return b.join(s, on="q_idx", how="left").with_columns(
        (pl.col("cos") - pl.col("cos2").fill_null(-1.0)).alias("margin"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--work-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()
    P = Paths(args.data_dir, args.work_dir)
    with timer("tune cosine/margin thresholds on validation"):
        tr = top2(P, "train")
        s1 = pl.read_parquet(P.w("train", "s1.parquet"), columns=["idx", "fold"])
        val = s1.filter(pl.col("fold") == 0).select(pl.col("idx").alias("s1_idx"))
        gt = pl.read_parquet(P.w("train", "gt.parquet")).join(val, on="s1_idx")
        tv = tr.join(val, on="s1_idx")
        ts = np.unique(np.quantile(tv["cos"].to_numpy(), np.linspace(0.01, 0.7, 36)))
        best = (-1.0, 0.0, 0.0)
        for t in ts:
            for m in (0.0, 0.005, 0.01, 0.02, 0.03, 0.05):
                pred = tv.filter((pl.col("cos") >= t) & (pl.col("margin") >= m)).select("s1_idx", "q_idx")
                f, _ = f05_macro(pred, gt, val)
                if f > best[0]:
                    best = (f, float(t), m)
        f, t, m = best
        log.info(f"VALIDATION macro F0.5 (dense top-1 baseline) = {f:.5f} at cos>={t:.4f}, margin>={m}")
    with timer("write test matching_results.tsv"):
        te = top2(P, "test").filter((pl.col("cos") >= t) & (pl.col("margin") >= m))
        s1t = pl.read_parquet(P.w("test", "s1.parquet"), columns=["idx", "entity_id"]).rename(
            {"idx": "s1_idx", "entity_id": "source1_entity_id"})
        qid = pl.read_parquet(P.w("test", "q.parquet"), columns=["idx", "entity_id"]).rename(
            {"idx": "q_idx", "entity_id": "qid"})
        agg = (te.select("s1_idx", "q_idx").join(qid, on="q_idx").sort("s1_idx", "qid")
                 .group_by("s1_idx", maintain_order=True).agg(pl.col("qid").str.join(",").alias("matched_entity_ids")))
        out = (s1t.join(agg, on="s1_idx", how="left").with_columns(pl.col("matched_entity_ids").fill_null(""))
                  .sort("s1_idx").select("source1_entity_id", "matched_entity_ids"))
        os.makedirs(args.out_dir, exist_ok=True)
        path = os.path.join(args.out_dir, "matching_results.tsv")
        out.write_csv(path, separator="\t", quote_style="never")
        log.info(f"wrote {path}: {out.height} rows, {(out['matched_entity_ids'] != '').sum()} non-empty, {te.height} ids")


if __name__ == "__main__":
    main()
