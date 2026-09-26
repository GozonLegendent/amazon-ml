"""Label-free diagnosis: where do test predictions differ from validation?

For the argmax link of every query (validation-fold queries on train, all queries
on test) it reports, per country: predicted links per Source 1 entity, the
probability distribution of kept links, and how often kept links carry decoy
signatures (moved house number, added Group/Holdings, number conflict, extra
words). On validation the same statistics are split into true and false links,
so a test profile that looks like the validation false links points to where the
extra test errors come from. Also counts links per country in submission files.
"""
import argparse
import json

import numpy as np
import polars as pl

from .common import Paths, log

SIG = ["num_move", "leg_add_strong", "hn_conflict", "nothing_added", "qx_n", "hn_eq", "qx_unk"]
BUCKETS = [0.0, 0.5, 0.75, 0.9, 0.97, 1.01]


def profile(df, t, label):
    kept = df.filter(pl.col("p") >= t)
    out = {"links": kept.height}
    if kept.height == 0:
        print(f"  {label:34s} links=0")
        return
    for lo, hi in zip(BUCKETS[:-1], BUCKETS[1:]):
        lo2 = max(lo, t)
        if lo2 < hi:
            out[f"p[{lo2:.2f},{min(hi, 1):.2f})"] = round(kept.filter((pl.col("p") >= lo2) & (pl.col("p") < hi)).height / max(1, kept.height), 4)
    for s in SIG:
        v = kept[s]
        out[s] = round(float(((v > 0) if s in ("qx_n", "qx_unk") else (v == 1)).mean()), 4)
    print(f"  {label:34s} " + " ".join(f"{k}={v}" for k, v in out.items()))


def load(P, split):
    sc = pl.read_parquet(P.w("scored", f"{split}.parquet"), columns=["q_idx", "s1_idx", "p"])
    f2 = pl.read_parquet(P.w("feats2", f"{split}.parquet"), columns=SIG)
    df = pl.concat([sc, f2], how="horizontal")
    df = df.sort("q_idx", "p", descending=[False, True]).group_by("q_idx", maintain_order=True).first()
    s1 = pl.read_parquet(P.w(split, "s1.parquet"), columns=["idx", "country"] + (["fold"] if split == "train" else []))
    return df.join(s1.rename({"idx": "s1_idx"}), on="s1_idx"), s1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--work-dir", required=True)
    ap.add_argument("--t", type=float, default=None, help="threshold (default: from decision.json)")
    ap.add_argument("--pred", action="append", default=[], help="submission TSVs to count per country")
    args = ap.parse_args()
    P = Paths(args.data_dir, args.work_dir)
    t = args.t if args.t is not None else json.load(open(P.w("decision.json")))["threshold"]
    log.info(f"threshold {t}")

    va, s1v = load(P, "train")
    q = pl.read_parquet(P.w("train", "q.parquet"), columns=["idx", "fold", "true_s1"])
    va = (va.filter(pl.col("fold") == 0).join(q.select(pl.col("idx").alias("q_idx"), pl.col("fold").alias("qf"), "true_s1"), on="q_idx")
            .filter(pl.col("qf") == 0).with_columns((pl.col("true_s1") == pl.col("s1_idx")).fill_null(False).alias("y")))
    n_val = s1v.filter(pl.col("fold") == 0).group_by("country").len()
    print("\n=== VALIDATION (fold 0) ===")
    for (c,), g in va.group_by("country"):
        n = n_val.filter(pl.col("country") == c)["len"][0]
        kept = g.filter(pl.col("p") >= t)
        print(f"[{c}] links per S1 {kept.height / n:.3f} | precision {kept['y'].mean():.4f} | "
              f"true links per S1 {g.filter(pl.col('y')).height / n:.3f}")
        profile(g.filter(pl.col("y")), t, "true links")
        profile(g.filter(~pl.col("y")), t, "false links")
        profile(g, t, "all kept")

    te, s1t = load(P, "test")
    n_te = s1t.group_by("country").len()
    print("\n=== TEST ===")
    for (c,), g in te.group_by("country"):
        n = n_te.filter(pl.col("country") == c)["len"][0]
        print(f"[{c}] links per S1 {g.filter(pl.col('p') >= t).height / n:.3f} | queries with a candidate per S1 {g.height / n:.3f}")
        profile(g, t, "all kept")

    if args.pred:
        ids = s1t.join(pl.read_parquet(P.w("test", "s1.parquet"), columns=["idx", "entity_id"]), on="idx").select(
            pl.col("entity_id").alias("source1_entity_id"), "country")
        print("\n=== submission files: links per S1 by country ===")
        for path in args.pred:
            d = (pl.read_csv(path, separator="\t", quote_char=None, infer_schema=False)
                   .with_columns(pl.col("matched_entity_ids").fill_null("").str.split(",").list.eval(
                       pl.element().filter(pl.element() != "")).list.len().alias("n"))
                   .join(ids, on="source1_entity_id"))
            r = d.group_by("country").agg(pl.col("n").mean().round(3).alias("links_per_s1"),
                                          (pl.col("n") == 0).mean().round(4).alias("empty_frac")).sort("country")
            print(path, dict(zip(r["country"], zip(r["links_per_s1"], r["empty_frac"]))))


if __name__ == "__main__":
    main()
