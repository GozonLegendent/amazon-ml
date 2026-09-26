"""Label-free sanity check of test predictions, per country (France included).

Builds high-confidence "silver" pairs from the test inputs with a simple rule:
same canonical core name, same first house number, at least one shared street
word, and the (country, core name, number) key belongs to exactly one Source 1
entity. On the US training labels this rule is ~99.8% precise. It only covers
easy matches, so it measures misses of easy matches and conflicting assignments,
not the full metric. The silver pairs are never used for training; together with
bucket_diag.py they served as label-free evidence when the submitted threshold (0.75)
was chosen.

Usage: python -m src.silver_check --data-dir D --work-dir W --pred A.tsv [--pred B.tsv ...]
"""
import argparse

import polars as pl

from .common import Paths, log


def silver(P):
    s1 = pl.read_parquet(P.w("test", "s1.parquet"), columns=["idx", "entity_id", "country", "ncore", "anum", "astreet"])
    q = pl.read_parquet(P.w("test", "q.parquet"), columns=["idx", "entity_id", "country", "ncore", "anum", "astreet"])
    key = lambda d: d.with_columns(pl.col("anum").str.split(" ").list.first().alias("fn"),
                                   pl.col("astreet").str.split(" ").alias("st")).filter(
        (pl.col("ncore") != "") & pl.col("fn").is_not_null() & (pl.col("fn") != ""))
    a, b = key(s1), key(q)
    uniq = a.group_by("country", "ncore", "fn").len().filter(pl.col("len") == 1).drop("len")
    a = a.join(uniq, on=["country", "ncore", "fn"])
    j = b.join(a, on=["country", "ncore", "fn"], suffix="_s1")
    j = j.filter(pl.col("st").list.set_intersection(pl.col("st_s1")).list.len() > 0)
    return j.select(pl.col("entity_id").alias("qid"), pl.col("entity_id_s1").alias("sid"), "country")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--work-dir", required=True)
    ap.add_argument("--pred", action="append", required=True)
    args = ap.parse_args()
    P = Paths(args.data_dir, args.work_dir)
    sv = silver(P)
    log.info(f"silver pairs: {dict(sv['country'].value_counts().iter_rows())}")
    for path in args.pred:
        pr = (pl.read_csv(path, separator="\t", quote_char=None, infer_schema=False)
                .with_columns(pl.col("matched_entity_ids").fill_null("").str.split(","))
                .explode("matched_entity_ids").filter(pl.col("matched_entity_ids") != "")
                .select(pl.col("matched_entity_ids").alias("qid"), pl.col("source1_entity_id").alias("pid")))
        m = sv.join(pr, on="qid", how="left")
        res = m.group_by("country").agg(
            (pl.col("pid") == pl.col("sid")).fill_null(False).mean().alias("silver_found"),
            pl.col("pid").is_null().mean().alias("unassigned"),
            ((pl.col("pid") != pl.col("sid")) & pl.col("pid").is_not_null()).mean().alias("sent_to_other_s1"),
            pl.len().alias("n")).sort("country")
        log.info(f"{path}: {pr.height} predicted pairs")
        print(res)


if __name__ == "__main__":
    main()
