"""Per-country, per-probability-band precision estimates on TEST (no labels used for fitting).

Two independent estimates per band:
  * silver precision - among test queries in the label-free "silver" set (same core name,
    same house number, shared street word, unique on the S1 side), the share of the
    model's links in this band that point to the silver entity. Covers every country,
    France included, but only easy queries.
  * implied precision (US / India only) - validation true links per S1 in the band
    divided by test links per S1 in the band (both splits come from the same generator,
    so true links per entity per band should match; the excess on test is false links).
Under macro F0.5 a link is worth keeping when it is right at least ~77% of the time,
so the best per-country threshold is the lowest band edge above which estimates stay
above that break-even.
"""
import argparse

import polars as pl

from .common import Paths, log
from .silver_check import silver

EDGES = [0.36, 0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.97, 0.99, 1.0001]


def band(col="p"):
    e = pl.lit(None, pl.Utf8)
    for lo, hi in zip(EDGES[:-1], EDGES[1:]):
        e = pl.when((pl.col(col) >= lo) & (pl.col(col) < hi)).then(pl.lit(f"{lo:.2f}-{min(hi, 1):.2f}")).otherwise(e)
    return e.alias("band")


def argmax(P, split):
    sc = pl.read_parquet(P.w("scored", f"{split}.parquet"), columns=["q_idx", "s1_idx", "p"])
    return sc.sort("q_idx", "p", descending=[False, True]).group_by("q_idx", maintain_order=True).first()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--work-dir", required=True)
    ap.add_argument("--strict", action="store_true", help="strict silver (real street word + city present)")
    args = ap.parse_args()
    P = Paths(args.data_dir, args.work_dir)
    pl.Config.set_tbl_rows(-1); pl.Config.set_tbl_cols(-1); pl.Config.set_tbl_width_chars(250)

    # validation: true links per S1 per band (fold-0 entities, fold-0 queries)
    s1v = pl.read_parquet(P.w("train", "s1.parquet"), columns=["idx", "fold", "country"])
    qv = pl.read_parquet(P.w("train", "q.parquet"), columns=["idx", "fold", "true_s1"])
    va = (argmax(P, "train").join(s1v.rename({"idx": "s1_idx", "fold": "sf"}), on="s1_idx")
            .join(qv.rename({"idx": "q_idx", "fold": "qf"}), on="q_idx")
            .filter((pl.col("sf") == 0) & (pl.col("qf") == 0) & (pl.col("p") >= EDGES[0]))
            .with_columns((pl.col("true_s1") == pl.col("s1_idx")).fill_null(False).alias("y"), band()))
    nv = s1v.filter(pl.col("fold") == 0).group_by("country").len().rename({"len": "nS1_val"})
    vb = (va.group_by("country", "band").agg(pl.col("y").sum().alias("val_true"), pl.len().alias("val_links"))
            .join(nv, on="country").with_columns((pl.col("val_true") / pl.col("nS1_val")).alias("val_true_perS1"),
                                                  (pl.col("val_true") / pl.col("val_links")).alias("val_prec")))

    # test: links per S1 per band, silver precision per band
    s1t = pl.read_parquet(P.w("test", "s1.parquet"), columns=["idx", "entity_id", "country"])
    qt = pl.read_parquet(P.w("test", "q.parquet"), columns=["idx", "entity_id"])
    te = (argmax(P, "test").filter(pl.col("p") >= EDGES[0]).join(s1t.rename({"idx": "s1_idx"}), on="s1_idx")
            .join(qt.rename({"idx": "q_idx", "entity_id": "qid"}), on="q_idx").with_columns(band()))
    sv = silver(P, strict=args.strict).select("qid", "sid")
    log.info(f"silver mode: {'strict' if args.strict else 'loose'} | silver pairs {sv.height}")
    te = te.join(sv, on="qid", how="left").with_columns((pl.col("entity_id") == pl.col("sid")).alias("silver_ok"))
    nt = s1t.group_by("country").len().rename({"len": "nS1_test"})
    tb = (te.group_by("country", "band").agg(pl.len().alias("test_links"), pl.col("sid").is_not_null().sum().alias("silver_links"),
                                             pl.col("silver_ok").sum().alias("silver_right"))
            .join(nt, on="country").with_columns((pl.col("test_links") / pl.col("nS1_test")).alias("test_perS1"),
                                                  (pl.col("silver_right") / pl.col("silver_links")).alias("silver_prec")))
    out = (tb.join(vb.select("country", "band", "val_true_perS1", "val_prec"), on=["country", "band"], how="left")
             .with_columns((pl.col("val_true_perS1") / pl.col("test_perS1")).alias("implied_prec"))
             .select("country", "band", "test_links", "silver_links", pl.col("silver_prec").round(3),
                     pl.col("implied_prec").round(3), pl.col("val_prec").round(3))
             .sort("country", "band"))
    print(out)
    log.info("keep a band when its precision is >= ~0.77 (macro F0.5 break-even)")


if __name__ == "__main__":
    main()
