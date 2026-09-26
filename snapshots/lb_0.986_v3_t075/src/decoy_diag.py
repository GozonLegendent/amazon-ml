"""Why are decoy-signature pairs accepted on test but rejected on validation?

A decoy signature = a house number moved up by a small offset AND a legal word or
extra name word added (on training labels ~89% of such pairs are decoys, 0.15%
of true pairs have it). For those pairs this compares the final ranker's inputs
and output between validation false links, validation true links and test, so a
feature whose test values look like validation *true* links is the one fooling it.
"""
import argparse

import numpy as np
import polars as pl

from .common import Paths
from .ranker import load_matrix

OFF = [1, 2, 3, 4, 5, 7, 9, 11, 13, 21]
SHOW = ["p", "xe", "xe_qmargin", "xe_qrank", "p_a", "pa_qmargin", "cos", "margin_q", "q_rank", "rq", "rs",
        "cl_same_qnum", "cl_same_snum", "cl_n", "s_n", "q_n", "pr_q_n", "pr_s_n", "xe_s_npos",
        "qx_lo_min", "qx_unk", "leg_add_n", "hn_eq", "s_fullname_freq", "afine1_in_q", "st_tset", "a_tset", "n_tset"]


def frame(P, split):
    ids, X = load_matrix(P, split, "final")
    sc = pl.read_parquet(P.w("scored", f"{split}.parquet"), columns=["p"])
    d = pl.concat([ids.select("q_idx", "s1_idx"), X, sc], how="horizontal")
    d = d.with_columns(pl.col("p").rank("ordinal", descending=True).over("q_idx").alias("_r")).filter(pl.col("_r") == 1)
    sig = ((pl.col("num_move") == 1) & pl.col("num_move_signed").is_in(OFF) & ((pl.col("leg_add_n") > 0) | (pl.col("qx_n") > 0)))
    return d.with_columns(sig.alias("sig"))


def summ(d, name):
    cols = [c for c in SHOW if c in d.columns]
    row = {"group": name, "n": d.height}
    for c in cols:
        row[c] = round(float(d[c].mean()), 3) if d.height else None
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--work-dir", required=True)
    ap.add_argument("--t", type=float, default=0.39)
    args = ap.parse_args()
    P = Paths(args.data_dir, args.work_dir)
    tr = frame(P, "train")
    q = pl.read_parquet(P.w("train", "q.parquet"), columns=["idx", "fold", "true_s1"])
    s1 = pl.read_parquet(P.w("train", "s1.parquet"), columns=["idx", "fold"])
    tr = (tr.join(q.select(pl.col("idx").alias("q_idx"), pl.col("fold").alias("qf"), "true_s1"), on="q_idx")
            .join(s1.select(pl.col("idx").alias("s1_idx"), pl.col("fold").alias("sf")), on="s1_idx")
            .filter((pl.col("qf") == 0) & (pl.col("sf") == 0))
            .with_columns((pl.col("true_s1") == pl.col("s1_idx")).fill_null(False).alias("y")))
    te = frame(P, "test")
    rows = []
    for nm, d in (("val decoy-sig, WRONG link", tr.filter(pl.col("sig") & ~pl.col("y"))),
                  ("val decoy-sig, TRUE link", tr.filter(pl.col("sig") & pl.col("y"))),
                  ("val no-sig, TRUE link", tr.filter(~pl.col("sig") & pl.col("y"))),
                  ("test decoy-sig", te.filter(pl.col("sig"))),
                  ("test no-sig", te.filter(~pl.col("sig")))):
        rows.append(summ(d, nm))
        kept = d.filter(pl.col("p") >= args.t).height
        rows[-1]["kept_frac"] = round(kept / max(1, d.height), 4)
    out = pl.DataFrame(rows)
    pl.Config.set_tbl_cols(-1); pl.Config.set_tbl_rows(-1); pl.Config.set_tbl_width_chars(400)
    print(out.transpose(include_header=True, header_name="feature", column_names="group"))
    print(f"\nshare of argmax links that carry the decoy signature: val {tr['sig'].mean():.4f} | test {te['sig'].mean():.4f}")


if __name__ == "__main__":
    main()
