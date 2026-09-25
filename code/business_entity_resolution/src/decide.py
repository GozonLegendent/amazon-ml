"""Step 8: turn pair probabilities into matches and write the submission files.

In the training labels every Source 2/3 record matches at most one Source 1
entity, so each query is assigned to its single most probable Source 1 entity,
and only if that probability clears a threshold t. t is chosen by maximising the
exact challenge metric (macro F0.5 over the validation-fold Source 1 entities,
singletons included) on the training split, then applied unchanged to test.

Writes output/matching_results.tsv and output/candidate_pairs.tsv (the pruned
candidate set that the final model scored).
"""
import argparse
import os

import numpy as np
import polars as pl

from .common import Paths, log, save_json, timer


def best_per_query(scored):
    return (scored.sort("q_idx", "p", descending=[False, True])
                  .group_by("q_idx", maintain_order=True).first())


def f05_macro(pred, gt, universe):
    """pred/gt: (s1_idx, q_idx) pairs; universe: s1_idx of entities to average over."""
    tp = pred.join(gt, on=["s1_idx", "q_idx"]).group_by("s1_idx").len().rename({"len": "tp"})
    npred = pred.group_by("s1_idx").len().rename({"len": "npred"})
    ngt = gt.group_by("s1_idx").len().rename({"len": "ngt"})
    u = (universe.join(tp, on="s1_idx", how="left").join(npred, on="s1_idx", how="left")
                 .join(ngt, on="s1_idx", how="left").fill_null(0))
    P = pl.col("tp") / pl.col("npred")
    R = pl.col("tp") / pl.col("ngt")
    f = (pl.when((pl.col("ngt") == 0) & (pl.col("npred") == 0)).then(1.0)
           .when(pl.col("tp") == 0).then(0.0)
           .otherwise(1.25 * P * R / (0.25 * P + R)))
    u = u.with_columns(f.alias("f"))
    return u["f"].mean(), u


def load_scores(P, split, source):
    """source='scored': final ranker; source='pruned': pruner probabilities (quick baseline)."""
    if source == "pruned":
        return pl.read_parquet(P.w("pruned", f"{split}.parquet"), columns=["q_idx", "s1_idx", "p_a"]).rename({"p_a": "p"})
    return pl.read_parquet(P.w("scored", f"{split}.parquet"))


def tune(P, args):
    scored = load_scores(P, "train", args.scores)
    s1 = pl.read_parquet(P.w("train", "s1.parquet"), columns=["idx", "fold", "country"])
    val = s1.filter(pl.col("fold") == 0).select(pl.col("idx").alias("s1_idx"), "country")
    gt = pl.read_parquet(P.w("train", "gt.parquet")).join(val.select("s1_idx"), on="s1_idx")
    # only validation-fold queries: their scores are out-of-sample for every model
    qv = pl.read_parquet(P.w("train", "q.parquet"), columns=["idx", "fold"]).filter(pl.col("fold") == 0)
    best = (best_per_query(scored).join(val.select("s1_idx"), on="s1_idx")
                                  .join(qv.select(pl.col("idx").alias("q_idx")), on="q_idx"))
    res = []
    for t in np.round(np.arange(0.05, 0.96, 0.01), 3):
        f, _ = f05_macro(best.filter(pl.col("p") >= t).select("s1_idx", "q_idx"), gt, val.select("s1_idx"))
        res.append((float(t), float(f)))
    t_best, f_best = max(res, key=lambda r: r[1])
    log.info("threshold sweep: " + " ".join(f"{t:.2f}:{f:.4f}" for t, f in res[::5]))
    f, u = f05_macro(best.filter(pl.col("p") >= t_best).select("s1_idx", "q_idx"), gt, val.select("s1_idx"))
    u = u.join(val, on="s1_idx")
    sing = u.filter(pl.col("ngt") == 0)
    nons = u.filter(pl.col("ngt") > 0)
    log.info(f"VALIDATION macro F0.5 = {f:.5f} at t={t_best:.2f} | {u.height} S1 entities")
    log.info(f"  singletons: {sing.height} scored {sing['f'].mean():.4f} | "
             f"non-singletons: {nons.height} scored {nons['f'].mean():.4f}")
    log.info(f"  micro precision {u['tp'].sum() / max(1, u['npred'].sum()):.4f} | "
             f"micro recall {u['tp'].sum() / max(1, u['ngt'].sum()):.4f}")
    for c, g in u.group_by("country"):
        log.info(f"  country {c[0]}: F0.5 {g['f'].mean():.4f} over {g.height}")
    save_json({"threshold": t_best, "val_f05": f, "scores": args.scores, "sweep": res},
              P.w("decision.json" if args.scores == "scored" else f"decision_{args.scores}.json"))
    return t_best


def write_outputs(P, t, out_dir, source):
    scored = load_scores(P, "test", source)
    pruned = pl.read_parquet(P.w("pruned", "test.parquet"), columns=["q_idx", "s1_idx"])
    s1 = pl.read_parquet(P.w("test", "s1.parquet"), columns=["idx", "entity_id"]).rename(
        {"idx": "s1_idx", "entity_id": "source1_entity_id"})
    qid = pl.read_parquet(P.w("test", "q.parquet"), columns=["idx", "entity_id"]).rename(
        {"idx": "q_idx", "entity_id": "qid"})
    best = best_per_query(scored).filter(pl.col("p") >= t).select("s1_idx", "q_idx")
    os.makedirs(out_dir, exist_ok=True)
    for pairs, col, fname in ((best, "matched_entity_ids", "matching_results.tsv"),
                              (pruned, "candidate_entity_ids", "candidate_pairs.tsv")):
        agg = (pairs.unique().join(qid, on="q_idx").sort("s1_idx", "qid")
                    .group_by("s1_idx", maintain_order=True).agg(pl.col("qid").str.join(",").alias(col)))
        out = (s1.join(agg, on="s1_idx", how="left").with_columns(pl.col(col).fill_null(""))
                 .sort("s1_idx").select("source1_entity_id", col))
        path = os.path.join(out_dir, fname)
        out.write_csv(path, separator="\t", quote_style="never")
        n_nonempty = (out[col] != "").sum()
        n_ids = pairs.height
        log.info(f"wrote {path}: {out.height} rows, {n_nonempty} non-empty, {n_ids} ids")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--work-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--threshold", type=float, default=None, help="skip tuning and use this")
    ap.add_argument("--scores", choices=["scored", "pruned"], default="scored",
                    help="which pair probabilities to decode (pruned = quick baseline before the cross-encoder)")
    args = ap.parse_args()
    P = Paths(args.data_dir, args.work_dir)
    with timer("tune threshold on validation fold"):
        t = args.threshold if args.threshold is not None else tune(P, args)
    with timer("write test outputs"):
        write_outputs(P, t, args.out_dir, args.scores)


if __name__ == "__main__":
    main()
