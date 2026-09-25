"""Steps 5 and 7: LightGBM pair classifiers.

--stage prune : trained on blocking candidates with the cheap features (dense +
                string + number + record). For every query it keeps the top
                --keep candidates with probability >= --pmin. The surviving
                pairs are the final candidate set (pruned/{split}.parquet); they
                are what the cross-encoder and the final ranker score, and what
                is written to candidate_pairs.tsv.
--stage final : trained on the pruned pairs with all features plus the
                cross-encoder logit and "competition" features (rank / margin of
                the pair among the query's and the S1's other candidates).

Fold protocol (Source 1 entity folds, queries inherit their true S1's fold):
  folds 5-9 -> base models (bi-encoder, cross-encoder)   [never used here]
  folds 1-4 -> ranker training                             (out-of-sample base scores)
  fold  0   -> validation (early stopping + threshold tuning in decide.py)
"""
import argparse
import os

import lightgbm as lgb
import numpy as np
import polars as pl

from .common import Paths, effective_cpus, log, save_json, timer

RANK_FOLDS = [1, 2, 3, 4]


def labels_and_folds(P, cands):
    q = pl.read_parquet(P.w("train", "q.parquet"), columns=["idx", "fold", "true_s1"])
    c = cands.select("q_idx", "s1_idx").join(q.rename({"idx": "q_idx"}), on="q_idx", how="left",
                                             maintain_order="left")  # labels must stay row-aligned
    y = (c["true_s1"] == c["s1_idx"]).fill_null(False).to_numpy().astype(np.float32)
    return y, c["fold"].to_numpy()


def competition_feats(df, col, prefix):
    """Rank / margin of `col` among the pair's query group and S1 group."""
    df = df.with_columns(
        pl.col(col).rank("ordinal", descending=True).over("q_idx").cast(pl.Float32).alias(f"{prefix}_qrank"),
        pl.col(col).max().over("q_idx").alias("_qmax"),
        pl.col(col).sort(descending=True).slice(1, 1).first().over("q_idx").alias("_q2"),
        pl.col(col).rank("ordinal", descending=True).over("s1_idx").cast(pl.Float32).alias(f"{prefix}_srank"),
        pl.col(col).max().over("s1_idx").alias("_smax"),
    )
    df = df.with_columns(
        pl.when(pl.col(f"{prefix}_qrank") == 1).then(pl.col(col) - pl.col("_q2").fill_null(-20))
          .otherwise(pl.col(col) - pl.col("_qmax")).cast(pl.Float32).alias(f"{prefix}_qmargin"),
        (pl.col(col) - pl.col("_smax")).cast(pl.Float32).alias(f"{prefix}_sgap"),
    ).drop("_qmax", "_q2", "_smax")
    return df


def load_matrix(P, split, stage):
    if stage == "prune":
        cands = pl.read_parquet(P.w("cands", f"{split}.parquet"), columns=["q_idx", "s1_idx"])
        X = pl.read_parquet(P.w("feats", f"{split}.parquet"))
        return cands, X
    pr = pl.read_parquet(P.w("pruned", f"{split}.parquet"))
    feats = pl.read_parquet(P.w("feats", f"{split}.parquet"))
    X = feats[pr["row"].to_numpy()]
    xe = np.load(P.w("xenc", f"{split}.npy"))
    if len(xe) != pr.height:
        raise RuntimeError(f"xenc/{split}.npy has {len(xe)} scores but pruned has {pr.height} pairs: re-run xenc")
    df = pr.select("q_idx", "s1_idx", "p_a").with_columns(pl.Series("xe", xe, pl.Float32))
    df = competition_feats(df, "xe", "xe")
    df = competition_feats(df, "p_a", "pa")
    df = df.with_columns(
        (pl.col("xe") > 0).sum().over("s1_idx").cast(pl.Float32).alias("xe_s_npos"),
        pl.len().over("q_idx").cast(pl.Float32).alias("pr_q_n"),
        pl.len().over("s1_idx").cast(pl.Float32).alias("pr_s_n"),
    )
    parts = [X]
    f2p = P.w("feats2", f"{split}.parquet")
    if os.path.exists(f2p):  # second-stage sibling / number / cluster features
        f2 = pl.read_parquet(f2p)
        if f2.height != pr.height:
            raise RuntimeError(f"feats2/{split}.parquet has {f2.height} rows but pruned has {pr.height}: re-run feats2")
        parts.append(f2)
    parts.append(df.drop("q_idx", "s1_idx"))
    X = pl.concat(parts, how="horizontal")
    return pr.select("q_idx", "s1_idx", "row"), X


PARAMS = dict(objective="binary", learning_rate=0.1, force_col_wise=True, num_leaves=255, min_data_in_leaf=200,
              feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0,
              max_bin=255, verbose=-1)


def fit(P, stage, args):
    cands, X = load_matrix(P, "train", stage)
    y, fold = labels_and_folds(P, cands)
    tr = np.isin(fold, RANK_FOLDS)
    va = fold == 0
    if args.max_train_rows and tr.sum() > args.max_train_rows:
        rng = np.random.default_rng(0)
        keep = np.zeros_like(tr)
        idx = np.flatnonzero(tr)
        keep[rng.choice(idx, args.max_train_rows, replace=False)] = True
        tr = keep
    names = X.columns
    Xn = X.to_numpy().astype(np.float32)
    log.info(f"[{stage}] train rows {tr.sum()} (pos {y[tr].mean():.4f}) | val rows {va.sum()} | {len(names)} features")
    params = dict(PARAMS, num_threads=args.threads, seed=0)
    dtr = lgb.Dataset(Xn[tr], y[tr], feature_name=names, free_raw_data=True)
    dva = lgb.Dataset(Xn[va], y[va], reference=dtr)
    with timer(f"[{stage}] lightgbm fit"):
        bst = lgb.train(params, dtr, num_boost_round=args.rounds, valid_sets=[dva], valid_names=["val"],
                        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(50)])
    bst.save_model(P.w("models", f"lgb_{stage}.txt"))
    imp = sorted(zip(names, bst.feature_importance("gain")), key=lambda t: -t[1])
    log.info(f"[{stage}] top features: " + ", ".join(f"{n}:{g:.0f}" for n, g in imp[:20]))
    save_json({n: float(g) for n, g in imp}, P.w("models", f"lgb_{stage}_importance.json"))
    return bst


def predict(P, stage, bst, split, args):
    cands, X = load_matrix(P, split, stage)
    with timer(f"[{stage}] predict {split} ({X.height} rows)"):
        p = bst.predict(X.to_numpy().astype(np.float32), num_threads=args.threads)
    if stage == "prune":
        df = cands.with_columns(pl.Series("p_a", p, pl.Float32), pl.arange(0, cands.height, dtype=pl.Int64).alias("row"))
        df = (df.sort("q_idx", "p_a", descending=[False, True])
                .group_by("q_idx", maintain_order=True).head(args.keep)
                .filter(pl.col("p_a") >= args.pmin))
        df.write_parquet(P.w("pruned", f"{split}.parquet"))
        log.info(f"[prune] {split}: kept {df.height} of {cands.height} pairs "
                 f"({df.height / max(1, df['q_idx'].n_unique()):.2f} per surviving query)")
        if split == "train":
            report_recall(P, df)
    else:
        out = cands.with_columns(pl.Series("p", p, pl.Float32))
        out.write_parquet(P.w("scored", f"{split}.parquet"))


def report_recall(P, pruned):
    gt = pl.read_parquet(P.w("train", "gt.parquet"))
    s1 = pl.read_parquet(P.w("train", "s1.parquet"), columns=["idx", "fold"])
    g = gt.join(s1.rename({"idx": "s1_idx"}), on="s1_idx").filter(pl.col("fold") == 0)
    hit = g.join(pruned.select("q_idx", "s1_idx", pl.lit(1).alias("h")), on=["q_idx", "s1_idx"], how="left")
    log.info(f"[prune] recall after pruning (val fold): {hit['h'].is_not_null().mean():.5f} over {hit.height} true pairs")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--work-dir", required=True)
    ap.add_argument("--stage", choices=["prune", "final"], required=True)
    ap.add_argument("--keep", type=int, default=4, help="[prune] max candidates kept per query")
    ap.add_argument("--pmin", type=float, default=0.003, help="[prune] min probability kept")
    ap.add_argument("--rounds", type=int, default=1500)
    ap.add_argument("--max-train-rows", type=int, default=12_000_000)
    ap.add_argument("--threads", type=int, default=effective_cpus())
    ap.add_argument("--splits", default="train,test")
    args = ap.parse_args()
    P = Paths(args.data_dir, args.work_dir)
    bst = fit(P, args.stage, args)
    for split in args.splits.split(","):
        predict(P, args.stage, bst, split, args)


if __name__ == "__main__":
    main()
