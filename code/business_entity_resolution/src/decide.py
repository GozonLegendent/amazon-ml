"""Step 8: turn pair probabilities into matches and write the submission files.

In the training labels every Source 2/3 record matches at most one Source 1
entity, so each query is first reduced to its single most probable Source 1
entity. Two ways to decide which of those links to keep are evaluated on the
validation fold with the exact challenge metric (macro F0.5 per Source 1 entity,
singletons included), and the better one is applied unchanged to test:
  * threshold - keep a link if p >= t
  * ef        - expected-F0.5 decoding: for each Source 1 entity, enumerate every
                true/false labelling of its candidate links weighted by their
                (temperature-calibrated) probabilities and keep the top-k that
                maximises expected F0.5 (k = 0 is the "singleton" choice)

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


def _calib(p, a):
    # float64 throughout: in float32, 1/(1+exp(-z)) rounds to exactly 1.0 for z > ~16 and log(1-p) = -inf
    p = np.clip(np.asarray(p, np.float64), 1e-6, 1 - 1e-6)
    z = a * np.log(p / (1 - p))
    return np.clip(1 / (1 + np.exp(-z)), 1e-12, 1 - 1e-12)


def ef_select(best, a=1.0, miss=0.0, floor=0.01, nmax=10):
    """Expected-F0.5 top-k per Source 1 entity. best: (q_idx, s1_idx, p) with one row per query."""
    c = (best.filter(pl.col("p") >= floor).sort("s1_idx", "p", descending=[False, True])
             .group_by("s1_idx", maintain_order=True).head(nmax)
             .with_columns(pl.col("p").rank("ordinal", descending=True).over("s1_idx").cast(pl.Int32).alias("r"),
                           pl.len().over("s1_idx").alias("n")))
    keep = []
    for n in sorted(c["n"].unique().to_list()):
        g = c.filter(pl.col("n") == n).sort("s1_idx", "r")
        m = g.height // n
        pm = _calib(g["p"].to_numpy(), a).reshape(m, n)
        Y = ((np.arange(2 ** n)[:, None] >> np.arange(n)[None, :]) & 1).astype(np.float64)   # configs x n
        W = np.exp(np.log(pm) @ Y.T + np.log1p(-pm) @ (1 - Y).T)                              # m x configs
        gsum = Y.sum(1) + miss
        E = np.empty((m, n + 1))
        E[:, 0] = W @ ((Y.sum(1) == 0) * np.exp(-miss))
        for k in range(1, n + 1):
            tp = Y[:, :k].sum(1)
            E[:, k] = W @ np.where(tp > 0, 1.25 * tp / (0.25 * gsum + k), 0.0)
        kstar = E.argmax(1)
        sel = (np.arange(1, n + 1)[None, :] <= kstar[:, None]).ravel()
        keep.append(g.filter(pl.Series(sel)))
    if not keep:
        return best.head(0).select("s1_idx", "q_idx")
    return pl.concat(keep).select("s1_idx", "q_idx")


def load_scores(P, split, source):
    """source='scored': final ranker; source='pruned': pruner probabilities (quick baseline)."""
    if source == "pruned":
        return pl.read_parquet(P.w("pruned", f"{split}.parquet"), columns=["q_idx", "s1_idx", "p_a"]).rename({"p_a": "p"})
    return pl.read_parquet(P.w("scored", f"{split}.parquet"))


def tune(P, args):
    scored = pl.read_parquet(args.scored_file) if args.scored_file else load_scores(P, "train", args.scores)
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
    log.info(f"best threshold {t_best:.2f}: {f_best:.5f}")
    ef_res = []
    for a in (0.8, 1.0, 1.25, 1.5, 2.0):
        for miss in (0.0, 0.05, 0.15):
            fe, _ = f05_macro(ef_select(best, a, miss), gt, val.select("s1_idx"))
            ef_res.append((a, miss, float(fe)))
    a_best, m_best, fe_best = max(ef_res, key=lambda r: r[2])
    log.info("expected-F sweep (a, miss, F): " + " ".join(f"{a}/{m}:{f:.4f}" for a, m, f in ef_res))
    mode = {"mode": "ef", "a": a_best, "miss": m_best} if fe_best > f_best else {"mode": "threshold", "t": t_best}
    log.info(f"decision rule: {mode} (threshold {f_best:.5f} vs expected-F {fe_best:.5f})")
    pred = (ef_select(best, a_best, m_best) if mode["mode"] == "ef"
            else best.filter(pl.col("p") >= t_best).select("s1_idx", "q_idx"))
    f, u = f05_macro(pred, gt, val.select("s1_idx"))
    u = u.join(val, on="s1_idx")
    sing = u.filter(pl.col("ngt") == 0)
    nons = u.filter(pl.col("ngt") > 0)
    log.info(f"VALIDATION macro F0.5 = {f:.5f} with {mode} | {u.height} S1 entities")
    log.info(f"  singletons: {sing.height} scored {sing['f'].mean():.4f} | "
             f"non-singletons: {nons.height} scored {nons['f'].mean():.4f}")
    log.info(f"  micro precision {u['tp'].sum() / max(1, u['npred'].sum()):.4f} | "
             f"micro recall {u['tp'].sum() / max(1, u['ngt'].sum()):.4f}")
    for c, g in u.group_by("country"):
        log.info(f"  country {c[0]}: F0.5 {g['f'].mean():.4f} over {g.height}")
    if not args.eval_only:
        save_json({"threshold": t_best, "rule": mode, "val_f05": f, "scores": args.scores, "sweep": res, "ef_sweep": ef_res},
                  P.w("decision.json" if args.scores == "scored" else f"decision_{args.scores}.json"))
    return mode


def write_outputs(P, rule, out_dir, source):
    scored = load_scores(P, "test", source)
    pruned = pl.read_parquet(P.w("pruned", "test.parquet"), columns=["q_idx", "s1_idx"])
    s1 = pl.read_parquet(P.w("test", "s1.parquet"), columns=["idx", "entity_id"]).rename(
        {"idx": "s1_idx", "entity_id": "source1_entity_id"})
    qid = pl.read_parquet(P.w("test", "q.parquet"), columns=["idx", "entity_id"]).rename(
        {"idx": "q_idx", "entity_id": "qid"})
    top = best_per_query(scored)
    if rule.get("country_t"):
        # per-country thresholds (countries not listed use rule["t"])
        ct = pl.read_parquet(P.w("test", "s1.parquet"), columns=["idx", "country"]).rename({"idx": "s1_idx"})
        tmap = pl.DataFrame({"country": list(rule["country_t"]), "tc": [float(v) for v in rule["country_t"].values()]})
        top = (top.join(ct, on="s1_idx").join(tmap, on="country", how="left")
                  .with_columns(pl.col("tc").fill_null(rule["t"])))
        best = top.filter(pl.col("p") >= pl.col("tc")).select("s1_idx", "q_idx")
    else:
        best = (ef_select(top, rule["a"], rule["miss"]) if rule["mode"] == "ef"
                else top.filter(pl.col("p") >= rule["t"]).select("s1_idx", "q_idx"))
    log.info(f"test decision rule: {rule}")
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
    ap.add_argument("--country-t", default=None, help='per-country test thresholds, e.g. "France:0.9,US:0.75" (others use --threshold)')
    ap.add_argument("--scored-file", default=None, help="evaluate this train scores parquet instead (with --eval-only)")
    ap.add_argument("--eval-only", action="store_true", help="only report validation, write nothing")
    ap.add_argument("--scores", choices=["scored", "pruned"], default="scored",
                    help="which pair probabilities to decode (pruned = quick baseline before the cross-encoder)")
    args = ap.parse_args()
    P = Paths(args.data_dir, args.work_dir)
    with timer("tune decision rule on validation fold"):
        rule = {"mode": "threshold", "t": args.threshold} if args.threshold is not None else tune(P, args)
        if args.country_t:
            rule = dict(rule, mode="threshold", t=rule.get("t", args.threshold),
                        country_t={k: float(v) for k, v in (x.split(":") for x in args.country_t.split(","))})
    if args.eval_only:
        return
    with timer("write test outputs"):
        write_outputs(P, rule, args.out_dir, args.scores)


if __name__ == "__main__":
    main()
