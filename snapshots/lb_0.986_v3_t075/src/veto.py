"""Decoy veto: drop kept links that carry the sibling-decoy signature.

Signature (measured on the training labels): some house number on the record is
the entity's number moved UP by a small offset (q - s in OFFSETS; digit drops and
prefixes excluded via containment), AND the record's name adds a legal word or a
non-noise extra word. About 89% of US decoys carry it versus 0.15% of true pairs.

Usage
  test : python -m src.veto --data-dir D --work-dir W --pred in.tsv --out out.tsv
  val  : python -m src.veto --data-dir D --work-dir W --val     (F0.5 before/after on fold 0)
The candidate file is unchanged; vetoed links stay candidates, so matches remain
a subset of candidates.
"""
import argparse
import json
from multiprocessing import get_context

import numpy as np
import polars as pl

from .common import Paths, effective_cpus, log
from .decide import best_per_query, f05_macro
from .features2 import LEGAL_WORDS, _extra

OFFSETS = {1, 2, 3, 4, 5, 7, 9, 11, 13, 21}
NOISE = {"svc", "ctr", "dba", "formerly", "fka", "aka", "as", "nee", "ta", "doing", "business", "known",
         "com", "www", "trading"}
_G = {}


def signature(s_tok, q_tok, s_num, q_num, offsets=OFFSETS):
    sa, sb = set(s_num), set(q_num)
    moved = False
    for x in sa - sb:
        if len(x) > 9:
            continue
        for y in sb - sa:
            if len(y) > 9 or x in y or y in x:
                continue
            if int(y) - int(x) in offsets:
                moved = True
                break
        if moved:
            break
    if not moved:
        return False
    if (set(q_tok) & LEGAL_WORDS) - set(s_tok):
        return True
    return any(t not in NOISE for t in _extra(s_tok, q_tok))


def _chunk(span):
    ia, ib, st, qt, sn, qn = _G["v"]
    return np.array([signature(st[ia[k]], qt[ib[k]], sn[ia[k]], qn[ib[k]]) for k in range(*span)], bool)


def flag(pairs, s1, q):
    """pairs: (s1_idx, q_idx). s1 / q: tables with idx, ntok, anum (idx == row position)."""
    sp = lambda L: [x.split() if x else [] for x in L]
    ia, ib = pairs["s1_idx"].to_numpy(), pairs["q_idx"].to_numpy()
    _G["v"] = (ia, ib, sp(s1["ntok"].to_list()), sp(q["ntok"].to_list()), sp(s1["anum"].to_list()), sp(q["anum"].to_list()))
    n = len(ia)
    step = max(20000, n // (effective_cpus() * 4) + 1)
    spans = [(i, min(n, i + step)) for i in range(0, n, step)]
    with get_context("fork").Pool(effective_cpus()) as pool:
        parts = pool.map(_chunk, spans)
    return np.concatenate(parts) if parts else np.zeros(0, bool)


def run_val(P):
    rule = json.load(open(P.w("decision.json")))
    t = rule["threshold"]
    scored = pl.read_parquet(P.w("scored", "train.parquet"), columns=["q_idx", "s1_idx", "p"])
    s1 = pl.read_parquet(P.w("train", "s1.parquet"), columns=["idx", "fold", "ntok", "anum", "country"])
    q = pl.read_parquet(P.w("train", "q.parquet"), columns=["idx", "fold", "ntok", "anum"])
    val = s1.filter(pl.col("fold") == 0).select(pl.col("idx").alias("s1_idx"))
    gt = pl.read_parquet(P.w("train", "gt.parquet")).join(val, on="s1_idx")
    qv = q.filter(pl.col("fold") == 0).select(pl.col("idx").alias("q_idx"))
    best = best_per_query(scored).join(val, on="s1_idx").join(qv, on="q_idx")
    pred = best.filter(pl.col("p") >= t).select("s1_idx", "q_idx")
    sig = flag(pred, s1, q)
    kept = pred.filter(pl.Series(~sig))
    f0, u0 = f05_macro(pred, gt, val)
    f1, u1 = f05_macro(kept, gt, val)
    lab = pred.with_columns(pl.Series("sig", sig)).join(gt.with_columns(pl.lit(True).alias("y")), on=["s1_idx", "q_idx"], how="left")
    ns, nt = int(sig.sum()), int(lab.filter(pl.col("sig") & pl.col("y").is_not_null()).height)
    log.info(f"VALIDATION t={t}: F0.5 {f0:.5f} -> {f1:.5f} after veto | vetoed {ns} links, of which true {nt} "
             f"({nt / max(1, ns):.1%}) | micro precision {u0['tp'].sum() / u0['npred'].sum():.4f} -> "
             f"{u1['tp'].sum() / u1['npred'].sum():.4f} | recall {u0['tp'].sum() / u0['ngt'].sum():.4f} -> {u1['tp'].sum() / u1['ngt'].sum():.4f}")


def run_test(P, pred_path, out_path):
    s1 = pl.read_parquet(P.w("test", "s1.parquet"), columns=["idx", "entity_id", "ntok", "anum", "country"])
    q = pl.read_parquet(P.w("test", "q.parquet"), columns=["idx", "entity_id", "ntok", "anum"])
    pr = (pl.read_csv(pred_path, separator="\t", quote_char=None, infer_schema=False)
            .with_columns(pl.col("matched_entity_ids").fill_null("").str.split(","))
            .explode("matched_entity_ids").filter(pl.col("matched_entity_ids") != "")
            .join(s1.select(pl.col("entity_id").alias("source1_entity_id"), pl.col("idx").alias("s1_idx"), "country"), on="source1_entity_id")
            .join(q.select(pl.col("entity_id").alias("matched_entity_ids"), pl.col("idx").alias("q_idx")), on="matched_entity_ids"))
    sig = flag(pr, s1, q)
    pr = pr.with_columns(pl.Series("sig", sig))
    log.info("vetoed per country: " + str(dict(pr.group_by("country").agg(pl.col("sig").sum()).sort("country").iter_rows())))
    kept = pr.filter(~pl.col("sig"))
    agg = (kept.sort("s1_idx", "matched_entity_ids").group_by("source1_entity_id", maintain_order=True)
               .agg(pl.col("matched_entity_ids").str.join(",")))
    out = (s1.select(pl.col("entity_id").alias("source1_entity_id"), "idx").join(agg, on="source1_entity_id", how="left")
             .sort("idx").select("source1_entity_id", pl.col("matched_entity_ids").fill_null("")))
    out.write_csv(out_path, separator="\t", quote_style="never")
    log.info(f"wrote {out_path}: {out.height} rows, links {pr.height} -> {kept.height} (vetoed {int(sig.sum())})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--work-dir", required=True)
    ap.add_argument("--val", action="store_true")
    ap.add_argument("--pred")
    ap.add_argument("--out")
    args = ap.parse_args()
    P = Paths(args.data_dir, args.work_dir)
    if args.val:
        run_val(P)
    if args.pred:
        run_test(P, args.pred, args.out)


if __name__ == "__main__":
    main()
