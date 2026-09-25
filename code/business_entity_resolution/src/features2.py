"""Step 6b: second-stage features on the pruned pairs, aimed at "sibling" decoys.

A sibling decoy is a different business whose name is the real one plus an extra
word and whose address is a few doors down the same street ("XMM Staffing Group,
10 Stagecoach Rd" vs "XMM Staffing Corp, 8 Stagecoach Rd"). They make up a large
share of the unmatched Source 2/3 records.

  * house numbers  - equality, digit-drop containment, numeric gap, conflict flag,
                     share of each side's numbers found on the other side
  * extra words    - name words present on one side only (not a fuzzy variant of a
                     word on the other side): counts, IDF mass, and a log-odds of
                     "this extra word appears in true pairs vs in wrong pairs",
                     learned on the base-model folds (5-9) only
  * cluster        - among the other pruned records of the same Source 1 entity, how
                     many share this record's house number vs the entity's own
Output: feats2/{split}.parquet aligned row-by-row with pruned/{split}.parquet.
"""
import argparse
import math
from collections import Counter
from multiprocessing import get_context

import numpy as np
import polars as pl
from rapidfuzz import fuzz

from .biencoder import BASE_FOLDS
from .common import Paths, effective_cpus, log, save_json, timer

_G = {}


def _pmap(fn, n, parts):
    step = max(20000, n // parts + 1)
    spans = [(i, min(n, i + step)) for i in range(0, n, step)]
    with get_context("fork").Pool(effective_cpus()) as pool:
        return pool.map(fn, spans)


def _extra(a_toks, b_toks, thr=75):
    """Tokens of b with no fuzzy counterpart in a."""
    out = []
    for t in b_toks:
        if t in a_toks:
            continue
        if any(fuzz.ratio(t, u) >= thr for u in a_toks):
            continue
        out.append(t)
    return out


def _pair_chunk(span):
    lo, hi = span
    ia, ib, s_core, q_core, s_num, q_num = _G["pair"]
    n = hi - lo
    o = {k: np.full(n, -1.0, np.float32) for k in
         ("hn_eq", "hn_contain", "hn_gap", "hn_conflict", "s_nums_in_q", "q_nums_in_s")}
    q_extra, s_miss = [], []
    for k in range(lo, hi):
        a_n, b_n = s_num[ia[k]], q_num[ib[k]]
        j = k - lo
        if a_n and b_n:
            fa, fb = a_n[0], b_n[0]
            o["hn_eq"][j] = float(fa == fb)
            o["hn_contain"][j] = float(fa != fb and (fa in fb or fb in fa))
            if len(fa) <= 9 and len(fb) <= 9:
                o["hn_gap"][j] = math.log1p(abs(int(fa) - int(fb)))
            sa, sb = set(a_n), set(b_n)
            inter = sa & sb
            contain = any(x in y or y in x for x in sa for y in sb)
            o["hn_conflict"][j] = float(not inter and not contain)
            o["s_nums_in_q"][j] = len(inter) / len(sa)
            o["q_nums_in_s"][j] = len(inter) / len(sb)
        a_t, b_t = s_core[ia[k]], q_core[ib[k]]
        q_extra.append(_extra(a_t, b_t))
        s_miss.append(_extra(b_t, a_t))
    return o, q_extra, s_miss


def pair_features(pr, s1, q):
    ia, ib = pr["s1_idx"].to_numpy(), pr["q_idx"].to_numpy()
    split_nums = lambda L: [x.split() if x else [] for x in L]
    split_toks = lambda L: [x.split() if x else [] for x in L]
    _G["pair"] = (ia, ib, split_toks(s1["ncore"].to_list()), split_toks(q["ncore"].to_list()),
                  split_nums(s1["anum"].to_list()), split_nums(q["anum"].to_list()))
    parts = _pmap(_pair_chunk, len(ia), effective_cpus() * 4)
    F = {k: np.concatenate([p[0][k] for p in parts]) for k in parts[0][0]}
    q_extra = [x for p in parts for x in p[1]]
    s_miss = [x for p in parts for x in p[2]]
    return F, q_extra, s_miss


def learn_logodds(pr_train, q_extra, s_miss, q_train, alpha=2.0):
    """log P(token extra | true pair) - log P(token extra | wrong pair), base folds only."""
    lab = (pr_train.join(q_train.select(pl.col("idx").alias("q_idx"), "fold", "true_s1"), on="q_idx", how="left",
                         maintain_order="left"))
    base = lab["fold"].is_in(BASE_FOLDS).to_numpy()
    y = (lab["true_s1"] == lab["s1_idx"]).fill_null(False).to_numpy()
    out = {}
    for name, lists in (("q_extra", q_extra), ("s_miss", s_miss)):
        pos, neg = Counter(), Counter()
        for k in np.flatnonzero(base):
            (pos if y[k] else neg).update(set(lists[k]))
        npos, nneg = max(1, int((y & base).sum())), max(1, int((~y & base).sum()))
        lo = {}
        for t in set(pos) | set(neg):
            if pos[t] + neg[t] < 20:
                continue
            lo[t] = math.log((pos[t] + alpha) / (npos + alpha)) - math.log((neg[t] + alpha) / (nneg + alpha))
        out[name] = lo
        top = sorted(lo.items(), key=lambda kv: kv[1])
        log.info(f"{name}: {len(lo)} tokens | most 'different business': {[t for t, _ in top[:12]]}"
                 f" | most 'noise': {[t for t, _ in top[-12:]]}")
    return out


def lo_feats(lists, lo, prefix):
    mx = np.zeros(len(lists), np.float32)
    mn = np.zeros(len(lists), np.float32)
    sm = np.zeros(len(lists), np.float32)
    unk = np.zeros(len(lists), np.float32)
    for k, L in enumerate(lists):
        v = [lo[t] for t in L if t in lo]
        unk[k] = len(L) - len(v)
        if v:
            mx[k], mn[k], sm[k] = max(v), min(v), sum(v)
    return {f"{prefix}_lo_max": mx, f"{prefix}_lo_min": mn, f"{prefix}_lo_sum": sm, f"{prefix}_unk": unk,
            f"{prefix}_n": np.array([len(L) for L in lists], np.float32)}


def cluster_feats(pr, s1, q):
    first = lambda L: [x.split()[0] if x else None for x in L]
    s_first = pl.DataFrame({"s1_idx": s1["idx"], "s_fn": first(s1["anum"].to_list())})
    q_first = pl.DataFrame({"q_idx": q["idx"], "q_fn": first(q["anum"].to_list())})
    d = (pr.select("q_idx", "s1_idx").with_row_index("r")
           .join(q_first, on="q_idx", how="left").join(s_first, on="s1_idx", how="left"))
    d = d.with_columns(
        pl.when(pl.col("q_fn").is_null()).then(-1).otherwise(pl.len().over("s1_idx", "q_fn") - 1).alias("cl_same_qnum"),
        pl.when(pl.col("s_fn").is_null()).then(-1).otherwise(
            (pl.col("q_fn") == pl.col("s_fn")).cast(pl.Int32).sum().over("s1_idx")
            - (pl.col("q_fn") == pl.col("s_fn")).cast(pl.Int32).fill_null(0)).alias("cl_same_snum"),
        pl.len().over("s1_idx").alias("cl_n"),
    ).sort("r")
    return {k: d[k].fill_null(-1).cast(pl.Float32).to_numpy() for k in ("cl_same_qnum", "cl_same_snum", "cl_n")}


def build(P, split, lo_tables):
    pr = pl.read_parquet(P.w("pruned", f"{split}.parquet"), columns=["q_idx", "s1_idx"])
    cols = ["idx", "ncore", "anum"]
    s1 = pl.read_parquet(P.w(split, "s1.parquet"), columns=cols)
    q = pl.read_parquet(P.w(split, "q.parquet"), columns=cols + (["fold", "true_s1"] if split == "train" else []))
    with timer(f"{split}: pair features on {pr.height} pruned pairs"):
        F, q_extra, s_miss = pair_features(pr, s1, q)
    if lo_tables is None:
        lo_tables = learn_logodds(pr, q_extra, s_miss, q)
        save_json(lo_tables, P.w("models", "extra_word_logodds.json"))
    F.update(lo_feats(q_extra, lo_tables["q_extra"], "qx"))
    F.update(lo_feats(s_miss, lo_tables["s_miss"], "sm"))
    with timer(f"{split}: cluster features"):
        F.update(cluster_feats(pr, s1, q))
    out = pl.DataFrame({k: np.asarray(v, np.float32) for k, v in F.items()})
    out.write_parquet(P.w("feats2", f"{split}.parquet"))
    log.info(f"{split}: wrote {out.height} x {out.width} second-stage features")
    return lo_tables


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--work-dir", required=True)
    args = ap.parse_args()
    P = Paths(args.data_dir, args.work_dir)
    lo = build(P, "train", None)
    build(P, "test", lo)


if __name__ == "__main__":
    main()
