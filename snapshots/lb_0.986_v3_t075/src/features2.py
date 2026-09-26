"""Step 6b: second-stage features on the pruned pairs, aimed at "sibling" decoys.

A sibling decoy is a different business whose name is the real one plus a word
(often a legal word such as Group / Holdings / Corp) and whose address is a few
doors up the same street ("XMM Staffing Group, 10 Stagecoach Rd" vs
"XMM Staffing Corp, 8 Stagecoach Rd"). In training they are about half of all
unmatched US records. True pairs also carry number noise (formatting, a dropped
or changed digit, an injected "H.NO 53" in front), so no single rule works; the
ranker learns from these signals:

  * house numbers  - first-number equality / containment / gap / signed gap, and
                     set-level signals: a number moved (present on one side only,
                     a nearby value on the other) with its signed distance, shared
                     count, share of each side's numbers found on the other side
  * extra words    - full-name words (legal words included) present on one side
                     only and not a fuzzy variant of a word on the other side:
                     counts and a log-odds of "appears in true pairs vs wrong pairs"
                     learned on the base-model folds (5-9) only
  * legal words    - legal words added, strong "group/holding" additions, and a
                     conflict flag when both sides carry different legal forms
  * ambiguity      - how many Source 1 entities in the country share this exact
                     full name (matters when the record has no address)
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
from .textnorm import LEGAL

LEGAL_WORDS = LEGAL - {"the", "of", "&", "m", "ms"}
LEGAL_FORMS = {"inc", "corp", "co", "ltd", "pvt", "llc", "llp", "lp", "plc", "pllc", "pc", "pa",
               "sarl", "sas", "sasu", "sa", "eurl", "sci", "snc", "scop", "selarl", "gie"}
STRONG_ADD = {"group", "holding", "holdings"}
MOVE_MAX = 50  # a "moved" number: present on one side only, within this distance of one on the other side
NUM_FEATS = ("hn_eq", "hn_contain", "hn_gap", "hn_signed", "hn_conflict", "s_nums_in_q", "q_nums_in_s",
             "num_shared", "num_move", "num_move_signed", "leg_add_n", "leg_add_strong", "leg_form_conflict",
             "leg_miss_n", "nothing_added")

_G = {}


def _pmap(fn, n, parts):
    step = max(20000, n // parts + 1)
    spans = [(i, min(n, i + step)) for i in range(0, n, step)]
    with get_context("fork").Pool(effective_cpus()) as pool:
        return pool.map(fn, spans)


def _extra(a_toks, b_toks, thr=75):
    """Tokens of b with no exact or fuzzy counterpart in a."""
    out = []
    for t in b_toks:
        if t in a_toks:
            continue
        if any(fuzz.ratio(t, u) >= thr for u in a_toks):
            continue
        out.append(t)
    return out


def _as_int(x):
    return int(x) if len(x) <= 9 else None


def _pair_chunk(span):
    lo, hi = span
    ia, ib, s_tok, q_tok, s_num, q_num = _G["pair"]
    n = hi - lo
    o = {k: np.full(n, -1.0, np.float32) for k in NUM_FEATS}
    o["hn_signed"][:] = 0.0
    o["num_move_signed"][:] = 0.0
    q_extra, s_miss = [], []
    for k in range(lo, hi):
        j = k - lo
        a_n, b_n = s_num[ia[k]], q_num[ib[k]]
        if a_n and b_n:
            fa, fb = a_n[0], b_n[0]
            o["hn_eq"][j] = float(fa == fb)
            o["hn_contain"][j] = float(fa != fb and (fa in fb or fb in fa))
            xa, xb = _as_int(fa), _as_int(fb)
            if xa is not None and xb is not None:
                o["hn_gap"][j] = math.log1p(abs(xb - xa))
                o["hn_signed"][j] = float(max(-100000, min(100000, xb - xa)))
            sa, sb = set(a_n), set(b_n)
            inter = sa & sb
            contain = any(x in y or y in x for x in sa for y in sb)
            o["hn_conflict"][j] = float(not inter and not contain)
            o["s_nums_in_q"][j] = len(inter) / len(sa)
            o["q_nums_in_s"][j] = len(inter) / len(sb)
            o["num_shared"][j] = len(inter)
            best = None
            for x in sa - sb:
                vx = _as_int(x)
                if vx is None:
                    continue
                for y in sb - sa:
                    vy = _as_int(y)
                    if vy is None or x in y or y in x:
                        continue
                    d = vy - vx
                    if best is None or abs(d) < abs(best):
                        best = d
            o["num_move"][j] = float(best is not None and 0 < abs(best) <= MOVE_MAX)
            o["num_move_signed"][j] = float(best) if best is not None and abs(best) <= MOVE_MAX else 0.0
        a_t, b_t = s_tok[ia[k]], q_tok[ib[k]]
        qe, sm = _extra(a_t, b_t), _extra(b_t, a_t)
        q_extra.append(qe)
        s_miss.append(sm)
        la, lb = set(a_t) & LEGAL_WORDS, set(b_t) & LEGAL_WORDS
        add, miss = lb - la, la - lb
        o["leg_add_n"][j] = len(add)
        o["leg_miss_n"][j] = len(miss)
        o["leg_add_strong"][j] = float(bool(add & STRONG_ADD))
        o["leg_form_conflict"][j] = float(bool(add & LEGAL_FORMS) and bool(miss & LEGAL_FORMS))
        o["nothing_added"][j] = float(not qe)
    return o, q_extra, s_miss


def pair_features(pr, s1, q):
    ia, ib = pr["s1_idx"].to_numpy(), pr["q_idx"].to_numpy()
    split = lambda L: [x.split() if x else [] for x in L]
    _G["pair"] = (ia, ib, split(s1["ntok"].to_list()), split(q["ntok"].to_list()),
                  split(s1["anum"].to_list()), split(q["anum"].to_list()))
    parts = _pmap(_pair_chunk, len(ia), effective_cpus() * 4)
    F = {k: np.concatenate([p[0][k] for p in parts]) for k in NUM_FEATS}
    q_extra = [x for p in parts for x in p[1]]
    s_miss = [x for p in parts for x in p[2]]
    return F, q_extra, s_miss


def learn_logodds(pr_train, q_extra, s_miss, q_train, alpha=2.0):
    """log P(word extra | true pair) - log P(word extra | wrong pair), base folds only."""
    lab = pr_train.join(q_train.select(pl.col("idx").alias("q_idx"), "fold", "true_s1"), on="q_idx", how="left",
                        maintain_order="left")
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
        log.info(f"{name}: {len(lo)} words | most 'different business': {[t for t, _ in top[:15]]}"
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
    same = (pl.col("q_fn") == pl.col("s_fn")).cast(pl.Int32)
    d = d.with_columns(
        pl.when(pl.col("q_fn").is_null()).then(-1).otherwise(pl.len().over("s1_idx", "q_fn") - 1).alias("cl_same_qnum"),
        pl.when(pl.col("s_fn").is_null()).then(-1).otherwise(same.sum().over("s1_idx") - same.fill_null(0)).alias("cl_same_snum"),
        pl.len().over("s1_idx").alias("cl_n"),
    ).sort("r")
    return {k: d[k].fill_null(-1).cast(pl.Float32).to_numpy() for k in ("cl_same_qnum", "cl_same_snum", "cl_n")}


def fullname_freq(pr, s1):
    """log(1 + number of S1 entities in the same country with the same full-name token set)."""
    key = s1.select("idx", "country", pl.col("ntok").str.split(" ").list.unique().list.sort().list.join(" ").alias("k"))
    cnt = key.join(key.group_by("country", "k").len(), on=["country", "k"], how="left").sort("idx")["len"].to_numpy()
    return {"s_fullname_freq": np.log1p(cnt[pr["s1_idx"].to_numpy()]).astype(np.float32)}


def build(P, split, lo_tables):
    pr = pl.read_parquet(P.w("pruned", f"{split}.parquet"), columns=["q_idx", "s1_idx"])
    s1 = pl.read_parquet(P.w(split, "s1.parquet"), columns=["idx", "ntok", "anum", "country"])
    q = pl.read_parquet(P.w(split, "q.parquet"), columns=["idx", "ntok", "anum"] + (["fold", "true_s1"] if split == "train" else []))
    with timer(f"{split}: pair features on {pr.height} pruned pairs"):
        F, q_extra, s_miss = pair_features(pr, s1, q)
    if lo_tables is None:
        lo_tables = learn_logodds(pr, q_extra, s_miss, q)
        save_json(lo_tables, P.w("models", "extra_word_logodds.json"))
    F.update(lo_feats(q_extra, lo_tables["q_extra"], "qx"))
    F.update(lo_feats(s_miss, lo_tables["s_miss"], "sm"))
    with timer(f"{split}: cluster + ambiguity features"):
        F.update(cluster_feats(pr, s1, q))
        F.update(fullname_freq(pr, s1))
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
