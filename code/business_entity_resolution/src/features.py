"""Step 4: pairwise features for every candidate pair.

Feature groups (all country-agnostic; country is never a feature so the model
transfers to labels unseen in training, e.g. France):
  * dense      - bi-encoder cosine, its ranks in both directions, margins to the
                 best / runner-up candidate of the same query and the same S1
  * name       - rapidfuzz ratios on canonical tokens and core tokens, Jaro-Winkler,
                 concatenated (space-free) similarity for domain-style names,
                 IDF-weighted token cosine and char-3gram TF-IDF cosine
  * address    - rapidfuzz ratios, IDF-weighted token cosine, char-3gram cosine,
                 number overlap (house / unit / PIN numbers)
  * record     - script, missing address, domain-name flag, token counts, source,
                 how common the S1 core name is inside its country
Output: feats/{split}.parquet aligned row-by-row with cands/{split}.parquet.
"""
import argparse

import numpy as np
import polars as pl
import scipy.sparse as sp
from rapidfuzz import fuzz, process
from rapidfuzz.distance import JaroWinkler
from sklearn.feature_extraction.text import HashingVectorizer, TfidfTransformer
from sklearn.preprocessing import normalize

from .common import Paths, log, timer

N_FEAT_HASH = 2 ** 21


def _vec(texts_a, texts_b, analyzer, ngram=(1, 1)):
    """Fit IDF on both sides together, return L2-normalised TF-IDF CSR matrices."""
    hv = HashingVectorizer(n_features=N_FEAT_HASH, analyzer=analyzer, ngram_range=ngram,
                           alternate_sign=False, norm=None, token_pattern=r"\S+",
                           lowercase=False, binary=True)
    A, B = hv.transform(texts_a), hv.transform(texts_b)
    tf = TfidfTransformer(sublinear_tf=True).fit(sp.vstack([A, B]))
    return normalize(tf.transform(A)).tocsr(), normalize(tf.transform(B)).tocsr()


def _rowdot(A, B, ia, ib, chunk=2_000_000):
    out = np.empty(len(ia), np.float32)
    for i in range(0, len(ia), chunk):
        out[i:i + chunk] = np.asarray(A[ia[i:i + chunk]].multiply(B[ib[i:i + chunk]]).sum(1)).ravel()
    return out


def _cp(a, b, scorer, workers):
    return process.cpdist(a, b, scorer=scorer, workers=workers, dtype=np.float32)


def _num_sets(anum):
    return [frozenset(x.split()) if x else frozenset() for x in anum]


def num_feats(na, nb, ia, ib):
    """Number overlap between two addresses: shared count, Jaccard, first-number match."""
    inter = np.empty(len(ia), np.float32)
    jac = np.empty(len(ia), np.float32)
    first = np.empty(len(ia), np.float32)
    for k, (x, y) in enumerate(zip(ia, ib)):
        a, b = na[x], nb[y]
        if not a or not b:
            inter[k], jac[k], first[k] = -1, -1, -1
            continue
        i = len(a & b)
        inter[k] = i
        jac[k] = i / len(a | b)
        first[k] = 0
    return inter, jac, first


def build(P, split, workers):
    cands = pl.read_parquet(P.w("cands", f"{split}.parquet"))
    cols = ["idx", "ntok", "ncore", "ndom", "atok", "anum", "nscript", "amiss", "country"]
    s1 = pl.read_parquet(P.w(split, "s1.parquet"), columns=cols)
    q = pl.read_parquet(P.w(split, "q.parquet"), columns=cols + ["src"])
    ia, ib = cands["s1_idx"].to_numpy(), cands["q_idx"].to_numpy()
    F = {}

    with timer(f"{split}: dense/context features ({cands.height} pairs)"):
        c = cands.with_columns(
            pl.col("cos").max().over("q_idx").alias("q_best"),
            pl.col("cos").rank("ordinal", descending=True).over("q_idx").alias("q_rank"),
            pl.len().over("q_idx").alias("q_n"),
            pl.col("cos").max().over("s1_idx").alias("s_best"),
            pl.col("cos").rank("ordinal", descending=True).over("s1_idx").alias("s_rank"),
            pl.len().over("s1_idx").alias("s_n"),
        )
        # runner-up of the query: second-best cosine among its candidates
        c = c.with_columns(
            pl.col("cos").sort(descending=True).slice(1, 1).first().over("q_idx").fill_null(-1).alias("q_second"))
        F["cos"] = c["cos"].to_numpy()
        F["rq"] = c["rq"].cast(pl.Float32).to_numpy()
        F["rs"] = c["rs"].cast(pl.Float32).to_numpy()
        F["q_rank"] = c["q_rank"].cast(pl.Float32).to_numpy()
        F["s_rank"] = c["s_rank"].cast(pl.Float32).to_numpy()
        F["q_n"] = c["q_n"].cast(pl.Float32).to_numpy()
        F["s_n"] = c["s_n"].cast(pl.Float32).to_numpy()
        F["gap_q_best"] = (c["cos"] - c["q_best"]).to_numpy()
        F["gap_s_best"] = (c["cos"] - c["s_best"]).to_numpy()
        # margin over the strongest competitor of the same query (positive only for its top-1)
        F["margin_q"] = np.where(F["q_rank"] == 1, c["cos"] - c["q_second"], c["cos"] - c["q_best"]).astype(np.float32)

    s_ntok, q_ntok = s1["ntok"].to_list(), q["ntok"].to_list()
    s_core, q_core = s1["ncore"].to_list(), q["ncore"].to_list()
    s_atok, q_atok = s1["atok"].to_list(), q["atok"].to_list()

    with timer(f"{split}: string similarity (rapidfuzz, {workers} workers)"):
        A_n = [s_ntok[i] for i in ia]; B_n = [q_ntok[i] for i in ib]
        F["n_ratio"] = _cp(A_n, B_n, fuzz.ratio, workers)
        F["n_tsort"] = _cp(A_n, B_n, fuzz.token_sort_ratio, workers)
        F["n_tset"] = _cp(A_n, B_n, fuzz.token_set_ratio, workers)
        F["n_partial"] = _cp(A_n, B_n, fuzz.partial_ratio, workers)
        del A_n, B_n
        A_c = [s_core[i] for i in ia]; B_c = [q_core[i] for i in ib]
        F["c_ratio"] = _cp(A_c, B_c, fuzz.ratio, workers)
        F["c_tset"] = _cp(A_c, B_c, fuzz.token_set_ratio, workers)
        F["c_jw"] = _cp(A_c, B_c, JaroWinkler.normalized_similarity, workers)
        A_cc = [x.replace(" ", "") for x in A_c]; B_cc = [x.replace(" ", "") for x in B_c]
        F["c_concat"] = _cp(A_cc, B_cc, fuzz.ratio, workers)
        F["c_concat_partial"] = _cp(A_cc, B_cc, fuzz.partial_ratio, workers)
        del A_c, B_c, A_cc, B_cc
        A_a = [s_atok[i] for i in ia]; B_a = [q_atok[i] for i in ib]
        F["a_ratio"] = _cp(A_a, B_a, fuzz.ratio, workers)
        F["a_tset"] = _cp(A_a, B_a, fuzz.token_set_ratio, workers)
        F["a_tsort"] = _cp(A_a, B_a, fuzz.token_sort_ratio, workers)
        F["a_partial"] = _cp(A_a, B_a, fuzz.partial_ratio, workers)
        del A_a, B_a

    with timer(f"{split}: TF-IDF cosines"):
        for name, sa, qa, an, ng in (("c_tfidf", s_core, q_core, "word", (1, 1)),
                                     ("c_char3", s_core, q_core, "char_wb", (3, 3)),
                                     ("a_tfidf", s_atok, q_atok, "word", (1, 1)),
                                     ("a_char3", s_atok, q_atok, "char_wb", (3, 3))):
            A, B = _vec(sa, qa, an, ng)
            F[name] = _rowdot(A, B, ia, ib)
            del A, B

    with timer(f"{split}: number overlap"):
        s_num, q_num = s1["anum"].to_list(), q["anum"].to_list()
        A, B = _vec(s_num, q_num, "word")
        F["num_tfidf"] = _rowdot(A, B, ia, ib)
        na, nb = _num_sets(s_num), _num_sets(q_num)
        F["num_inter"], F["num_jac"], _ = num_feats(na, nb, ia, ib)
        s_first = np.array([x.split()[0] if x else "" for x in s_num], dtype=object)
        q_first = np.array([x.split()[0] if x else "" for x in q_num], dtype=object)
        fa, fb = s_first[ia], q_first[ib]
        F["num_first_eq"] = np.where((fa == "") | (fb == ""), -1, (fa == fb)).astype(np.float32)
        F["s_nnum"] = np.array([len(x) for x in na], np.float32)[ia]
        F["q_nnum"] = np.array([len(x) for x in nb], np.float32)[ib]

    with timer(f"{split}: record features"):
        F["q_script"] = q["nscript"].to_numpy()[ib].astype(np.float32)
        F["q_amiss"] = q["amiss"].to_numpy()[ib].astype(np.float32)
        F["q_dom"] = q["ndom"].to_numpy()[ib].astype(np.float32)
        F["q_src"] = q["src"].to_numpy()[ib].astype(np.float32)
        ntok_len = lambda L: np.array([len(x.split()) for x in L], np.float32)
        F["s_ncore_len"] = ntok_len(s_core)[ia]
        F["q_ncore_len"] = ntok_len(q_core)[ib]
        F["s_atok_len"] = ntok_len(s_atok)[ia]
        F["q_atok_len"] = ntok_len(q_atok)[ib]
        freq = s1.group_by("country", "ncore").len()
        sf = s1.join(freq, on=["country", "ncore"], how="left").sort("idx")["len"].to_numpy()
        F["s_core_freq"] = np.log1p(sf[ia]).astype(np.float32)

    feats = pl.DataFrame({k: np.asarray(v, np.float32) for k, v in F.items()})
    feats.write_parquet(P.w("feats", f"{split}.parquet"))
    log.info(f"{split}: wrote {feats.height} x {feats.width} features")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--work-dir", required=True)
    ap.add_argument("--workers", type=int, default=-1)
    ap.add_argument("--splits", default="train,test")
    args = ap.parse_args()
    P = Paths(args.data_dir, args.work_dir)
    for split in args.splits.split(","):
        build(P, split, args.workers)


if __name__ == "__main__":
    main()
