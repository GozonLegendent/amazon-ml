"""Step 9 (optional): removal-only post-processing of a matching file.

Every rule only removes links, so matches stay a subset of candidates.
  fr_swap  - France: the record keeps the entity's first house number but exactly one
             core-name word is replaced by another word, and both words belong to the
             "qualifier swap" vocabulary. That vocabulary is learned from the file's own
             French links: words that appear both as added and as removed (>= 200 times as
             removed, added/removed ratio within 0.2-5). True training pairs never swap one
             meaningful qualifier for another (0 of ~150k US/India word-swaps); true noise
             only adds generic words. French decoys do exactly this
             ("Nogent Primaire SAS" vs "Nogent Patrimoine SAS" at the same address).
  namesake - the record has no address and its canonical name equals the names of >= 2
             Source 1 entities in the country. The link is then a coin flip (training
             precision 0.19-0.50), below the F0.5 break-even.
  cap      - more S2 links than 5 or S3 links than 6 on one entity exceeds the training
             maximum; the lowest-probability extra links are dropped.
Usage: python -m src.postfix --data-dir D --work-dir W --pred in.tsv --out out.tsv [--rules fr_swap,namesake,cap]
"""
import argparse
from collections import Counter

import polars as pl

from .common import Paths, log
from .features2 import _extra

# words that true French pairs also add or drop as noise (family suffixes, generic words): never swap evidence
AMBIGUOUS = {"ctr", "france", "ste", "gestion", "conseil", "fils", "freres", "associes", "associe"}
CAP = {2: 5, 3: 6}


def fr_swap_flags(pairs, s_core, q_core, s_num, q_num, min_removed=200):
    """pairs: rows with s_i, q_i (list positions). Returns (flags, vocabulary)."""
    ex, mi, feq = [], [], []
    for si, qi in zip(pairs["s_i"].to_list(), pairs["q_i"].to_list()):
        a, b = s_core[si].split(), q_core[qi].split()
        ex.append(_extra(a, b))
        mi.append(_extra(b, a))
        na, nb = s_num[si].split(), q_num[qi].split()
        feq.append(bool(na and nb and na[0] == nb[0]))
    swap = [f and len(e) == 1 and len(m) == 1 for f, e, m in zip(feq, ex, mi)]
    ne = Counter(e[0] for s, e in zip(swap, ex) if s)
    nm = Counter(m[0] for s, m in zip(swap, mi) if s)
    vocab = {w for w in nm if nm[w] >= min_removed and 0.2 * nm[w] <= ne[w] <= 5 * nm[w]} - AMBIGUOUS
    flags = [s and e[0] in vocab and m[0] in vocab for s, e, m in zip(swap, ex, mi)]
    return flags, vocab


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--work-dir", required=True)
    ap.add_argument("--pred", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--rules", default="fr_swap,namesake,cap")
    args = ap.parse_args()
    rules = set(args.rules.split(","))
    P = Paths(args.data_dir, args.work_dir)
    s1 = pl.read_parquet(P.w("test", "s1.parquet"), columns=["idx", "entity_id", "country", "ntok", "ncore", "anum"])
    q = pl.read_parquet(P.w("test", "q.parquet"), columns=["idx", "entity_id", "ntok", "ncore", "anum", "amiss", "src"])
    pr = (pl.read_csv(args.pred, separator="\t", quote_char=None, infer_schema=False)
            .with_columns(pl.col("matched_entity_ids").fill_null("").str.split(","))
            .explode("matched_entity_ids").filter(pl.col("matched_entity_ids") != "")
            .rename({"source1_entity_id": "sid", "matched_entity_ids": "qid"})
            .join(s1.select(pl.col("entity_id").alias("sid"), pl.col("idx").alias("s_i"), "country"), on="sid")
            .join(q.select(pl.col("entity_id").alias("qid"), pl.col("idx").alias("q_i"), "amiss", "src", pl.col("ntok").alias("q_ntok")), on="qid"))
    n0 = pr.height
    drop = pl.Series("drop", [False] * pr.height)

    if "fr_swap" in rules:
        fr = pr.with_row_index("r").filter(pl.col("country") == "France")
        flags, vocab = fr_swap_flags(fr, s1["ncore"].to_list(), q["ncore"].to_list(), s1["anum"].to_list(), q["anum"].to_list())
        log.info(f"fr_swap: learned qualifier vocabulary ({len(vocab)}): {sorted(vocab)}")
        idx = fr.filter(pl.Series(flags))["r"].to_list()
        d = drop.to_list()
        for i in idx:
            d[i] = True
        drop = pl.Series("drop", d)
        log.info(f"fr_swap: removes {len(idx)} links")

    if "namesake" in rules:
        m_ex = s1.group_by("country", "ntok").len().rename({"ntok": "q_ntok", "len": "m_ex"})
        t = pr.join(m_ex, on=["country", "q_ntok"], how="left", maintain_order="left").with_columns(pl.col("m_ex").fill_null(0))
        hit = (t["amiss"] & (t["m_ex"] >= 2)).fill_null(False)
        log.info(f"namesake: removes {int(hit.sum())} links " + str(dict(t.filter(hit).group_by("country").len().iter_rows())))
        drop = drop | hit

    pr = pr.with_columns(drop)
    if "cap" in rules:
        sc = pl.read_parquet(P.w("scored", "test.parquet"), columns=["q_idx", "s1_idx", "p"]).rename({"q_idx": "q_i", "s1_idx": "s_i"})
        t = pr.with_row_index("r").join(sc, on=["s_i", "q_i"], how="left", maintain_order="left").filter(~pl.col("drop"))
        t = t.with_columns(pl.col("p").rank("ordinal", descending=True).over("s_i", "src").alias("rk"),
                           pl.col("src").replace_strict(CAP, default=99).alias("cap"))
        over = t.filter(pl.col("rk") > pl.col("cap"))["r"].to_list()
        d = pr["drop"].to_list()
        for i in over:
            d[i] = True
        pr = pr.with_columns(pl.Series("drop", d))
        log.info(f"cap: removes {len(over)} links")

    kept = pr.filter(~pl.col("drop"))
    log.info(f"links {n0} -> {kept.height} (removed {n0 - kept.height}) | removed per country: "
             f"{dict(pr.filter(pl.col('drop')).group_by('country').len().sort('country').iter_rows())}")
    agg = kept.sort("s_i", "qid").group_by("sid", maintain_order=True).agg(pl.col("qid").str.join(",").alias("matched_entity_ids"))
    out = (s1.select(pl.col("entity_id").alias("source1_entity_id"), "idx")
             .join(agg.rename({"sid": "source1_entity_id"}), on="source1_entity_id", how="left")
             .sort("idx").select("source1_entity_id", pl.col("matched_entity_ids").fill_null("")))
    out.write_csv(args.out, separator="\t", quote_style="never")
    log.info(f"wrote {args.out}: {out.height} rows")


if __name__ == "__main__":
    main()
