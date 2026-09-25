"""Step 1: load raw TSVs, learn a native-script -> Latin token dictionary from the
training pairs, normalise every record and write parquet tables.

Outputs (under --work-dir):
  {split}/s1.parquet, {split}/q.parquet   one row per record (q = Source 2 + Source 3)
  train/gt.parquet                        q_idx -> s1_idx for every true pair
  translit.json                           learned token dictionary
"""
import argparse
import os
import re
from collections import Counter, defaultdict
from multiprocessing import Pool

import polars as pl

from . import textnorm as tn
from .common import Paths, effective_cpus, log, read_tsv, save_json, timer, val_fold

_TD = (None, None)
_SPLIT = re.compile(r"[^0-9a-zऀ-෿]+")


def _init(td):
    global _TD
    _TD = td


def _norm_chunk(rows):
    ndict, adict = _TD
    out = []
    for name, addr in rows:
        nt, dom = tn.name_tokens(name, ndict)
        core = tn.name_core(nt)
        at, nums = tn.addr_tokens(addr, adict)
        out.append((
            tn.model_text(name, addr), " ".join(nt), " ".join(core), dom,
            " ".join(at), " ".join(nums), tn.script_of(name or ""),
            not bool(tn.clean_address_raw(addr)),
        ))
    return out


NORM_COLS = [("mtext", pl.Utf8), ("ntok", pl.Utf8), ("ncore", pl.Utf8), ("ndom", pl.Boolean),
             ("atok", pl.Utf8), ("anum", pl.Utf8), ("nscript", pl.Int8), ("amiss", pl.Boolean)]


def normalise(df, td, workers, chunk=20000, block=1_000_000):
    """Normalise in blocks of `block` rows so peak Python-object memory stays bounded."""
    out = []
    with Pool(workers, initializer=_init, initargs=(td,)) as pool:
        for b in range(0, df.height, block):
            part = df.slice(b, block)
            rows = list(zip(part["business_name"].fill_null("").to_list(),
                            part["business_address"].to_list()))
            res = pool.map(_norm_chunk, [rows[i:i + chunk] for i in range(0, len(rows), chunk)])
            flat = [r for p in res for r in p]
            cols = list(zip(*flat)) if flat else [[] for _ in NORM_COLS]
            out.append(part.with_columns([pl.Series(n, c, t) for (n, t), c in zip(NORM_COLS, cols)]))
            del rows, res, flat, cols
    return pl.concat(out)


def load_split(P, split):
    frames = []
    for k in (1, 2, 3):
        df = read_tsv(P.src(split, k))
        df = df.select("entity_id", "business_name", "business_address", "country")
        df = df.with_columns(pl.lit(k, pl.Int8).alias("src"),
                             pl.col("country").fill_null("UNK"))
        frames.append(df)
        log.info(f"{split} source{k}: {df.height} rows")
    s1 = frames[0].with_row_index("idx")
    q = pl.concat(frames[1:]).with_row_index("idx")
    return s1, q


def _toks(s):
    s = tn.fold(s or "").replace("'", "")
    return [t for t in _SPLIT.split(s) if t]


def learn_translit(s1, q, gt):
    """Vote native-script token -> Latin token using aligned true pairs."""
    indic = "[ऀ-෿]"
    qx = q.select("idx", pl.col("business_name").fill_null("").alias("qn"),
                  pl.col("business_address").fill_null("").alias("qa")).filter(
        pl.col("qn").str.contains(indic) | pl.col("qa").str.contains(indic))
    px = (gt.join(qx.rename({"idx": "q_idx"}), on="q_idx")
            .join(s1.select(pl.col("idx").alias("s1_idx"), pl.col("business_name").fill_null("").alias("sn"),
                            pl.col("business_address").fill_null("").alias("sa")), on="s1_idx"))
    log.info(f"pairs with native-script text: {px.height}")
    votes = {"name": defaultdict(Counter), "addr": defaultdict(Counter)}
    for sn, sa, qn, qa in px.select("sn", "sa", "qn", "qa").iter_rows():
        for key, a_txt, b_txt in (("name", sn, qn), ("addr", sa, qa)):
            if not tn.has_indic(b_txt):
                continue
            B = _toks(b_txt)
            native = [t for t in B if not tn.is_latin_token(t)]
            bl = {t for t in B if tn.is_latin_token(t)}
            un = [a for a in _toks(a_txt) if a not in bl and not a.isdigit()]
            if not un or not native:
                continue
            v = votes[key]
            if len(un) == len(native):
                for t, a in zip(native, un):
                    v[t][a] += 1.0
            else:
                w = 1.0 / len(un)
                for t in native:
                    for a in un:
                        v[t][a] += w
    out = {}
    for key, v in votes.items():
        d = {}
        for t, c in v.items():
            best, sc = c.most_common(1)[0]
            if sc >= 2 and sc / sum(c.values()) >= 0.3:
                d[t] = best
        out[key] = d
        log.info(f"translit dict[{key}]: {len(d)} tokens (from {len(v)} seen)")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--work-dir", required=True)
    ap.add_argument("--workers", type=int, default=effective_cpus())
    args = ap.parse_args()
    P = Paths(args.data_dir, args.work_dir)

    with timer("load train"):
        s1, q = load_split(P, "train")
        gt = read_tsv(P.gt()).with_columns(
            pl.col("matched_entity_ids").fill_null("").str.split(",").alias("m"))
        gt = gt.explode("m").filter(pl.col("m") != "").select(
            pl.col("source1_entity_id").alias("s1_id"), pl.col("m").alias("q_id"))
        gt = (gt.join(s1.select(pl.col("entity_id").alias("s1_id"), pl.col("idx").alias("s1_idx")), on="s1_id")
                .join(q.select(pl.col("entity_id").alias("q_id"), pl.col("idx").alias("q_idx")), on="q_id")
                .select("q_idx", "s1_idx"))
        log.info(f"train gt pairs: {gt.height}")

    s1 = s1.with_columns(pl.Series("fold", val_fold(s1["entity_id"].to_list()), pl.Int8))
    with timer("learn transliteration dictionary"):
        # learned only from non-validation pairs so the validation score stays honest
        td = learn_translit(s1, q, gt.join(s1.filter(pl.col("fold") != 0).select(pl.col("idx").alias("s1_idx")), on="s1_idx"))
        save_json(td, P.w("translit.json"))
    tdt = (td["name"], td["addr"])

    # folds: S1 by id hash (above); queries inherit their true S1's fold, else own hash
    q = q.with_columns(pl.Series("fold", val_fold(q["entity_id"].to_list()), pl.Int8))
    qf = gt.join(s1.select(pl.col("idx").alias("s1_idx"), pl.col("fold").alias("tf")), on="s1_idx")
    q = q.join(qf.select(pl.col("q_idx").alias("idx"), "tf", pl.col("s1_idx").alias("true_s1")),
               on="idx", how="left")
    q = q.with_columns(pl.coalesce("tf", "fold").alias("fold")).drop("tf").sort("idx")
    gt.write_parquet(P.w("train", "gt.parquet"))

    for split in ("train", "test"):
        if split == "test":
            with timer("load test"):
                s1, q = load_split(P, "test")
        for nm, df in (("s1", s1), ("q", q)):
            with timer(f"normalise {split}/{nm} ({df.height} rows)"):
                df = normalise(df, tdt, args.workers)
                df.write_parquet(P.w(split, f"{nm}.parquet"))
        log.info(f"{split} countries S1: {dict(s1['country'].value_counts().iter_rows())}")


if __name__ == "__main__":
    main()
