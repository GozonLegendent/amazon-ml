"""Step 3: candidate generation (blocking) with exact dense nearest-neighbour search.

Within each country label (an open set - whatever strings appear in the data),
every Source 2/3 record retrieves its top-K Source 1 records by cosine similarity
of the fine-tuned bi-encoder embeddings, and every Source 1 record retrieves its
top-R Source 2/3 records (reverse direction, so entities with many near-duplicate
records still keep their own candidates). The union is the candidate set.

Search is brute-force matrix multiplication on the GPU (fp16), so there is no
approximate-index recall loss.

Output: cands/{split}.parquet with q_idx, s1_idx, cos, rq (rank of the S1 in the
query's list, 255 if absent), rs (rank of the query in the S1's list, 255 if absent).
"""
import argparse

import numpy as np
import polars as pl
import torch

from .common import Paths, log, timer


def topk_chunks(A, B, k, chunk):
    """For each row of A (on device) return top-k cosine vs rows of B (on device)."""
    k = min(k, B.shape[0])
    vals, idx = [], []
    for i in range(0, A.shape[0], chunk):
        s = A[i:i + chunk] @ B.T
        v, j = s.topk(k, dim=1)
        vals.append(v.float().cpu())
        idx.append(j.int().cpu())
    return torch.cat(vals).numpy(), torch.cat(idx).numpy()


def retrieve_split(P, split, args, dev):
    s1 = pl.read_parquet(P.w(split, "s1.parquet"), columns=["idx", "country"])
    q = pl.read_parquet(P.w(split, "q.parquet"), columns=["idx", "country"])
    E1 = np.load(P.w("emb", f"{split}_s1.npy"), mmap_mode="r")
    EQ = np.load(P.w("emb", f"{split}_q.npy"), mmap_mode="r")
    parts = []
    countries = sorted(set(s1["country"].unique().to_list()) | set(q["country"].unique().to_list()))
    for c in countries:
        si = s1.filter(pl.col("country") == c)["idx"].to_numpy()
        qi = q.filter(pl.col("country") == c)["idx"].to_numpy()
        if len(si) == 0 or len(qi) == 0:
            log.info(f"{split}/{c}: S1={len(si)} Q={len(qi)} -> no candidates")
            continue
        with timer(f"{split}/{c}: dense kNN S1={len(si)} Q={len(qi)}"):
            A = torch.from_numpy(np.ascontiguousarray(E1[si])).to(dev)
            B = torch.from_numpy(np.ascontiguousarray(EQ[qi])).to(dev)
            fv, fj = topk_chunks(B, A, args.k, args.chunk)          # query -> S1
            rv, rj = topk_chunks(A, B, args.r, max(64, args.chunk // 4))  # S1 -> query
            del A, B
            torch.cuda.empty_cache() if dev == "cuda" else None
        kq, kr = fj.shape[1], rj.shape[1]
        fwd = pl.DataFrame({
            "q_idx": np.repeat(qi, kq).astype(np.int32),
            "s1_idx": si[fj.ravel()].astype(np.int32),
            "cos": fv.ravel().astype(np.float32),
            "rq": np.tile(np.arange(kq, dtype=np.uint8), len(qi)),
        })
        rev = pl.DataFrame({
            "q_idx": qi[rj.ravel()].astype(np.int32),
            "s1_idx": np.repeat(si, kr).astype(np.int32),
            "cos": rv.ravel().astype(np.float32),
            "rs": np.tile(np.arange(kr, dtype=np.uint8), len(si)),
        })
        if args.rev_min_cos > -1:
            rev = rev.filter(pl.col("cos") >= args.rev_min_cos)
        m = fwd.join(rev.select("q_idx", "s1_idx", "rs", pl.col("cos").alias("cos_r")),
                     on=["q_idx", "s1_idx"], how="full", coalesce=True)
        m = m.with_columns(pl.coalesce("cos", "cos_r").alias("cos"),
                           pl.col("rq").fill_null(255), pl.col("rs").fill_null(255)).drop("cos_r")
        parts.append(m)
        log.info(f"{split}/{c}: {m.height} candidate pairs ({m.height / len(qi):.2f} per query)")
    cands = pl.concat(parts).sort("q_idx", "cos", descending=[False, True])
    cands.write_parquet(P.w("cands", f"{split}.parquet"))
    return cands


def recall_report(P, cands):
    """Blocking recall on the training split, overall and for the validation fold."""
    gt = pl.read_parquet(P.w("train", "gt.parquet"))
    s1 = pl.read_parquet(P.w("train", "s1.parquet"), columns=["idx", "fold"])
    g = gt.join(s1.rename({"idx": "s1_idx"}), on="s1_idx")
    hit = g.join(cands.select("q_idx", "s1_idx", "rq", "rs"), on=["q_idx", "s1_idx"], how="left")
    for name, sub in (("all", hit), ("val", hit.filter(pl.col("fold") == 0))):
        r = sub["rq"].is_not_null().mean()
        msg = f"blocking recall[{name}] = {r:.5f} over {sub.height} true pairs"
        for k in (1, 3, 5, 10, 20):
            msg += f" | @{k}:{(sub['rq'].fill_null(255) < k).mean():.4f}"
        log.info(msg)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--work-dir", required=True)
    ap.add_argument("--k", type=int, default=10, help="S1 candidates per query")
    ap.add_argument("--r", type=int, default=10, help="query candidates per S1 (reverse)")
    ap.add_argument("--rev-min-cos", type=float, default=-2.0)
    ap.add_argument("--chunk", type=int, default=2048)
    ap.add_argument("--splits", default="train,test")
    args = ap.parse_args()
    P = Paths(args.data_dir, args.work_dir)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    for split in args.splits.split(","):
        c = retrieve_split(P, split, args, dev)
        n = c.height
        log.info(f"{split}: total candidate pairs {n}")
        if split == "train":
            recall_report(P, c)


if __name__ == "__main__":
    main()
