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


def knn(Q, D, k, budget, dev):
    """Exact top-k inner product of every row of Q against rows of D.

    Q, D are CPU fp16 tensors. Works inside a fixed GPU memory budget (bytes):
    D is processed in column blocks that fit, Q in row chunks, and per-block
    top-k lists are merged, so any size runs on a small / shared GPU.
    """
    nq, d = Q.shape
    nd = D.shape[0]
    k = min(k, nd)
    db = int(min(nd, max(1024, budget * 0.35 // (2 * d))))           # D block rows
    qc = int(max(64, min(nq, budget * 0.45 // (db * 2 * 3))))         # sims + topk workspace
    best_v = torch.empty((nq, k), dtype=torch.float32)
    best_i = torch.empty((nq, k), dtype=torch.int32)
    for bi, c0 in enumerate(range(0, nd, db)):
        Dg = D[c0:c0 + db].to(dev, non_blocking=True)
        for q0 in range(0, nq, qc):
            s = Q[q0:q0 + qc].to(dev, non_blocking=True) @ Dg.T
            v, j = s.topk(min(k, Dg.shape[0]), dim=1)
            v, j = v.float(), (j + c0).int()
            if bi > 0:
                v = torch.cat([best_v[q0:q0 + qc].to(dev), v], 1)
                j = torch.cat([best_i[q0:q0 + qc].to(dev), j], 1)
                v, o = v.topk(k, dim=1)
                j = j.gather(1, o)
            if v.shape[1] < k:  # only when the first block is smaller than k
                pad = k - v.shape[1]
                v = torch.cat([v, torch.full((v.shape[0], pad), -9.0, device=dev)], 1)
                j = torch.cat([j, torch.zeros((j.shape[0], pad), dtype=torch.int32, device=dev)], 1)
            best_v[q0:q0 + qc] = v.cpu()
            best_i[q0:q0 + qc] = j.cpu()
        del Dg
    return best_v.numpy(), best_i.numpy()


def gpu_budget(args):
    """GPU bytes we allow ourselves: --gpu-mem-gb, else 60% of what is free right now."""
    if args.gpu_mem_gb:
        return int(args.gpu_mem_gb * 2**30)
    if torch.cuda.is_available():
        return int(torch.cuda.mem_get_info()[0] * 0.6)
    return 2 * 2**30


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
            A = torch.from_numpy(np.ascontiguousarray(E1[si]))
            B = torch.from_numpy(np.ascontiguousarray(EQ[qi]))
            if dev == "cuda":
                A, B = A.pin_memory(), B.pin_memory()
            else:
                A, B = A.float(), B.float()
            budget = gpu_budget(args)
            fv, fj = knn(B, A, args.k, budget, dev)                  # query -> S1
            if args.r > 0:
                rv, rj = knn(A, B, args.r, budget, dev)              # S1 -> query
            else:
                rv, rj = np.zeros((len(si), 0), np.float32), np.zeros((len(si), 0), np.int32)
            del A, B
            if dev == "cuda":
                torch.cuda.empty_cache()
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
    ap.add_argument("--gpu-mem-gb", type=float, default=0, help="GPU memory budget; 0 = 60%% of free")
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
