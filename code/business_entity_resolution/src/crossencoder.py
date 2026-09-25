"""Step 5: cross-encoder that reads a (Source 1, Source 2/3) pair jointly and scores it.

Backbone: the fine-tuned bi-encoder (intfloat/multilingual-e5-small, MIT, 118M
params) with a linear head on the mean-pooled pair representation. Input: "<s1 text> </s> <query text>".

Training data: queries whose true Source 1 entity is in the base-model folds (5-9),
plus unmatched queries hashed into those folds. Every such query contributes its
surviving candidates after pruning, so negatives are exactly the hard confusions
the earlier stages produce (same-name franchises, same-street neighbours).
Loss: listwise softmax over each query's candidates plus a learned "no match"
slot, so the model learns both *which* S1 wins and *whether any* S1 matches.
It is then applied to every pruned candidate pair of both splits; its logit
becomes a feature for the final ranker.
"""
import argparse
import math
import os
import random

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np
import polars as pl
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup

from .biencoder import BASE_FOLDS, device
from .common import Paths, effective_cpus, log, timer


class CrossEncoder(torch.nn.Module):
    def __init__(self, name):
        super().__init__()
        self.m = AutoModel.from_pretrained(name)
        self.head = torch.nn.Linear(self.m.config.hidden_size, 1)

    def forward(self, ids, mask):
        h = self.m(input_ids=ids, attention_mask=mask).last_hidden_state
        m = mask.unsqueeze(-1).to(h.dtype)
        e = (h * m).sum(1) / m.sum(1).clamp(min=1)
        # match the head's dtype (fp32 in training under autocast, bf16 when scoring)
        return self.head(e.to(self.head.weight.dtype)).float().squeeze(-1)


class GroupDS(Dataset):
    """Each item = a batch of query groups; each group = (q, [s1...], label_pos or -1)."""

    def __init__(self, batches, s_txt, q_txt, tok, max_len):
        self.b, self.s, self.q, self.tok, self.L = batches, s_txt, q_txt, tok, max_len

    def __len__(self):
        return len(self.b)

    def __getitem__(self, i):
        groups = self.b[i]
        a, b, gid, lbl = [], [], [], []
        for g, (qi, sis, pos) in enumerate(groups):
            for si in sis:
                a.append(self.s[si])
                b.append(self.q[qi])
                gid.append(g)
            lbl.append(pos)
        e = self.tok(a, b, max_length=self.L, truncation="longest_first", padding=True,
                     return_tensors="pt")
        return e, torch.tensor(gid), torch.tensor(lbl)


def train_groups(P, n_cand, max_queries, seed):
    c = pl.read_parquet(P.w("pruned", "train.parquet"), columns=["q_idx", "s1_idx", "p_a"])
    q = pl.read_parquet(P.w("train", "q.parquet"), columns=["idx", "fold", "true_s1"])
    q = q.filter(pl.col("fold").is_in(BASE_FOLDS))
    if q.height > max_queries:
        q = q.sample(max_queries, seed=seed)
    c = c.join(q.rename({"idx": "q_idx"}), on="q_idx")
    c = c.sort("q_idx", "p_a", descending=[False, True]).group_by("q_idx", maintain_order=True).head(n_cand)
    g = c.group_by("q_idx", maintain_order=True).agg(pl.col("s1_idx"), pl.col("true_s1").first())
    groups = []
    for qi, sis, ts in g.iter_rows():
        pos = sis.index(ts) if ts is not None and ts in sis else -1
        groups.append((qi, sis, pos))
    log.info(f"cross-encoder train groups: {len(groups)} "
             f"(with a match in candidates: {sum(p >= 0 for _, _, p in groups)})")
    return groups


def listwise_loss(logits, gid, lbl, none_logit):
    """Softmax over each group's candidates + a 'no match' slot."""
    ng = lbl.shape[0]
    maxn = int(torch.bincount(gid, minlength=ng).max())
    mat = torch.full((ng, maxn + 1), -1e4, device=logits.device, dtype=logits.dtype)
    pos_in_group = torch.zeros_like(gid)
    # position of each row inside its group (rows of a group are contiguous)
    starts = torch.zeros(ng, dtype=torch.long, device=gid.device)
    counts = torch.bincount(gid, minlength=ng)
    starts[1:] = torch.cumsum(counts, 0)[:-1]
    pos_in_group = torch.arange(len(gid), device=gid.device) - starts[gid]
    mat[gid, pos_in_group] = logits
    mat[:, maxn] = none_logit
    target = torch.where(lbl >= 0, lbl, torch.full_like(lbl, maxn))
    return F.cross_entropy(mat, target)


def train(P, args):
    dev = device()
    s_txt = pl.read_parquet(P.w("train", "s1.parquet"), columns=["mtext"])["mtext"].to_list()
    q_txt = pl.read_parquet(P.w("train", "q.parquet"), columns=["mtext"])["mtext"].to_list()
    groups = train_groups(P, args.n_cand, args.max_queries, seed=0)
    rng = random.Random(0)
    rng.shuffle(groups)
    gpb = max(1, args.bs // args.n_cand)
    batches = [groups[i:i + gpb] for i in range(0, len(groups), gpb)]
    tok = AutoTokenizer.from_pretrained(args.model)
    model = CrossEncoder(args.model).to(dev)
    try:
        model.m.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    except TypeError:
        model.m.gradient_checkpointing_enable()
    none_logit = torch.nn.Parameter(torch.zeros((), device=dev))
    opt = torch.optim.AdamW([{"params": model.parameters()}, {"params": [none_logit], "lr": 1e-3}],
                            lr=args.lr, weight_decay=0.01)
    total = len(batches) * args.epochs
    sched = get_linear_schedule_with_warmup(opt, int(0.05 * total), total)
    model.train()
    step = 0
    for ep in range(args.epochs):
        dl = DataLoader(GroupDS(batches, s_txt, q_txt, tok, args.max_len), batch_size=None,
                        shuffle=ep > 0, num_workers=args.num_workers,
                        prefetch_factor=4 if args.num_workers else None)
        for e, gid, lbl in dl:
            gid, lbl = gid.to(dev), lbl.to(dev)
            with torch.autocast(dev, dtype=torch.bfloat16, enabled=dev == "cuda"):
                logits = model(e["input_ids"].to(dev), e["attention_mask"].to(dev))
            loss = listwise_loss(logits.float(), gid, lbl, none_logit)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            step += 1
            if step % 200 == 0 or step == 1:
                mem = torch.cuda.max_memory_allocated() / 2**30 if dev == "cuda" else 0.0
                log.info(f"ep {ep} step {step}/{total} loss {loss.item():.4f} "
                         f"none_logit {none_logit.item():.2f} peak mem {mem:.1f} GiB")
    out = P.w("models", "xenc", "config.json")
    model.m.save_pretrained(os.path.dirname(out))
    tok.save_pretrained(os.path.dirname(out))
    torch.save({"head": model.head.state_dict(), "none_logit": none_logit.item()},
               os.path.join(os.path.dirname(out), "head.pt"))


class PairDS(Dataset):
    def __init__(self, ia, ib, s_txt, q_txt, order, tok, bs, max_len):
        self.ia, self.ib, self.s, self.q = ia, ib, s_txt, q_txt
        self.order, self.tok, self.bs, self.L = order, tok, bs, max_len

    def __len__(self):
        return math.ceil(len(self.order) / self.bs)

    def __getitem__(self, i):
        rows = self.order[i * self.bs:(i + 1) * self.bs]
        e = self.tok([self.s[self.ia[r]] for r in rows], [self.q[self.ib[r]] for r in rows],
                     max_length=self.L, truncation="longest_first", padding=True, return_tensors="pt")
        return torch.from_numpy(rows), e


@torch.no_grad()
def score(P, args):
    dev = device()
    path = os.path.dirname(P.w("models", "xenc", "config.json"))
    tok = AutoTokenizer.from_pretrained(path)
    model = CrossEncoder(path)
    model.head.load_state_dict(torch.load(os.path.join(path, "head.pt"))["head"])
    model = model.to(dev).eval()
    if dev == "cuda":
        model = model.to(torch.bfloat16)
    for split in args.splits.split(","):
        out_p = P.w("xenc", f"{split}.npy")
        c = pl.read_parquet(P.w("pruned", f"{split}.parquet"), columns=["q_idx", "s1_idx"])
        if os.path.exists(out_p) and not args.overwrite and np.load(out_p, mmap_mode="r").shape[0] == c.height:
            log.info(f"skip existing {out_p}")
            continue
        ia, ib = c["s1_idx"].to_numpy(), c["q_idx"].to_numpy()
        s_txt = pl.read_parquet(P.w(split, "s1.parquet"), columns=["mtext"])["mtext"].to_list()
        q_txt = pl.read_parquet(P.w(split, "q.parquet"), columns=["mtext"])["mtext"].to_list()
        slen = np.fromiter((len(t) for t in s_txt), np.int32, len(s_txt))
        qlen = np.fromiter((len(t) for t in q_txt), np.int32, len(q_txt))
        order = np.argsort(slen[ia] + qlen[ib], kind="stable")
        out = np.zeros(len(ia), np.float32)
        dl = DataLoader(PairDS(ia, ib, s_txt, q_txt, order, tok, args.enc_bs, args.max_len),
                        batch_size=None, num_workers=args.num_workers,
                        prefetch_factor=4 if args.num_workers else None)
        with timer(f"cross-encoder score {split} ({len(ia)} pairs)"):
            for n, (rows, e) in enumerate(dl):
                out[rows.numpy()] = model(e["input_ids"].to(dev), e["attention_mask"].to(dev)).float().cpu().numpy()
                if n % 2000 == 0:
                    log.info(f"  {split}: {min((n + 1) * args.enc_bs, len(ia))}/{len(ia)} pairs")
        np.save(out_p, out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--work-dir", required=True)
    ap.add_argument("--model", default=None,
                    help="backbone; default = the fine-tuned bi-encoder (domain-adapted e5-small)")
    ap.add_argument("--n-cand", type=int, default=4, help="candidates per query in training")
    ap.add_argument("--max-queries", type=int, default=800_000)
    ap.add_argument("--bs", type=int, default=512, help="pairs per training batch")
    ap.add_argument("--enc-bs", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=4e-5)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--max-len", type=int, default=128)
    ap.add_argument("--num-workers", type=int, default=max(2, min(12, effective_cpus() - 3)))
    ap.add_argument("--splits", default="train,test")
    ap.add_argument("--skip-train", action="store_true")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    P = Paths(args.data_dir, args.work_dir)
    if args.model is None:
        bienc = os.path.dirname(P.w("models", "bienc", "config.json"))
        args.model = bienc if os.path.exists(os.path.join(bienc, "config.json")) else "intfloat/multilingual-e5-small"
    log.info(f"cross-encoder backbone: {args.model}")
    torch.manual_seed(0)
    torch.backends.cuda.matmul.allow_tf32 = True
    if not args.skip_train:
        with timer("train cross-encoder"):
            train(P, args)
    score(P, args)


if __name__ == "__main__":
    main()
