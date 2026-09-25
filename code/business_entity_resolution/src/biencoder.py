"""Step 2: fine-tune a multilingual bi-encoder on true (Source1, Source2/3) pairs,
then embed every record of the train and test splits.

Model: intfloat/multilingual-e5-small (MIT licence, 118M parameters).
Loss : symmetric InfoNCE with in-batch negatives. Batches are drawn from a single
       country so in-batch negatives are realistic, and every pair carries one mined
       hard negative: another Source 1 record with the same core name (different
       entity), which forces the encoder to use the address to disambiguate.
Only pairs whose Source 1 entity is in the training folds (fold != 0) are used,
so the validation fold stays unseen.
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

from .common import Paths, log, timer

DEFAULT_MODEL = "intfloat/multilingual-e5-small"


class Encoder(torch.nn.Module):
    def __init__(self, name):
        super().__init__()
        self.m = AutoModel.from_pretrained(name)

    def forward(self, ids, mask):
        h = self.m(input_ids=ids, attention_mask=mask).last_hidden_state
        m = mask.unsqueeze(-1).to(h.dtype)
        e = (h * m).sum(1) / m.sum(1).clamp(min=1)
        return F.normalize(e.float(), dim=-1)


class BatchDS(Dataset):
    def __init__(self, batches, s_txt, q_txt, tok, max_len):
        self.b, self.s, self.q, self.tok, self.L = batches, s_txt, q_txt, tok, max_len

    def __len__(self):
        return len(self.b)

    def __getitem__(self, i):
        qi, si, hi = self.b[i]
        enc = lambda txt: self.tok(txt, max_length=self.L, truncation=True, padding=True,
                                   return_tensors="pt")
        return (enc([self.q[j] for j in qi]), enc([self.s[j] for j in si]),
                enc([self.s[j] for j in hi]), torch.tensor(si), torch.tensor(hi))


def make_batches(pairs, s1, bs, max_pairs, seed):
    """Country-homogeneous batches of (q_idx, s1_idx, hardneg_s1_idx)."""
    rng = random.Random(seed)
    s1c = s1.select("idx", "country", "ncore")
    p = pairs.join(s1c, left_on="s1_idx", right_on="idx")
    if p.height > max_pairs:
        p = p.sample(max_pairs, seed=seed)
    # hard negatives: random other S1 with same (country, core name)
    grp = s1c.group_by("country", "ncore").agg(pl.col("idx"))
    gmap = {(c, n): ids for c, n, ids in grp.iter_rows()}
    all_by_c = {c: s1c.filter(pl.col("country") == c)["idx"].to_list()
                for c in s1c["country"].unique().to_list()}
    batches = []
    for c in p["country"].unique().to_list():
        sub = p.filter(pl.col("country") == c)
        rows = list(zip(sub["q_idx"].to_list(), sub["s1_idx"].to_list(), sub["ncore"].to_list()))
        rng.shuffle(rows)
        pool = all_by_c[c]
        for i in range(0, len(rows) - bs + 1, bs):
            chunk = rows[i:i + bs]
            hn = []
            for _, si, nc in chunk:
                g = gmap.get((c, nc), [])
                h = si
                if len(g) > 1:
                    for _ in range(4):
                        h = g[rng.randrange(len(g))]
                        if h != si:
                            break
                if h == si:
                    h = pool[rng.randrange(len(pool))]
                hn.append(h)
            batches.append(([r[0] for r in chunk], [r[1] for r in chunk], hn))
    rng.shuffle(batches)
    return batches


def train(P, args):
    dev = "cuda"
    s1 = pl.read_parquet(P.w("train", "s1.parquet"), columns=["idx", "country", "ncore", "fold", "mtext"])
    q = pl.read_parquet(P.w("train", "q.parquet"), columns=["idx", "mtext"])
    gt = pl.read_parquet(P.w("train", "gt.parquet"))
    tr = gt.join(s1.select(pl.col("idx").alias("s1_idx"), "fold"), on="s1_idx").filter(pl.col("fold") != 0)
    s_txt, q_txt = s1["mtext"].to_list(), q["mtext"].to_list()
    tok = AutoTokenizer.from_pretrained(args.model)
    model = Encoder(args.model).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    steps_total = 0
    batches_ep = []
    for ep in range(args.epochs):
        batches_ep.append(make_batches(tr, s1, args.bs, args.max_pairs, seed=ep))
        steps_total += len(batches_ep[-1])
    sched = get_linear_schedule_with_warmup(opt, int(0.05 * steps_total), steps_total)
    log.info(f"bi-encoder: {tr.height} train pairs, {steps_total} steps, bs={args.bs}")
    step = 0
    model.train()
    for ep in range(args.epochs):
        dl = DataLoader(BatchDS(batches_ep[ep], s_txt, q_txt, tok, args.max_len), batch_size=None,
                        shuffle=False, num_workers=args.num_workers, prefetch_factor=4 if args.num_workers else None)
        for qe, se, he, si, hi in dl:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                zq = model(qe["input_ids"].to(dev), qe["attention_mask"].to(dev))
                zs = model(se["input_ids"].to(dev), se["attention_mask"].to(dev))
                zh = model(he["input_ids"].to(dev), he["attention_mask"].to(dev))
            si, hi = si.to(dev), hi.to(dev)
            cand = torch.cat([zs, zh])                       # 2B candidates
            cid = torch.cat([si, hi])
            logits = zq @ cand.T / args.tau                  # B x 2B
            same = si[:, None] == cid[None, :]
            eye = torch.zeros_like(same)
            eye[:, :len(si)] = torch.eye(len(si), dtype=torch.bool, device=dev)
            logits = logits.masked_fill(same & ~eye, -1e4)
            lbl = torch.arange(len(si), device=dev)
            l1 = F.cross_entropy(logits, lbl)
            logits2 = zs @ zq.T / args.tau                   # S1 -> query direction
            same2 = si[:, None] == si[None, :]
            logits2 = logits2.masked_fill(same2 & ~torch.eye(len(si), dtype=torch.bool, device=dev), -1e4)
            loss = l1 + F.cross_entropy(logits2, lbl)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            step += 1
            if step % 200 == 0 or step == 1:
                acc = (logits.argmax(1) == lbl).float().mean().item()
                log.info(f"ep {ep} step {step}/{steps_total} loss {loss.item():.4f} in-batch acc {acc:.4f}")
    P.w("models", "bienc", "config.json")  # creates the directory
    model.m.save_pretrained(P.w("models", "bienc"))
    tok.save_pretrained(P.w("models", "bienc"))


class TextDS(Dataset):
    def __init__(self, txt, order, tok, bs, max_len):
        self.txt, self.order, self.tok, self.bs, self.L = txt, order, tok, bs, max_len

    def __len__(self):
        return math.ceil(len(self.order) / self.bs)

    def __getitem__(self, i):
        ids = self.order[i * self.bs:(i + 1) * self.bs]
        e = self.tok([self.txt[j] for j in ids], max_length=self.L, truncation=True,
                     padding=True, return_tensors="pt")
        return torch.from_numpy(ids), e


@torch.no_grad()
def embed(P, args):
    dev = "cuda"
    path = P.w("models", "bienc")
    tok = AutoTokenizer.from_pretrained(path)
    model = Encoder(path).to(dev).eval().to(torch.bfloat16)
    for split in ("train", "test"):
        for nm in ("s1", "q"):
            out_p = P.w("emb", f"{split}_{nm}.npy")
            if os.path.exists(out_p) and not args.overwrite:
                log.info(f"skip existing {out_p}")
                continue
            txt = pl.read_parquet(P.w(split, f"{nm}.parquet"), columns=["mtext"])["mtext"].to_list()
            order = np.argsort(np.fromiter((len(t) for t in txt), np.int32, len(txt)))
            out = np.lib.format.open_memmap(out_p, mode="w+", dtype=np.float16,
                                            shape=(len(txt), model.m.config.hidden_size))
            dl = DataLoader(TextDS(txt, order, tok, args.enc_bs, args.max_len), batch_size=None,
                            num_workers=args.num_workers)
            with timer(f"embed {split}/{nm} ({len(txt)})"):
                for ids, e in dl:
                    z = model(e["input_ids"].to(dev), e["attention_mask"].to(dev))
                    out[ids.numpy()] = z.half().cpu().numpy()
            out.flush()
            del out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--work-dir", required=True)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--bs", type=int, default=1024)
    ap.add_argument("--enc-bs", type=int, default=2048)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--tau", type=float, default=0.05)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--max-pairs", type=int, default=4_000_000)
    ap.add_argument("--max-len", type=int, default=80)
    ap.add_argument("--num-workers", type=int, default=6)
    ap.add_argument("--skip-train", action="store_true")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    P = Paths(args.data_dir, args.work_dir)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.manual_seed(0)
    if not args.skip_train:
        with timer("train bi-encoder"):
            train(P, args)
    embed(P, args)


if __name__ == "__main__":
    main()
