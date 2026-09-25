"""Create a small copy of the dataset (same layout) for smoke-testing the pipeline.

Keeps a random sample of Source 1 entities, all their true matches, and a
proportional random sample of the Source 2/3 records that match nothing.
"""
import argparse
import os

import polars as pl


def scan(p):
    return pl.scan_csv(p, separator="\t", quote_char=None, infer_schema=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--frac", type=float, default=0.01)
    args = ap.parse_args()
    for split in ("train", "test"):
        d_in = os.path.join(args.data_dir, split)
        d_out = os.path.join(args.out_dir, split)
        os.makedirs(d_out, exist_ok=True)
        s1 = scan(os.path.join(d_in, f"{split}_source1.tsv")).collect().sample(fraction=args.frac, seed=0)
        s1.write_csv(os.path.join(d_out, f"{split}_source1.tsv"), separator="\t")
        if split == "train":
            gt_all = scan(os.path.join(d_in, "train_ground_truth.tsv"))
            gt = gt_all.join(s1.lazy().select(pl.col("entity_id").alias("source1_entity_id")),
                             on="source1_entity_id").collect()
            gt.write_csv(os.path.join(d_out, "train_ground_truth.tsv"), separator="\t")
            ids = lambda g: (g.select(pl.col("matched_entity_ids").fill_null("").str.split(",").alias("entity_id"))
                             .explode("entity_id").filter(pl.col("entity_id") != ""))
            keep = ids(gt.lazy())
            matched_all = ids(gt_all)
        for k in (2, 3):
            df = scan(os.path.join(d_in, f"{split}_source{k}.tsv"))
            if split == "train":
                a = df.join(keep, on="entity_id", how="semi").collect()
                b = df.join(matched_all, on="entity_id", how="anti").collect().sample(fraction=args.frac, seed=k)
                out = pl.concat([a, b]).sample(fraction=1.0, shuffle=True, seed=k)
            else:
                out = df.collect().sample(fraction=args.frac, seed=k)
            out.write_csv(os.path.join(d_out, f"{split}_source{k}.tsv"), separator="\t")
            print(split, k, out.height, flush=True)


if __name__ == "__main__":
    main()
