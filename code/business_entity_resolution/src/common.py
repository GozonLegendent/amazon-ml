"""Shared paths, IO helpers and logging."""
import json
import logging
import os
import sys
import time
import zlib
from contextlib import contextmanager

import polars as pl

log = logging.getLogger("ber")
if not log.handlers:
    h = logging.StreamHandler(sys.stdout)
    h.setFormatter(logging.Formatter("%(asctime)s %(message)s", "%H:%M:%S"))
    log.addHandler(h)
    log.setLevel(logging.INFO)


@contextmanager
def timer(msg):
    t = time.time()
    log.info(f"[start] {msg}")
    yield
    log.info(f"[done ] {msg} in {time.time() - t:.1f}s")


def read_tsv(path):
    """Read a challenge TSV with every column as a string (no quoting)."""
    return pl.read_csv(path, separator="\t", quote_char=None, infer_schema=False,
                       encoding="utf8")


def val_fold(entity_ids, n_folds=10):
    """Deterministic fold assignment by CRC32 of the id (fold 0 = validation)."""
    return [zlib.crc32(e.encode()) % n_folds for e in entity_ids]


class Paths:
    def __init__(self, data_dir, work_dir):
        self.data = data_dir
        self.work = work_dir
        os.makedirs(work_dir, exist_ok=True)

    def src(self, split, k):
        return os.path.join(self.data, split, f"{split}_source{k}.tsv")

    def gt(self):
        return os.path.join(self.data, "train", "train_ground_truth.tsv")

    def w(self, *parts):
        p = os.path.join(self.work, *parts)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        return p


def save_json(obj, path):
    with open(path, "w") as f:
        json.dump(obj, f, ensure_ascii=False, indent=1)


def load_json(path):
    with open(path) as f:
        return json.load(f)
