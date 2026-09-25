# Snapshot: leaderboard 0.972 (v0, pruner decoding)

| | |
|---|---|
| Public leaderboard F0.5 | **0.972** (submitted 26 Sep 2026) |
| Validation macro F0.5 | 0.9795 (220,907 fold-0 S1 entities; older protocol with hash-fold decoys, which was somewhat optimistic) |
| Code | this folder = repo commit `5cf390b` (`code/business_entity_resolution/`) |

## How the submission was produced

```bash
bash run.sh all1          # prepare -> bi-encoder (folds 5-9) -> exact GPU kNN retrieval
bash run.sh features      # 42 pair features
bash run.sh prune         # LightGBM pruner, keeps <= 4 candidates per record
python -m src.decide --data-dir <dataset> --work-dir <work> --out-dir <out> --scores pruned
```

The decision step assigned each Source 2/3 record to its highest-probability Source 1
entity, using the pruner's probabilities, when p >= 0.75. That threshold was tuned for
macro F0.5 on fold 0. The cross-encoder and the final ranker were not used.

## Validation breakdown

| | F0.5 |
|---|---|
| Singletons | 0.978 |
| Non-singletons | 0.980 |
| US | 0.978 |
| India | 0.982 |

Micro precision was 0.9947 and micro recall 0.951. Blocking recall was 99.73%, and 99.12% after pruning.
