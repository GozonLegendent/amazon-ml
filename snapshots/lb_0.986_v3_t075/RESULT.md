# Snapshot: leaderboard 0.986 (v3, threshold 0.75)

| | |
|---|---|
| Public leaderboard F0.5 | **0.986** (submitted 27 Sep 2026) |
| Validation macro F0.5 | 0.99147 at t = 0.75 (strict protocol; 0.99222 at the validation-tuned t = 0.36) |
| Code | this folder = repo commit `cc930d3` (`code/business_entity_resolution/`) |

## What changed vs the 0.972 snapshot

- Honest folds: unmatched records take the fold of the entity they compete with (`refold`).
- The pruner's scores are out-of-fold on the rankers' training folds.
- The cross-encoder (listwise loss, "no match" slot) is added; it is the dominant signal.
- Decoy features (`features2`): moved house numbers, added legal words, extra-word log-odds.
- Street and finest-locality address structure, French-safe normalisation, and the source id dropped.
- **Cluster-count features were removed from the final ranker.** On test, sibling decoys come as
  several records sharing the moved house number, which made them look genuine (test decoy
  acceptance fell from 9.1% to 0.98%).
- **Threshold 0.75 instead of the validation-optimal 0.36.** Test has about 2× more borderline
  links than validation, and their implied precision is below the F0.5 break-even of ~0.77.

## Reproduce

```bash
bash run.sh all1
bash run.sh features && bash run.sh prune && bash run.sh feats2 && bash run.sh xenc && bash run.sh final
THRESHOLD=0.75 bash run.sh decide && bash run.sh validate
```

## Test output

5,865,437 links. Links per S1: US 3.395, India 3.392, France 3.343.
