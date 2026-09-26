# Business Entity Resolution — reproducible pipeline

Links every Source 2 / Source 3 record to the Source 1 entity it describes (or to
nothing), then writes `output/matching_results.tsv` and `output/candidate_pairs.tsv`.

Only the provided training data is used. No external APIs, databases, geocoders
or look-ups. The only pretrained model is
[`intfloat/multilingual-e5-small`](https://huggingface.co/intfloat/multilingual-e5-small)
(MIT licence, 118M parameters), fine-tuned here.

## Environment

- Tested in `nvcr.io/nvidia/pytorch:24.06-py3`: Python 3.10.12, torch 2.4.0a0+f70bd71a48.nv24.06,
  numpy 1.24.4; NVIDIA A100 MIG 3g.40gb slice, 15-CPU container quota, 2 TB RAM.
- `pip install -r requirements.txt` (exact pins; torch comes from the container).
- The only pretrained weights are `intfloat/multilingual-e5-small` (MIT licence, 117.7M
  parameters), read from the HuggingFace cache or downloaded once. With `HF_HUB_OFFLINE=1`
  nothing else is fetched; the code makes no other network calls.

## Data layout

```
<ROOT>/student_resource/dataset/{train,test}/*.tsv     # challenge data (unchanged)
<ROOT>/submission/code/business_entity_resolution/     # this folder
<ROOT>/work/                                           # intermediate files (created)
output/  (next to code/)                               # final TSVs (created)
<ROOT>/logs/                                           # stage logs (created)
```

Other locations work too: `DATA=/path/to/dataset WORK=/path/to/work OUT=/path/to/output
VALIDATOR=/path/to/student_resource/utils/validate_submission.py bash run.sh ...`

## Run end-to-end

```bash
cd code/business_entity_resolution
export DATA=/path/to/student_resource/dataset
bash run.sh all1     # prepare -> bi-encoder -> retrieval -> refold                        (~2 h)
bash run.sh all2     # features -> prune -> feats2 -> cross-encoder -> final -> decide -> validate (~2.5 h)
```

`run.sh` defaults to the submitted decision threshold (`THRESHOLD=0.75`), so these two
commands regenerate the submitted `output/matching_results.tsv` and `output/candidate_pairs.tsv`.

`bash run.sh smoke` runs both chains on a 1% sample in a few minutes. Each stage
can also be run on its own (`bash run.sh <stage>`), and each one reads the
previous stage's files from `WORK`:

| Stage | Module | What it does | Writes |
|---|---|---|---|
| `prepare` | `src/prepare.py` | Loads TSVs, learns a native-script→Latin token dictionary from training pairs, normalises names/addresses, splits addresses into street + locality components, finds each S1 entity's finest locality (from S1 frequencies) | `work/{train,test}/{s1,q}.parquet`, `translit.json` |
| `bienc` | `src/biencoder.py` | Fine-tunes e5-small as a bi-encoder (InfoNCE + same-name hard negatives), embeds every record | `work/models/bienc`, `work/emb/*.npy` |
| `retrieve` | `src/retrieve.py` | Exact GPU kNN inside each country label, both directions (query→S1 top-10, S1→query top-10) | `work/cands/*.parquet` |
| `refold` | `src/refold.py` | Gives each unmatched training record the fold of the S1 entity it competes with (its top-1 candidate) | `work/train/q.parquet` |
| `features` | `src/features.py` | 50 pair features (49 used; the source id `q_src` is dropped): fuzzy/TF-IDF name & address similarity, street-only similarity, finest-locality presence, numbers, embedding ranks/margins | `work/feats/*.parquet` |
| `prune` | `src/ranker.py --stage prune` | LightGBM on those features (out-of-fold scores on folds 1-4), keeps ≤4 candidates per query (p ≥ 0.003) | `work/pruned/*.parquet` |
| `feats2` | `src/features2.py` | 29 decoy features on pruned pairs: moved / shared house numbers, added legal words, extra-word log-odds (learned on folds 5-9), full-name ambiguity, cluster consistency | `work/feats2/*.parquet` |
| `xenc` | `src/crossencoder.py` | Fine-tunes a cross-encoder (listwise loss with a "no match" slot), scores pruned pairs | `work/models/xenc`, `work/xenc/*.npy` |
| `final` | `src/ranker.py --stage final` | LightGBM on 87 features: the pruner's 49, 26 of the 29 decoy features, pruner probability, cross-encoder logit and their rank/margin/gap features. The 4 count features in `FINAL_DROP` are excluded | `work/scored/*.parquet` |
| `decide` | `src/decide.py` | Reduces each query to its best S1, then keeps links by a threshold or by expected-F0.5 top-k per entity, whichever scores higher on validation. With `THRESHOLD` set (default 0.75), a fixed threshold is used instead | `output/*.tsv`, `work/decision.json` |
| `validate` | challenge validator | Format check of the two output files | — |

Diagnostics (not part of the submission path):
- `src/analyze.py`: sampled validation misses by stage, and false merges.
- `src/silver_check.py`: per country, how many easy high-confidence test pairs (same core name,
  same house number, shared street word, unique on the S1 side) a prediction file recovers.
- `src/bucket_diag.py`: per-country, per-band test precision estimates.
- `src/decoy_diag.py` and `src/test_diag.py`: validation vs test feature and profile comparisons.
- `src/veto.py`: an unused decoy post-filter.
- `src/quick_dense.py`: the dense baseline (LB 0.753).
- Other tooling: `src/make_dev_subset.py` (smoke data), `src/gpucheck.py`, `experiments.sh`.

## Decision threshold

`decide` can tune the rule on validation (`THRESHOLD= bash run.sh decide`). For the
final model it picks t = 0.36. The submitted files use **t = 0.75**, which is `run.sh`'s
default:

```bash
bash run.sh decide        # same as THRESHOLD=0.75 bash run.sh decide
```

The reason is a measured shift between validation and test. Test has about 23% more
Source 2/3 records per entity (extra decoys), and about 2× more kept links fall in the
0.36–0.75 probability range than on validation. US and India come from the training
generator, so their true links per entity per probability band should match validation.
Dividing validation true links per entity by test links per entity in each band gives
an implied test precision of about 40–54% below 0.75, and 73–94% above it. Under macro
F0.5 a link is worth keeping only if it is right at least about 77% of the time (a
false link costs an entity about 0.19; an extra true link gains about 0.056). So 0.75
is where the implied test precision crosses break-even. On validation this costs
0.00075 (0.99222 → 0.99147). No test labels are used; only the unlabeled distribution
of the model's test probabilities.

## Validation protocol

Source 1 entities are split into 10 folds by CRC32 of their id. Each matched
query inherits the fold of its true Source 1 entity; each unmatched query takes
the fold of the Source 1 entity it competes with (its top-1 retrieved candidate),
so decoys live in the same fold as the entity they could be wrongly merged into.

- Folds 5–9 train the base models (bi-encoder, cross-encoder) and learn the extra-word log-odds.
- Folds 1–4 train the LightGBM rankers, so they see out-of-sample base-model scores, as on test.
  The pruner's scores on folds 1–4 are out-of-fold (4 extra models).
- Fold 0 is held out: early stopping, decision-rule tuning and the reported F0.5 use only
  fold-0 entities and fold-0 queries.

The test set is scored with the models trained this way; no model weight is fitted
on test data. Two model-selection choices did use unlabeled test statistics: the 0.75
threshold (above), and removing the four count features in `FINAL_DROP` (`src/ranker.py`).
The latter followed a comparison of test vs validation feature values on decoy-signature
links (`src/decoy_diag.py`).

## Determinism

Seeds are fixed (torch, Python `random`, LightGBM, polars sampling). A rerun is still not
bit-identical. bf16/TF32 GPU training kernels are non-deterministic. Some polars joins feed
seeded samples (bi-encoder pair sampling), and some tie-breaks (finest-locality choice,
transliteration votes) follow polars' unspecified row order. Expect near-identical
validation scores, but not identical borderline links. The submitted `output/*.tsv` are the
canonical results.
