# Business Entity Resolution — reproducible pipeline

Links every Source 2 / Source 3 record to the Source 1 entity it describes (or to
nothing), then writes `output/matching_results.tsv` and `output/candidate_pairs.tsv`.

Only the provided training data is used. No external APIs, databases, geocoders
or look-ups. The only pretrained model is
[`intfloat/multilingual-e5-small`](https://huggingface.co/intfloat/multilingual-e5-small)
(MIT licence, 118M parameters), fine-tuned here.

## Environment

- Python 3.10, PyTorch 2.x with CUDA (tested on an NVIDIA A100 MIG 3g.40gb slice,
  256 CPU cores, 2 TB RAM).
- `pip install -r requirements.txt` (torch is assumed pre-installed).
- The e5-small checkpoint must be in the HuggingFace cache or downloadable.

## Data layout

```
<ROOT>/student_resource/dataset/{train,test}/*.tsv     # challenge data (unchanged)
<ROOT>/submission/code/business_entity_resolution/     # this folder
<ROOT>/work/                                           # intermediate files (created)
<ROOT>/submission/output/                              # final TSVs (created)
<ROOT>/logs/                                           # stage logs (created)
```

Other locations work too: `DATA=/path/to/dataset WORK=/path/to/work OUT=/path/to/output bash run.sh ...`

## Run end-to-end

```bash
cd code/business_entity_resolution
bash run.sh all1     # prepare -> bi-encoder -> retrieval           (~1-1.5 h)
bash run.sh all2     # features -> prune -> cross-encoder -> final -> decide -> validate (~1.5 h)
```

`bash run.sh smoke` runs both chains on a 1% sample in a few minutes. Each stage
can also be run on its own (`bash run.sh <stage>`), and each one reads the
previous stage's files from `WORK`:

| Stage | Module | What it does | Writes |
|---|---|---|---|
| `prepare` | `src/prepare.py` | Loads TSVs, learns a native-script→Latin token dictionary from training pairs, normalises names/addresses | `work/{train,test}/{s1,q}.parquet`, `translit.json` |
| `bienc` | `src/biencoder.py` | Fine-tunes e5-small as a bi-encoder (InfoNCE + same-name hard negatives), embeds every record | `work/models/bienc`, `work/emb/*.npy` |
| `retrieve` | `src/retrieve.py` | Exact GPU kNN inside each country label, both directions (query→S1 top-10, S1→query top-10) | `work/cands/*.parquet` |
| `features` | `src/features.py` | 42 pair features (fuzzy/TF-IDF name & address similarity, numbers, embedding ranks/margins) | `work/feats/*.parquet` |
| `prune` | `src/ranker.py --stage prune` | LightGBM on cheap features, keeps ≤4 candidates per query (p ≥ 0.003) | `work/pruned/*.parquet` |
| `xenc` | `src/crossencoder.py` | Fine-tunes a cross-encoder (listwise loss with a "no match" slot), scores pruned pairs | `work/models/xenc`, `work/xenc/*.npy` |
| `final` | `src/ranker.py --stage final` | LightGBM on all features + cross-encoder logit + competition features | `work/scored/*.parquet` |
| `decide` | `src/decide.py` | Assigns each query to its best S1 if p ≥ t; t tuned for macro F0.5 on validation | `output/*.tsv`, `work/decision.json` |
| `validate` | challenge validator | Format check of the two output files | — |

## Validation protocol

Source 1 entities are split into 10 folds by CRC32 of their id. Each query
inherits the fold of its true Source 1 entity (unmatched queries hash their own id).

- Folds 5–9 train the base models (bi-encoder, cross-encoder).
- Folds 1–4 train the LightGBM rankers, so they see out-of-sample base-model scores, as on test.
- Fold 0 is held out for early stopping, threshold tuning and the reported F0.5.

The test set is scored with the models trained this way. Nothing about the test
set is used for fitting.

## Determinism

Seeds are fixed (torch/numpy/LightGBM/sampling). GPU kernels may still introduce
tiny floating-point differences between runs.
