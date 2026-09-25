# ML Challenge 2026: Business Entity Resolution Solution Template

**Team Name:** [Your Team Name]  
**Team Members:** [List all team members]  
**Submission Date:** [Date]

---

## 1. Executive Summary

We treat entity resolution as **assignment**. In the training labels every Source 2/3 record
matches at most one Source 1 entity, so each record is linked to its single most probable
Source 1 entity, or to none.

The pipeline has four stages:

1. **Retrieval.** A fine-tuned multilingual bi-encoder (e5-small, MIT) runs exact GPU
   nearest-neighbour search inside each country label, in both directions.
2. **Pruning.** A LightGBM model prunes the candidates using 50 string, number, address-structure
   and embedding features.
3. **Scoring.** A fine-tuned listwise cross-encoder and 29 "decoy" features score each surviving
   pair.
4. **Decision.** A final LightGBM model combines everything. A decision rule tuned directly for
   the challenge's macro F0.5 (either a threshold or expected-F0.5 top-k per entity) picks the
   matches.

What sets this apart from a plain blocking-plus-classifier setup:

- **A leak-free validation protocol** that places each decoy in the fold of the entity it
  competes with.
- **Features aimed at the dominant false-merge pattern**, "sibling" businesses a few doors down
  the same street.
- **Country-agnostic address structure** that generalises to France, which is absent from
  training.

---

## 2. Methodology

### 2.1 Problem Analysis

Findings from EDA on the training data (2.21M Source 1 entities, 5.03M Source 2 and 5.29M
Source 3 records):

| Finding | Number | Consequence |
|---|---|---|
| Each S2/S3 id appears in at most one ground-truth list | 7,638,365 pairs, 7,638,365 distinct ids | Assignment: each record goes to ≤ 1 entity |
| S2/S3 records that match nothing | ~2.68M (26%) | Needs an explicit "no match" option |
| Singleton S1 entities | 5.6% | An empty prediction earns 1.0; any false merge earns 0 |
| S1 entities sharing their exact name with another S1 | 38% ("Primary Care Group" ×253) | The name alone cannot identify an entity |
| True pairs whose country labels differ | 0 of 20,888 sampled | Blocking inside each country label is lossless |
| Names written in native Indic scripts | ~10% of S2, ~5% of S3 | Cross-script matching is required |
| Unmatched US S2 records that are "sibling decoys" | 57.6% (India 36.1%) | Main source of false merges (see below) |

**Sibling decoys.** These are different businesses whose name is the real one plus a word, and
whose address is a few doors up the same street. Example: `XMM Staffing Group, 10 Stagecoach Rd`
next to the real `XMM Staffing Corp, 8 Stagecoach Rd`. Measured on training labels:

| Signal | Decoys | True pairs |
|---|---|---|
| First house number equal | 0.9% (US) | 77% |
| A number moved to a nearby value | ~90% | 0.5–1.7% |
| "Group" or "Holdings" added to the name | ~18% | 0.00% |

True pairs carry their own number noise: zero-padding (`00955`), a dropped digit (`924` vs
`4924`), prefixes (`B3/303`), and in India an injected first number (`H.NO 53`) in 14% of pairs.
A number mismatch alone is therefore not a reliable reject rule; the model has to combine the
signals.

**Other noise patterns.**

- **Names:**
  - legal-suffix swaps
  - word reordering and duplicated tokens
  - injected accents and typos
  - junk affixes (`***`, `M/s`, `#51176`, `(ID: 47810)`)
  - domain forms (`servicespolytexagro.com`)
  - alias forms (`X dba Y`, `X fka Y`)
  - translation into native scripts
  - fully replaced trade names, which only the address can link
- **Addresses:**
  - upper-casing
  - street-type abbreviations
  - state abbreviations or native-script state names
  - reordered components
  - `NULL`/`N/A` placeholders
  - dropped components
  - alternate city names

**France**, which appears only in test (15% of test Source 1 entities), was analysed on the test
inputs without labels, using high-confidence rule-matched pairs:

| Source | Admin component on the S2/S3 side |
|---|---|
| S1 | always a région |
| S2/S3 | same région about ⅓ of the time, the département about ⅓, absent about ⅓ |

Whole-address similarity for French S2 therefore looks like training S3. We compare the street
component and the entity's finest locality instead. We also drop the source indicator from the
model, so French S2 records are not judged as out-of-distribution S2.

### 2.2 Solution Strategy

**Approach Type:** Blocking + learned re-ranking + assignment decoding (hybrid of dense
retrieval, gradient-boosted trees and a cross-encoder).

**Core Innovations:**

1. **Assignment decoding.** Each record is reduced to its best entity. The kept links are then
   chosen per entity to maximise macro F0.5, either by a threshold or by the expected-F0.5 top-k.
2. **Leak-free stacking and validation.**
   - Base models (bi-encoder, cross-encoder, extra-word statistics) train on folds 5–9.
   - The rankers train on folds 1–4, and the pruner's scores there are made out-of-fold.
   - Fold 0 is held out.
   - Unmatched records take the fold of the entity they compete with. Otherwise the decoys
     competing with validation entities are records the ranker was trained on, and false merges
     look rarer than they are.
3. **Decoy-aware features.**
   - moved house numbers with the signed distance
   - added legal words
   - extra-word log-odds learned from labels
   - cluster consistency among an entity's candidate records
4. **Country-agnostic address structure** (street component, finest locality) and a
   native-script→Latin dictionary learned only from training pairs.

---

## 3. Candidate Generation (Blocking)

- **Blocking keys used:**
  - **Country label** (open set) as a hard partition.
  - Inside each partition, exact **dense nearest-neighbour search** with the fine-tuned
    bi-encoder, run in both directions. Each S2/S3 record retrieves its top-10 Source 1
    entities, and each Source 1 entity retrieves its top-10 S2/S3 records. The union is kept.
  - The search is brute-force fp16 matrix multiplication on the GPU, run in blocks sized to the
    free memory, so there is no approximate-index loss.
- **Bi-encoder:**
  - `intfloat/multilingual-e5-small` (MIT, 118M parameters) with mean pooling.
  - Symmetric InfoNCE loss (τ = 0.05). Batches are drawn from a single country, and each pair
    carries one mined same-name hard negative.
  - Trained on 3M true pairs from folds 5–9.
  - It ranks the correct entity first for **97.96%** of held-out true pairs.
- **Second-stage filter (pruner):** LightGBM keeps at most 4 candidates per record with
  p ≥ 0.003. The pruned set is exactly what the final model scores and what
  `candidate_pairs.tsv` contains.
- **Candidate pairs generated (test):**

  | Set | Pairs | Per record |
  |---|---|---|
  | Blocking | 102,303,092 | ~10.3 |
  | Pruned (= `candidate_pairs.tsv`) | 8,702,689 | ~1.08 among records with any candidate |

- **Recall on held-out fold 0** (764,025 true pairs):
  - 99.73% after blocking (99.36% within each record's top-10)
  - 99.12% after pruning
- **How true matches were kept:**
  - bidirectional retrieval
  - exact search
  - hard-negative training, so the encoder uses the address when names collide
  - a pruning threshold chosen for recall

---

## 4. Matching Model

**Pair features, blocking stage (50):**

- **Name:**
  - rapidfuzz ratio, token-sort, token-set and partial ratio on canonical tokens
  - ratio, token-set and Jaro-Winkler on the core name (legal suffixes and honorifics removed;
    alias forms resolved to the real business)
  - space-free ratios for domain-style names
  - IDF-weighted word cosine and character-3-gram TF-IDF cosine
- **Address:**
  - fuzzy ratios, IDF-weighted and character-3-gram cosines over canonical tokens (street
    types, directions, French and Indian address words normalised)
  - a street-component-only similarity
  - whether the entity's two finest localities (learned from Source 1 frequencies) appear in
    the record, and how common the finest locality is
  - number overlap (TF-IDF, shared count, Jaccard, first-number equality)
- **Embedding and competition:**
  - bi-encoder cosine
  - rank of the pair in each direction
  - gaps to the record's and the entity's best candidate
  - margin over the record's runner-up
  - candidate counts
- **Record:**
  - name script
  - missing-address and domain flags
  - token counts
  - how common the entity's core name is in its country

**Decoy features on pruned pairs (29):**

- **House numbers:**
  - first-number equality, containment, gap and signed gap
  - set-level: a number moved to a nearby value (with its signed distance), shared numbers,
    the share of each side's numbers found on the other side
- **Extra words:** full-name words present on only one side (not fuzzy variants), with
  log-odds of "true pair vs wrong pair" learned on folds 5–9. The learned "different business"
  words include `holdings, group, pvt, public, east, west, north, south, metro, lakeside`.
- **Legal words:** legal words added or removed, a strong "group/holding" addition, and
  legal-form conflicts.
- **Ambiguity:** number of Source 1 entities sharing the exact full name.
- **Cluster:** how many of the entity's other candidate records share this record's house
  number versus the entity's own.

**Model type:**

1. **LightGBM pruner** (255 leaves, early stopping on fold 0; out-of-fold on folds 1–4).
2. **Cross-encoder:**
   - e5-small initialised from the fine-tuned bi-encoder
   - input `<S1 text> </s> <record text>`, mean-pooled linear head
   - listwise softmax over each record's candidates plus a learned "no match" slot
   - trained on folds 5–9
3. **LightGBM final ranker** on all of the above, plus the cross-encoder logit and its rank,
   margin and gap within the record's and the entity's candidate groups.

**Decision rule:**

- Each record keeps only its argmax entity.
- Two rules are compared on fold 0 with the exact metric (macro F0.5 per entity, singletons
  included):
  - a threshold sweep over 0.05–0.95
  - expected-F0.5 decoding: for each entity, enumerate every true/false labelling of its
    candidates, weighted by temperature-calibrated probabilities, and keep the top-k that
    maximises expected F0.5; k = 0 is the singleton choice
- The better rule is applied to test.

---

## 5. Results & Error Analysis

| Version | What changed | Validation macro F0.5 | Public LB |
|---|---|---|---|
| dense baseline | bi-encoder top-1 + cosine/margin threshold | — | 0.753 |
| v0 | LightGBM pruner, threshold | 0.9795 (older, optimistic protocol) | 0.972 |
| v1 | + cross-encoder + final ranker | 0.9906 (older protocol) | [TBD] |
| v2 | + honest folds, decoy features, address structure, French-safe normalisation, expected-F decoding | [TBD] | [TBD] |

- **v1 on validation:**
  - singletons 0.9965, non-singletons 0.9902
  - micro precision 0.9984, micro recall 0.9744
  - US 0.990, India 0.992
- **Common false positives (wrong merges):** sibling businesses on the same street (the name
  plus a qualifier or legal word, with the house number moved); franchises with identical
  names and missing addresses.
- **Common false negatives (missed matches):** records whose trade name was replaced and whose
  address is partial. Roughly 1.3% (US) / 1.9% (India) of true pairs carry no usable signal.

---

## 6. Conclusion

[TBD]

---

## Appendix

### A. Code Artefacts

`code/business_entity_resolution/`:

- `run.sh` — stage runner (`all1`, `all2`, or single stages)
- `src/textnorm.py` — normalisation rules
- `src/prepare.py` — loading, transliteration dictionary, address components, parquet tables
- `src/biencoder.py` — bi-encoder fine-tuning and embedding
- `src/retrieve.py` — exact blocked GPU kNN
- `src/refold.py` — decoy fold assignment
- `src/features.py` — pair features
- `src/ranker.py` — LightGBM pruner and final ranker
- `src/features2.py` — decoy features
- `src/crossencoder.py` — cross-encoder
- `src/decide.py` — assignment, decision-rule tuning, output writer
- `src/analyze.py`, `src/silver_check.py` — diagnostics only

Reproduce with `bash run.sh all1 && bash run.sh all2`. The README gives the exact environment
and runtime per stage.

**Compliance:**

- Only the provided data is used: no external APIs, databases, geocoding or look-ups.
- The only pretrained model is `intfloat/multilingual-e5-small` (MIT, 118M parameters), well
  under the 8B limit.
- The normalisation rules in `textnorm.py` encode generic spelling conventions only (Rd/Road,
  Ltd/Limited, R./Rue, N°, bis/ter, dba), not business or place data.
- Test inputs are used only in unsupervised ways that need no labels: IDF statistics, locality
  frequencies among Source 1 records, and the silver diagnostic, which is never used for
  training or tuning.

### B. Additional Results

[TBD]
