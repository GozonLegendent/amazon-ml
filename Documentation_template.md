# ML Challenge 2026: Business Entity Resolution Solution Template

**Team Name:** [Your Team Name]  
**Team Members:** [List all team members]  
**Submission Date:** [Date]

---

## 1. Executive Summary

We treat entity resolution as **assignment**. In the training labels every Source 2/3 record
matches at most one Source 1 entity, so each record is assigned to its single most probable
Source 1 entity, or to none. The pipeline has five stages:

1. A fine-tuned multilingual bi-encoder (e5-small, MIT) runs exact GPU nearest-neighbour search
   inside each country label.
2. A LightGBM pruner scores 42 string, number and embedding features.
3. A fine-tuned cross-encoder reads each surviving pair jointly.
4. A final LightGBM combines everything with "competition" features, i.e. how a pair ranks
   against the other candidates for the same record and the same entity.
5. One threshold, tuned directly for the challenge's macro F0.5 on a held-out fold, decides
   the matches.

Native-script names and addresses (Hindi, Tamil, Telugu, Kannada, Bengali, Gujarati, Malayalam,
Odia, Punjabi) are handled two ways: the multilingual transformers read them directly, and a
native-script→Latin token dictionary learned only from the training pairs normalises them for
the string features.

---

## 2. Methodology

### 2.1 Problem Analysis

Findings from EDA on the training data (2.21M Source 1 entities, 5.03M Source 2 and 5.29M
Source 3 records):

| Finding | Number | Consequence |
|---|---|---|
| Each S2/S3 id appears in at most one ground-truth list | 7,638,365 pairs, 7,638,365 distinct ids | Model the task as "assign each record to ≤ 1 entity" |
| S2/S3 records that match nothing (distractors) | ~2.68M (26%) | Needs an explicit "no match" option |
| Singleton S1 entities | 5.6% | Predicting an empty list earns 1.0; any false merge earns 0 |
| Matches per S1 entity | 0–11, mode 3 | Many records per entity, from both sources |
| S1 entities sharing their exact name with another S1 | 38% (e.g. "Primary Care Group" ×253) | The name alone cannot identify an entity; the address must decide |
| True pairs whose country labels differ | 0 of 20,888 sampled | Blocking inside each country label is lossless |
| Names written in native Indic scripts | ~10% of S2, ~5% of S3 | Cross-script matching is required |
| Records with an empty address | ~3% of S2/S3 | Name-only matching must still work |

Noise patterns observed (all generated on top of a clean S1 record):

- **Names:**
  - legal-suffix swaps (Pvt/Private, Ltd/Limited, LLC, SARL, S.A.S.)
  - word reordering ("Hotel Logistics Limited Private") and duplicated tokens ("Caressa's Caressa's")
  - injected accents (Í, á), character typos, and leet-style substitutions ("Roya1")
  - junk prefixes and suffixes (`***`, `<<`, `M/s`, `Dr`, `#51176`, `(ID: 47810)`, `- 1281460800`)
  - domain forms (`servicespolytexagro.com`)
  - transliteration into native scripts
  - occasionally a fully replaced trade name ("Evoorbi"), where only the address links the records
- **Addresses:**
  - upper-casing
  - street-type abbreviations (Rd, St, Ave, R., BD, IMP)
  - state abbreviations or native-script state names (TX↔Texas, MH↔महाराष्ट्र)
  - reordered components
  - `NULL`/`N/A` placeholders
  - dropped components
  - altered house numbers (`B3/303` vs `303`, `0042` vs `42`)
  - alternate city names (Ogden vs Spencerport)
  - region vs département for France (Nouvelle-Aquitaine vs Gironde)

The test set adds France (about 15% of test Source 1 entities), which never appears in training.
No part of the pipeline uses the country value as a feature; it only groups records for blocking.
Country is therefore an open set of labels, and France gets the same treatment as every other
country.

### 2.2 Solution Strategy

**Approach Type:** Blocking + learned re-ranking + assignment (hybrid: dense retrieval,
gradient-boosted trees, cross-encoder).

**Core Innovation:**

1. **Assignment decoding.** The ≤ 1-entity-per-record property is decoded directly: each record
   picks its best candidate and must clear a tuned threshold. This removes a whole class of
   false merges that independent pairwise thresholding produces.
2. **Leak-free stacking.** Base models (bi-encoder, cross-encoder) train on folds 5–9. The
   rankers train on folds 1–4, where base-model scores are out-of-sample, just as on test.
   Fold 0 is held out.
3. **Cross-script normalisation learned from the data.** A token dictionary is built by
   aligning native-script tokens with the Latin tokens of their matched Source 1 records; no
   external transliteration data is used.
4. **A listwise cross-encoder with a learned "no match" slot**, so the model learns both which
   entity wins and whether any entity matches.

---

## 3. Candidate Generation (Blocking)

- **Blocking keys used:**
  - **Country label** (open set) as a hard partition.
  - Inside each partition: exact **dense nearest-neighbour search** with the fine-tuned
    bi-encoder, run in both directions. Each S2/S3 record retrieves its top-10 Source 1
    entities, and each Source 1 entity retrieves its top-10 S2/S3 records. The union is kept.
  - The search is brute-force fp16 matrix multiplication on the GPU, run in blocks that fit the
    free memory, so there is no approximate-index recall loss.
- **Bi-encoder training:** `intfloat/multilingual-e5-small` (MIT, 118M parameters), mean pooling,
  symmetric InfoNCE (τ = 0.05). Batches are drawn from a single country, and each pair carries
  one mined hard negative: another Source 1 record with the same core name. It is trained on
  3M true pairs from folds 5–9 with gradient checkpointing.
- **Second-stage filter (pruner):** a LightGBM model on the 42 cheap features keeps at most 4
  candidates per record with p ≥ 0.003. The pruned set is exactly what the final model scores
  and what `candidate_pairs.tsv` contains.
- **Candidate pairs generated:** [TBD — blocking total and pruned total for test]
- **How true matches were kept:**
  - bidirectional retrieval, so an entity whose records crowd each other still keeps its own
    candidates
  - exact (not approximate) search
  - hard negatives, which teach the encoder to use the address when names collide
  - a pruner threshold chosen for recall
  - blocking recall is measured on the held-out fold before and after pruning:
    [TBD recall@10 / recall after pruning]

---

## 4. Matching Model

**Features used (42 pair features + 15 second-stage features):**

- **Name features:**
  - rapidfuzz ratio, token-sort, token-set and partial ratio on canonical tokens
  - ratio, token-set and Jaro-Winkler on the core name (legal suffixes, honorifics and
    stop-words removed)
  - space-free ratio and partial ratio, for domain-style names
  - IDF-weighted word cosine and character-3-gram TF-IDF cosine
- **Address features:**
  - rapidfuzz ratio, token-set, token-sort and partial ratio on canonical address tokens
    (street types, directions, French street types and Indian address words mapped to one form)
  - IDF-weighted token cosine and character-3-gram cosine
  - numbers: TF-IDF cosine over house / unit / PIN numbers, shared-number count, Jaccard,
    first-number equality, number counts
- **Embedding and competition features:**
  - bi-encoder cosine
  - rank of the pair in each direction
  - gap to the record's best candidate and to the entity's best candidate
  - margin over the record's runner-up
  - candidate counts
- **Record features:** script of the record name, missing-address flag, domain-name flag,
  source (S2/S3), token counts, how common the Source 1 core name is inside its country.
- **Second stage:**
  - cross-encoder logit
  - pruner probability
  - rank, margin and gap of both scores within the record's group and the entity's group
  - number of the entity's candidates with a positive cross-encoder logit

**Model type:**

1. LightGBM pruner (binary, 255 leaves).
2. Cross-encoder: e5-small initialised from the fine-tuned bi-encoder, input
   `<S1 text> </s> <record text>`, mean-pooled linear head, listwise softmax over each record's
   candidates plus a learned "no match" logit.
3. LightGBM final ranker on everything above.

**Threshold selection method:**

- Each record is assigned to its argmax entity when p ≥ t.
- t is swept over 0.05–0.95 and chosen to maximise the exact challenge metric: macro F0.5 per
  Source 1 entity, singletons included, over the held-out fold-0 entities.

---

## 5. Results & Error Analysis

- **F_0.5 Score (macro, held-out fold 0):** [TBD]
- **Singletons / non-singletons:** [TBD]
- **Micro precision / recall:** [TBD]
- **Per country:** [TBD]
- **Public leaderboard:** [TBD]
- **Common false positives (wrong merges):** [TBD from error analysis]
- **Common false negatives (missed matches):** [TBD from error analysis]

---

## 6. Conclusion

[TBD]

---

## Appendix

### A. Code Artefacts

`code/business_entity_resolution/`:

- `run.sh` — stage runner (`all1`, `all2`, or single stages)
- `src/textnorm.py` — normalisation rules
- `src/prepare.py` — loading, transliteration dictionary, parquet tables
- `src/biencoder.py` — bi-encoder fine-tuning and embedding
- `src/retrieve.py` — exact blocked GPU kNN
- `src/features.py` — pair features
- `src/ranker.py` — LightGBM pruner and final ranker
- `src/crossencoder.py` — cross-encoder
- `src/decide.py` — assignment, threshold tuning, output writer

Reproduce with `bash run.sh all1 && bash run.sh all2`. The README gives the exact environment
and runtime per stage.

**Compliance:**

- Only the provided data is used: no external APIs, databases, geocoding or look-ups.
- The only pretrained model is `intfloat/multilingual-e5-small` (MIT, 118M parameters), well
  under the 8B limit.
- The abbreviation maps in `textnorm.py` encode generic spelling conventions only (Rd/Road,
  Ltd/Limited, R./Rue), not any business or place data.

### B. Additional Results

[TBD]
