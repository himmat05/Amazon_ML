# ML Challenge 2026: Business Entity Resolution Solution

**Team Name:** Team EntityRes  
**Submission Date:** 2026-09-26  

---

## 1. Executive Summary
We developed an end-to-end, high-recall, scalable Machine Learning entity resolution architecture designed to match 1.73 million reference business entities against a multi-million candidate pool across the US, India, and France. Our solution couples a multi-angle inverted blocking engine (featuring character 3-gram MinHash LSH, phonetic Soundex, and OCR glyph folding) with a Tri-Model Gradient-Boosted Ensemble (LightGBM + CatBoost + XGBoost) trained on 1.87 million hard-mined candidate pairs, achieving **93.10% Macro $F_{0.5}$** and **98.33% Pairwise Precision** with graph transitivity post-processing.

---

## 2. Methodology

### 2.1 Problem Analysis
During exploratory data analysis across the three sources (Source 1 reference vs. Source 2 and Source 3 candidate pool), we identified four primary noise modalities:
1. **OCR / Typographical Glyph Confusion:** Visual substitution errors in digitized documents (e.g. lowercase `l` replacing capital `I`, digit `0` replacing letter `O`, digit `1` replacing `I`).
2. **Web Domain & URL Handles:** Source 2/3 entities frequently formatted as web domains or concatenated handles (e.g. `hhcharities.com` vs `Hugo Heritage Charities Corp`).
3. **Cross-Script Indic Transliteration:** Indian business names in Source 2/3 written in regional scripts (Devanagari, Tamil, Bengali) paired with Latin English representations in Source 1.
4. **Token Permutations & Missing Words:** Swapped word order (*"Williams Continental"* vs *"Continental Williams"*), dropped core words, and varying legal entity designations (*"Pvt Ltd"*, *"LLC"*, *"SAS"*, *"SARL"*).

### 2.2 Solution Strategy
**Approach Type:** Hybrid Multi-Angle Inverted Index Blocking + Tri-Model Decision Forest + Multi-Source Graph Transitivity.  
**Core Innovations:**
- **Zero-Dependency Multilingual Normalization:** Integrated `anyascii` script transliteration to project non-Latin scripts into standard Latin ASCII, alongside protocol and domain extension stripping (`.com`, `.co.in`, `.fr`).
- **Character 3-Gram MinHash LSH Invariance:** Character trigram sorting and extreme hashing to guarantee index collisions for misspelled or transposed names.
- **Active Hard Negative Mining:** Scaled training on 25,000 S1 records producing 1.87 million candidate pairs with a sharp 1:22 positive-to-negative boundary.
- **Tri-Model Ensemble with Soft Blending:** LightGBM (400 trees) + CatBoost (350 trees) + XGBoost (350 trees) weighted averaging.
- **Cross-Source Graph Transitivity:** Connected-components propagation ($S_1 \leftrightarrow S_2 \wedge S_2 \approx S_3 \implies S_1 \leftrightarrow S_3$).

---

## 3. Candidate Generation (Blocking)
To reduce the quadratic comparison space ($\approx 1.73 \times 10^6 \times 10^7 \approx 1.73 \times 10^{13}$ pairs) to a tight candidate set, we constructed a country-partitioned multi-key inverted index:

- **Blocking Keys Used:**
  1. `_P3:` / `_P4:` Core name 3-gram and 4-gram prefixes.
  2. `_W:` Significant non-stopword business name tokens.
  3. `_SX:` Soundex phonetic encoding for spelling and transliteration invariance.
  4. `_SWP:` Sorted word-pair prefix combinations (transposition invariance).
  5. `_ACR:` Bidirectional acronym and initialism extraction (e.g. `KFC` $\leftrightarrow$ `Kentucky Fried Chicken`).
  6. `_ST:` Street number paired with primary street name token.
  7. `_POST:` Postal / PIN code combined with name initial.
  8. `_AW:` Rare distinctive landmark / locality address tokens (len $\ge 6$).
  9. `_NUM:` Building number anchor paired with business initial.
  10. `_OCR3:` OCR-folded 3-character prefixes (maps `l` $\to$ `i`, `0` $\to$ `o`, `w` $\to$ `v`).
  11. `_NPOST:` / `_NNUM:` Rare distinctive name tokens paired with postal code / building number.
  12. `_LSH:` Character 3-Gram MinHash locality-sensitive hashing keys.
- **Candidate Pool & Depth:** Index capacity configured to max 75 candidates per entity.
- **Joint Scoring:** Candidate retrieval uses a joint objective: $\text{Score} = \text{sim}_{\text{name}} + 0.4 \cdot \text{sim}_{\text{addr}} + 12 \cdot \text{key\_freq}$.
- **Candidate Recall:** Captured **92.00%** ($4,843 / 5,264$) of true positive links in the held-out benchmark.

---

## 4. Matching Model

### Features Used (20 SIMD Pairwise Features)
1. **String Distance Metrics:** Levenshtein distance ratio, Jaro-Winkler similarity, Token Sort ratio, Token Set ratio, Partial Ratio.
2. **Length & Structure:** Absolute character length difference, length ratio.
3. **Phonetic & Morphological:** Soundex exact match, Acronym / initialism match, Canonical legal suffix match.
4. **Address & Geography:** Address token Jaccard similarity, Address token overlap count, Address token sort ratio, Rare address token overlap.
5. **Numerical Consistency:** Street building number exact match, Numerical discrepancy penalty (penalizes conflicting street/suite numbers), Postal / PIN code prefix match.
6. **Full Context:** Concatenated name + address token set ratio, Source indicator (`is_s2`).

### Model Architecture
- **Model 1: LightGBM Classifier:** 400 histogram-based trees, learning rate $0.035$, max depth $6$, subsample $0.85$.
- **Model 2: CatBoost Classifier:** 350 symmetric oblivious trees, learning rate $0.05$, max depth $6$.
- **Model 3: XGBoost Classifier:** 350 histogram trees, learning rate $0.05$, max depth $6$, subsample $0.85$.
- **Ensemble Blending:**
  $$P_{\text{final}} = 0.40 \cdot P_{\text{LightGBM}} + 0.35 \cdot P_{\text{CatBoost}} + 0.25 \cdot P_{\text{XGBoost}}$$
- **Threshold Selection:** Evaluated across 26 threshold values specifically optimizing the competition Macro $F_{0.5}$ objective. Optimal threshold calibrated at $\tau = 0.78$.

---

## 5. Results & Error Analysis

### Official Evaluation Benchmark (Held-Out Validation Set)
- **Macro $F_{0.5}$ Score:** **0.9310 (93.10%)**
- **Pairwise Precision:** **98.33%** ($4,598$ correct out of $4,676$ predicted links — only $78$ false positives).
- **Pairwise Recall:** **87.35%** ($4,598$ true matches linked out of $5,264$ ground truth targets).
- **Singleton Accuracy ($0$-match):** **93.41%** (Properly preserving singleton penalty credit).
- **Non-Singleton $F_{0.5}$:** **0.9308**

### Error Analysis
- **False Positives:** Mainly multi-tenant commercial complexes or retail plazas where distinct businesses share identical street addresses and have generic names (e.g. *"Salon"* vs *"Boutique"*).
- **False Negatives:** Complex acronyms where the company name was completely rephrased or transliterated with non-standard regional synonyms not captured by phonetic tables.

---

## 6. Conclusion
Our hybrid approach resolves business entities at enterprise scale without relying on any external APIs or lookups. Combining multi-angle MinHash LSH blocking, zero-dependency multilingual script transliteration, a 20-feature SIMD matrix, and a tri-model gradient-boosted ensemble produced an official benchmark score of **$93.10\%$ Macro $F_{0.5}$** with **$98.33\%$ Precision**, delivering a robust, fully validated submission for the Amazon ML Challenge 2026.

---

## Appendix

### A. Code Artefacts
All code is organized under `code/business_entity_resolution/`:
- `src/preprocess.py`: Normalization, Indic transliteration, URL & OCR folding.
- `src/blocking.py`: Scalable candidate generation and inverted indexing.
- `src/features.py`: 20-feature pairwise SIMD matrix extractor.
- `src/model.py`: Tri-model ensemble and threshold optimization.
- `src/postprocess.py`: Graph transitivity and equivalence linking.
- `src/pipeline.py`: End-to-end streaming test runner.
- `src/matching_model.pkl`: Serialized trained ensemble model artifact.
- `README.md` & `requirements.txt`: Environment and reproduction scripts.

### B. Hardware & Runtime
- **Inference Runtime:** Micro-batched country streaming processes $\sim 260$ entities/sec, completing full 1.73M entity resolution within RAM limits.
- **Model Parameters:** Tree ensemble total footprint $< 5\text{ MB}$, well within the 8 Billion parameter ceiling.
