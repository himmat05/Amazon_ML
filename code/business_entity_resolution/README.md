# Business Entity Resolution Pipeline

This repository contains the complete, self-contained end-to-end Machine Learning pipeline for resolving business entities across heterogeneous and noisy data sources (Source 1 reference, Source 2, and Source 3) for the Amazon ML Challenge 2026.

## Directory Structure
```
code/business_entity_resolution/
├── src/
│   ├── metric.py            # Official Macro F_0.5 evaluation metric with singleton scoring
│   ├── preprocess.py        # Text & address normalization (Indic transliteration, URL & OCR folding)
│   ├── blocking.py          # Scalable candidate generation (MinHash LSH, Soundex, acronyms, address tokens)
│   ├── features.py          # 20 SIMD pairwise similarity feature extractors
│   ├── model.py             # Tri-Model Ensemble (LightGBM + CatBoost + XGBoost) and threshold optimizer
│   ├── postprocess.py       # Graph transitivity and cross-source equivalence resolver
│   ├── train_scaled_model.py# Scaled training script with active hard negative mining
│   ├── pipeline.py          # End-to-end streaming inference pipeline
│   ├── create_val_split.py  # Local validation split generator
│   └── evaluate.py          # Benchmark evaluation script
├── requirements.txt         # Pinned python dependencies
└── README.md                # Execution and reproduction guide
```

## Setup & Reproduction

### 1. Environment Installation
```bash
pip install -r requirements.txt
```

### 2. End-to-End Test Inference
To generate the final competition submission files (`matching_results.tsv` and `candidate_pairs.tsv`) in the `output/` directory:
```bash
python src/pipeline.py --test-dir ../../student_resource/dataset/test --output-dir ../../output
```

### 3. Model Training from Scratch
To retrain the Tri-Model Ensemble on 25,000+ reference entities with active hard negative mining:
```bash
python src/train_scaled_model.py --train-dir ../../student_resource/dataset/train --val-dir ../../student_resource/dataset/val --n-s1 25000 --max-candidates 75
```

### 4. Output Validation
To verify schema and competition rule compliance:
```bash
python ../../student_resource/utils/validate_submission.py \
    --matching ../../output/matching_results.tsv \
    --candidate ../../output/candidate_pairs.tsv \
    --test-dir ../../student_resource/dataset/test
```
