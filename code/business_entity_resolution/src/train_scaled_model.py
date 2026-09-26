"""
Scaled Training Script for Business Entity Resolution.
Trains LightGBM + CatBoost Dual Ensemble on 20,000+ Source 1 reference entities
with active hard negative mining and 20 SIMD domain/regional features.
"""

import argparse
import os
import sys
import time
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from preprocess import preprocess_dataframe
from blocking import ScalableBlocker
from features import FeatureExtractor
from model import EntityMatchingModel, save_matching_results
from evaluate import evaluate


def run_scaled_training(
    train_dir: str = "student_resource/dataset/train",
    val_dir: str = "student_resource/dataset/val",
    output_model_path: str = "code/business_entity_resolution/src/matching_model.pkl",
    n_s1_train: int = 20000,
    max_candidates: int = 50,
):
    print("=" * 65)
    print("     SCALED ENTITY RESOLUTION ENSEMBLE TRAINING (20,000 S1)   ")
    print("=" * 65)
    t0 = time.time()

    # 1. Ingest S1 Training Entities & Ground Truth (from row 120,000 to avoid any val overlap)
    skip_rows = 120000
    print(f"Sampling {n_s1_train:,} training reference entities from {train_dir} (offset {skip_rows:,})...")
    s1_path = os.path.join(train_dir, "train_source1.tsv")
    gt_path = os.path.join(train_dir, "train_ground_truth.tsv")

    # Read S1 sample
    df_s1_sample = pd.read_csv(s1_path, sep="\t", skiprows=range(1, skip_rows), nrows=n_s1_train)
    s1_ids_set = set(df_s1_sample["entity_id"])

    # Load GT mapping in 3.5 seconds
    print("Loading ground truth mapping...")
    gt_df = pd.read_csv(gt_path, sep="\t")
    gt_dict = dict(zip(gt_df["source1_entity_id"], gt_df["matched_entity_ids"]))
    del gt_df

    gt_map_train = {}
    needed_s2 = set()
    needed_s3 = set()

    for sid in df_s1_sample["entity_id"]:
        mstr = str(gt_dict.get(sid, "")) if pd.notna(gt_dict.get(sid, "")) else ""
        if mstr and mstr != "nan":
            m_ids = set(m.strip() for m in mstr.split(",") if m.strip())
            gt_map_train[sid] = m_ids
            for mid in m_ids:
                if mid.startswith("S2-"):
                    needed_s2.add(mid)
                elif mid.startswith("S3-"):
                    needed_s3.add(mid)
        else:
            gt_map_train[sid] = set()

    del gt_dict

    n_pos_links = sum(len(v) for v in gt_map_train.values())
    print(f"Sampled {len(df_s1_sample):,} S1 entities ({n_pos_links:,} true matching links).")
    print(f"True positive targets: {len(needed_s2):,} from S2, {len(needed_s3):,} from S3.")

    # 2. Ingest Candidate Pool Records (True matches + Mined Distractors)
    print("Ingesting candidate pool records with guaranteed true matches + distractors...")
    pool_records = []
    distractors_per_source = max(100000, int(len(needed_s2) * 2.5))

    # Stream S2
    s2_path = os.path.join(train_dir, "train_source2.tsv")
    s2_pos_collected = 0
    s2_dist_collected = 0
    for chunk in pd.read_csv(s2_path, sep="\t", chunksize=150000):
        pos_chunk = chunk[chunk["entity_id"].isin(needed_s2)]
        if not pos_chunk.empty:
            pool_records.append(pos_chunk)
            s2_pos_collected += len(pos_chunk)
        if s2_dist_collected < distractors_per_source:
            neg_chunk = chunk[~chunk["entity_id"].isin(needed_s2)]
            sample_size = min(len(neg_chunk), 20000)
            pool_records.append(neg_chunk.sample(sample_size, random_state=42))
            s2_dist_collected += sample_size
        if s2_pos_collected >= len(needed_s2) and s2_dist_collected >= distractors_per_source:
            break

    # Stream S3
    s3_path = os.path.join(train_dir, "train_source3.tsv")
    s3_pos_collected = 0
    s3_dist_collected = 0
    for chunk in pd.read_csv(s3_path, sep="\t", chunksize=150000):
        pos_chunk = chunk[chunk["entity_id"].isin(needed_s3)]
        if not pos_chunk.empty:
            pool_records.append(pos_chunk)
            s3_pos_collected += len(pos_chunk)
        if s3_dist_collected < distractors_per_source:
            neg_chunk = chunk[~chunk["entity_id"].isin(needed_s3)]
            sample_size = min(len(neg_chunk), 20000)
            pool_records.append(neg_chunk.sample(sample_size, random_state=42))
            s3_dist_collected += sample_size
        if s3_pos_collected >= len(needed_s3) and s3_dist_collected >= distractors_per_source:
            break

    df_pool_raw = pd.concat(pool_records, ignore_index=True).drop_duplicates(subset=["entity_id"])
    print(f"Candidate pool compiled: {len(df_pool_raw):,} records (S2: {s2_pos_collected:,} pos, S3: {s3_pos_collected:,} pos).")
    del pool_records

    # 3. Preprocess
    print("Preprocessing text and addresses...")
    df_s1_clean = preprocess_dataframe(df_s1_sample)
    df_pool_clean = preprocess_dataframe(df_pool_raw)
    del df_pool_raw

    # 4. Generate Candidates & Mine Hard Negatives
    print(f"Building Phase 1 Inverted Blocking Index (max_candidates={max_candidates})...")
    blocker = ScalableBlocker(max_candidates_per_entity=max_candidates)
    blocker.fit_pool(df_pool_clean)

    print("Generating candidate pairs and mining hard negatives...")
    cand_map_train = blocker.generate_candidates(df_s1_clean)

    # 5. Extract 20 SIMD Features
    extractor = FeatureExtractor()
    print("Extracting 20 SIMD features for candidate pairs...")
    X_train, y_train, _ = extractor.extract_features_for_candidates(
        candidate_map=cand_map_train,
        df_s1=df_s1_clean,
        df_pool=df_pool_clean,
        gt_map=gt_map_train,
    )

    del df_s1_clean, df_pool_clean, blocker, cand_map_train

    print("\nLoading and preprocessing validation dataset with new OCR/URL/Indic features...")
    raw_s1_path = os.path.join(val_dir, "val_source1.tsv")
    if os.path.exists(raw_s1_path):
        val_s1 = preprocess_dataframe(pd.read_csv(raw_s1_path, sep="\t"))
        val_s2 = preprocess_dataframe(pd.read_csv(os.path.join(val_dir, "val_source2.tsv"), sep="\t"))
        val_s3 = preprocess_dataframe(pd.read_csv(os.path.join(val_dir, "val_source3.tsv"), sep="\t"))
    else:
        val_s1 = pd.read_csv(os.path.join(val_dir, "val_source1_clean.tsv"), sep="\t")
        val_s2 = pd.read_csv(os.path.join(val_dir, "val_source2_clean.tsv"), sep="\t")
        val_s3 = pd.read_csv(os.path.join(val_dir, "val_source3_clean.tsv"), sep="\t")
    val_pool = pd.concat([val_s2, val_s3], ignore_index=True)

    val_gt = pd.read_csv(os.path.join(val_dir, "val_ground_truth.tsv"), sep="\t")

    gt_map_val = {
        r["source1_entity_id"]: set(r["matched_entity_ids"].split(",")) if pd.notna(r["matched_entity_ids"]) and r["matched_entity_ids"] else set()
        for _, r in val_gt.iterrows()
    }

    # Val split (1,559 entities)
    np.random.seed(42)
    val_indices = np.random.permutation(len(val_s1))[3000:]
    s1_val = val_s1.iloc[val_indices].reset_index(drop=True)
    val_gt_subset = {sid: gt_map_val[sid] for sid in s1_val["entity_id"]}

    val_blocker = ScalableBlocker(max_candidates_per_entity=max_candidates)
    val_blocker.fit_pool(val_pool)
    cand_map_val = val_blocker.generate_candidates(s1_val)

    X_val, y_val, pairs_val = extractor.extract_features_for_candidates(
        candidate_map=cand_map_val,
        df_s1=s1_val,
        df_pool=val_pool,
        gt_map=val_gt_subset,
    )

    # 7. Train Tri-Model Ensemble (LightGBM + CatBoost + XGBoost)
    print("\nTraining Tri-Model Gradient-Boosted Ensemble (LightGBM + CatBoost + XGBoost)...")
    model = EntityMatchingModel(
        n_estimators=400,
        learning_rate=0.035,
        max_depth=6,
        subsample=0.85,
        colsample_bytree=0.85,
    )
    model.train(X_train, y_train, extractor.feature_names, X_val, y_val)

    # 8. Optimize Decision Threshold
    val_probs = model.predict_probabilities(X_val)
    s1_val_ids = s1_val["entity_id"].tolist()
    best_thresh = model.optimize_threshold(val_probs, pairs_val, val_gt_subset, s1_val_ids)

    # 9. Validation Audit (Phase 1: Dual Ensemble)
    pred_matches = model.predict_matches(val_probs, pairs_val, s1_val_ids, threshold=best_thresh)
    val_pred_path = "student_resource/output/val_matching_results.tsv"
    val_gt_subpath = "student_resource/output/val_gt_subset.tsv"
    save_matching_results(pred_matches, val_pred_path)

    pd.DataFrame([
        {"source1_entity_id": sid, "matched_entity_ids": ",".join(val_gt_subset[sid])}
        for sid in s1_val_ids
    ]).to_csv(val_gt_subpath, sep="\t", index=False)

    print("\n" + "=" * 65)
    print("   PHASE 1 EVALUATION AUDIT: DUAL ENSEMBLE (LIGHTGBM + CATBOOST)   ")
    print("=" * 65)
    metrics_p1 = evaluate(val_gt_subpath, val_pred_path)

    # 10. Phase 2: Graph Transitivity & Cross-Source Agreement Post-Processing
    print("\n" + "=" * 65)
    print("   PHASE 2 EVALUATION AUDIT: GRAPH TRANSITIVITY POST-PROCESSING    ")
    print("=" * 65)
    from collections import defaultdict
    from postprocess import apply_cross_source_transitivity

    candidate_scores = defaultdict(list)
    for (sid, pid), p in zip(pairs_val, val_probs):
        candidate_scores[sid].append((pid, float(p)))
    for sid in s1_val_ids:
        if sid not in candidate_scores:
            candidate_scores[sid] = []

    # Fast pool dictionary lookup for similarity
    val_pool_records = {
        str(r["entity_id"]): (
            str(r["core_name"]) if pd.notna(r.get("core_name")) else "",
            str(r["clean_address"]) if pd.notna(r.get("clean_address")) else "",
        )
        for _, r in val_pool.iterrows()
    }

    post_matches = apply_cross_source_transitivity(
        candidate_scores=candidate_scores,
        pool_records=val_pool_records,
        high_conf_thresh=best_thresh,
        recovery_thresh=max(0.60, best_thresh - 0.15),
    )

    val_post_pred_path = "student_resource/output/val_matching_results_post.tsv"
    save_matching_results(post_matches, val_post_pred_path)
    metrics_p2 = evaluate(val_gt_subpath, val_post_pred_path)

    # 11. Persist Model
    model.save_model(output_model_path)
    print(f"\nScaled Training & Multi-Phase Audit Completed in {time.time()-t0:.2f}s!")
    return model, metrics_p1, metrics_p2


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Scaled Entity Matching Model Training")
    parser.add_argument("--train-dir", default="student_resource/dataset/train", help="Train dir")
    parser.add_argument("--val-dir", default="student_resource/dataset/val", help="Val dir")
    parser.add_argument("--output-model", default="code/business_entity_resolution/src/matching_model.pkl", help="Model output")
    parser.add_argument("--n-s1", type=int, default=15000, help="Number of S1 training entities")
    parser.add_argument("--max-candidates", type=int, default=75, help="Max candidates per entity")

    args = parser.parse_args()

    run_scaled_training(
        train_dir=args.train_dir,
        val_dir=args.val_dir,
        output_model_path=args.output_model,
        n_s1_train=args.n_s1,
        max_candidates=args.max_candidates,
    )
