"""
Create a representative local validation split from train data.
Selects N Source 1 entities (preserving country & singleton distributions),
retrieves all their true positive matches from S2 and S3, and samples
realistic distractor pools to simulate the full search space.
"""

import os
import random
import pandas as pd

def create_validation_split(
    train_dir: str = "student_resource/dataset/train",
    val_out_dir: str = "student_resource/dataset/val",
    n_s1_samples: int = 25000,
    distractor_ratio: int = 3, # extra non-matching records to include in candidate pool
    seed: int = 42,
):
    random.seed(seed)
    os.makedirs(val_out_dir, exist_ok=True)

    print(f"Loading first 100,000 rows of train_source1 and ground truth for sampling...")
    df_s1 = pd.read_csv(os.path.join(train_dir, "train_source1.tsv"), sep="\t", nrows=100000)
    df_gt = pd.read_csv(os.path.join(train_dir, "train_ground_truth.tsv"), sep="\t", nrows=100000)

    # Merge on source1_entity_id
    merged = df_s1.merge(df_gt, left_on="entity_id", right_on="source1_entity_id")
    
    # Stratified sample by country if possible
    print(f"Sampling {n_s1_samples} Source 1 entities...")
    val_s1 = merged.sample(n=min(n_s1_samples, len(merged)), random_state=seed).reset_index(drop=True)

    # Collect all needed S2 and S3 IDs
    needed_s2 = set()
    needed_s3 = set()

    val_gt_rows = []
    for _, row in val_s1.iterrows():
        s1_id = row["entity_id"]
        matched_str = str(row["matched_entity_ids"]) if pd.notna(row["matched_entity_ids"]) else ""
        val_gt_rows.append({"source1_entity_id": s1_id, "matched_entity_ids": matched_str})
        
        if matched_str:
            ids = [i.strip() for i in matched_str.split(",") if i.strip()]
            for mid in ids:
                if mid.startswith("S2-"):
                    needed_s2.add(mid)
                elif mid.startswith("S3-"):
                    needed_s3.add(mid)

    val_gt_df = pd.DataFrame(val_gt_rows)
    val_s1_df = val_s1[["entity_id", "business_name", "business_address", "country"]]

    print(f"Validation set has {len(val_s1_df)} S1 entities.")
    print(f"True positive S2 IDs needed: {len(needed_s2)}, S3 IDs needed: {len(needed_s3)}")

    # Save val_source1 and val_ground_truth
    val_s1_path = os.path.join(val_out_dir, "val_source1.tsv")
    val_gt_path = os.path.join(val_out_dir, "val_ground_truth.tsv")
    val_s1_df.to_csv(val_s1_path, sep="\t", index=False)
    val_gt_df.to_csv(val_gt_path, sep="\t", index=False)
    print(f"Saved {val_s1_path} and {val_gt_path}")

    # Extract matching and distractor S2 records
    print("Reading S2 records to collect true matches and distractors...")
    s2_rows = []
    s2_distractors = []
    max_distractors = len(needed_s2) * distractor_ratio

    for chunk in pd.read_csv(os.path.join(train_dir, "train_source2.tsv"), sep="\t", chunksize=100000):
        # find matching
        matched_chunk = chunk[chunk["entity_id"].isin(needed_s2)]
        if not matched_chunk.empty:
            s2_rows.append(matched_chunk)
        
        # sample distractors
        if len(s2_distractors) < max_distractors:
            unmatched_chunk = chunk[~chunk["entity_id"].isin(needed_s2)]
            s2_distractors.append(unmatched_chunk.sample(min(len(unmatched_chunk), 5000), random_state=seed))
        
        # Check if we have collected all true positives
        curr_collected = sum(len(x) for x in s2_rows)
        if curr_collected >= len(needed_s2) and len(s2_distractors) >= max_distractors:
            break

    val_s2_df = pd.concat(s2_rows + s2_distractors, ignore_index=True).drop_duplicates(subset=["entity_id"])
    val_s2_path = os.path.join(val_out_dir, "val_source2.tsv")
    val_s2_df.to_csv(val_s2_path, sep="\t", index=False)
    print(f"Saved {val_s2_path} with {len(val_s2_df)} records.")

    # Extract matching and distractor S3 records
    print("Reading S3 records to collect true matches and distractors...")
    s3_rows = []
    s3_distractors = []
    max_distractors_s3 = len(needed_s3) * distractor_ratio

    for chunk in pd.read_csv(os.path.join(train_dir, "train_source3.tsv"), sep="\t", chunksize=100000):
        matched_chunk = chunk[chunk["entity_id"].isin(needed_s3)]
        if not matched_chunk.empty:
            s3_rows.append(matched_chunk)

        if len(s3_distractors) < max_distractors_s3:
            unmatched_chunk = chunk[~chunk["entity_id"].isin(needed_s3)]
            s3_distractors.append(unmatched_chunk.sample(min(len(unmatched_chunk), 5000), random_state=seed))

        curr_collected = sum(len(x) for x in s3_rows)
        if curr_collected >= len(needed_s3) and len(s3_distractors) >= max_distractors_s3:
            break

    val_s3_df = pd.concat(s3_rows + s3_distractors, ignore_index=True).drop_duplicates(subset=["entity_id"])
    val_s3_path = os.path.join(val_out_dir, "val_source3.tsv")
    val_s3_df.to_csv(val_s3_path, sep="\t", index=False)
    print(f"Saved {val_s3_path} with {len(val_s3_df)} records.")
    print("Validation split successfully created!")

if __name__ == "__main__":
    create_validation_split()
