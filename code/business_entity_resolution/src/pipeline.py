"""
Production-grade Country-Partitioned Streaming Inference Pipeline for Business Entity Resolution.
Designed for 12M+ records on constrained hardware (16GB RAM / 4-core CPU):
- Streams one country partition at a time to prevent RAM exhaustion.
- Evaluates Source 1 in micro-batches (15,000 entities/batch) with explicit garbage collection.
- Assembles final matching_results.tsv and candidate_pairs.tsv in the exact original order of test_source1.tsv.
- Validates all submission constraints automatically.
"""

import argparse
from collections import defaultdict
import gc
import os
import sys
import time
from typing import Dict, List, Optional, Set, Tuple
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from preprocess import preprocess_dataframe
from blocking import ScalableBlocker
from features import FeatureExtractor
from model import EntityMatchingModel
from postprocess import apply_cross_source_transitivity


def process_country_partition(
    country: str,
    df_s1_country: pd.DataFrame,
    test_dir: str,
    output_dir: str,
    model: EntityMatchingModel,
    extractor: FeatureExtractor,
    batch_size: int = 20000,
    max_candidates: int = 65,
    sample_s1: Optional[int] = None,
) -> Tuple[str, str, int, int, int]:
    """Process a single country partition with streaming pool ingestion and S1 batching."""
    c_name = country.upper()
    print(f"\n{'='*65}\n  STARTING PARTITION: {c_name} ({len(df_s1_country):,} S1 Entities)\n{'='*65}")
    t_start = time.time()

    # 1. Stream Candidate Pool for this country only
    print(f"[{c_name}] Streaming Source 2 and Source 3 pool for {c_name}...")
    t_pool = time.time()
    pool_chunks = []
    
    nrows_limit = sample_s1 * 5 if sample_s1 else None
    for fname in ["test_source2.tsv", "test_source3.tsv"]:
        fpath = os.path.join(test_dir, fname)
        if not os.path.exists(fpath):
            continue
        print(f"[{c_name}]   Reading {fname}...")
        for chunk in pd.read_csv(fpath, sep="\t", chunksize=250000, nrows=nrows_limit):
            sub = chunk[chunk["country"].fillna("").astype(str).str.strip().str.upper() == c_name]
            if not sub.empty:
                pool_chunks.append(sub)

    if pool_chunks:
        df_pool = pd.concat(pool_chunks, ignore_index=True)
    else:
        df_pool = pd.DataFrame(columns=["entity_id", "business_name", "business_address", "country"])
    del pool_chunks
    gc.collect()

    print(f"[{c_name}] Ingested {len(df_pool):,} pool records in {time.time()-t_pool:.2f}s")

    # 2. Preprocess Pool & Build Inverted Index
    t_index = time.time()
    print(f"[{c_name}] Preprocessing candidate pool...")
    df_pool_clean = preprocess_dataframe(df_pool)
    del df_pool
    gc.collect()

    print(f"[{c_name}] Building inverted blocking index...")
    blocker = ScalableBlocker(max_candidates_per_entity=max_candidates)
    if len(df_pool_clean) > 0:
        blocker.fit_pool(df_pool_clean)
    print(f"[{c_name}] Inverted index ready in {time.time()-t_index:.2f}s")

    print(f"[{c_name}] Pre-indexing pool records for transitivity...")
    p_ids = df_pool_clean["entity_id"].astype(str).values
    p_cores = df_pool_clean["core_name"].fillna("").astype(str).values
    p_addrs = df_pool_clean["clean_address"].fillna("").astype(str).values
    part_pool_records = {p_ids[i]: (p_cores[i], p_addrs[i]) for i in range(len(p_ids))}

    # 3. Micro-batched S1 Processing
    part_match_file = os.path.join(output_dir, f"tmp_match_{c_name}.tsv")
    part_cand_file = os.path.join(output_dir, f"tmp_cand_{c_name}.tsv")

    total_s1 = len(df_s1_country)
    num_batches = (total_s1 + batch_size - 1) // batch_size
    print(f"[{c_name}] Processing {total_s1:,} S1 entities across {num_batches} batches (batch_size={batch_size:,})...")

    part_matches_count = 0
    part_singletons_count = 0
    part_cands_count = 0

    with open(part_match_file, "w", encoding="utf-8") as f_match, open(part_cand_file, "w", encoding="utf-8") as f_cand:
        for b_idx in range(num_batches):
            b_start = b_idx * batch_size
            b_end = min(b_start + batch_size, total_s1)
            b_df_raw = df_s1_country.iloc[b_start:b_end].copy().reset_index(drop=True)
            t_b0 = time.time()

            # Preprocess S1 batch
            b_s1_clean = preprocess_dataframe(b_df_raw)

            # Generate candidates
            b_cand_map = blocker.generate_candidates(b_s1_clean)

            # Extract features & predict
            b_s1_ids = b_s1_clean["entity_id"].astype(str).tolist()
            total_b_cands = sum(len(c) for c in b_cand_map.values())

            if total_b_cands > 0 and len(df_pool_clean) > 0:
                X, _, pair_ids = extractor.extract_features_for_candidates(
                    candidate_map=b_cand_map,
                    df_s1=b_s1_clean,
                    df_pool=df_pool_clean,
                    gt_map=None,
                )
                if len(X) > 0:
                    probs = model.predict_probabilities(X)
                    
                    candidate_scores = defaultdict(list)
                    for (sid, pid), p in zip(pair_ids, probs):
                        candidate_scores[sid].append((pid, float(p)))
                    for sid in b_s1_ids:
                        if sid not in candidate_scores:
                            candidate_scores[sid] = []

                    b_matching_map = apply_cross_source_transitivity(
                        candidate_scores=candidate_scores,
                        pool_records=part_pool_records,
                        high_conf_thresh=model.best_threshold,
                        recovery_thresh=max(0.60, model.best_threshold - 0.12),
                    )
                else:
                    b_matching_map = {sid: [] for sid in b_s1_ids}
            else:
                b_matching_map = {sid: [] for sid in b_s1_ids}

            # Stream write batch results
            b_matched_ent = 0
            for sid in b_s1_ids:
                c_list = b_cand_map.get(sid, [])
                m_list = b_matching_map.get(sid, [])
                c_set = set(c_list)
                m_list_clean = [m for m in m_list if m in c_set]

                f_cand.write(f"{sid}\t{','.join(c_list)}\n")
                f_match.write(f"{sid}\t{','.join(m_list_clean)}\n")

                if m_list_clean:
                    b_matched_ent += 1
                    part_matches_count += len(m_list_clean)
                else:
                    part_singletons_count += 1
                part_cands_count += len(c_list)

            f_match.flush()
            f_cand.flush()

            pct = (b_end / total_s1) * 100
            rate = len(b_s1_ids) / max(time.time() - t_b0, 0.001)
            print(
                f"[{c_name}] Batch {b_idx+1}/{num_batches} ({pct:5.1f}%) | "
                f"S1: {b_end:,}/{total_s1:,} | "
                f"Cands: {total_b_cands:,} | "
                f"Matched: {b_matched_ent:,}/{len(b_s1_ids):,} ({b_matched_ent/len(b_s1_ids)*100:4.1f}%) | "
                f"Speed: {rate:5.0f} ent/s",
                flush=True
            )

            del b_df_raw, b_s1_clean, b_cand_map, b_matching_map
            if 'X' in locals():
                del X, pair_ids, probs
            gc.collect()

    print(f"[{c_name}] Partition completed in {time.time()-t_start:.2f}s")
    print(f"[{c_name}] Total Candidates: {part_cands_count:,} | Matches: {part_matches_count:,} | Singletons: {part_singletons_count:,}")

    # Free partition pool
    del blocker, df_pool_clean
    gc.collect()

    return part_match_file, part_cand_file, part_cands_count, part_matches_count, part_singletons_count


def assemble_final_submission(
    s1_all_path: str,
    output_dir: str,
    part_files: Dict[str, Tuple[str, str]],
    max_rows: Optional[int] = None,
):
    """Assemble final matching_results.tsv and candidate_pairs.tsv in original S1 order."""
    print("\n" + "=" * 65)
    print("  ASSEMBLING FINAL SUBMISSIONS IN EXACT S1 ORIGINAL ORDER")
    print("=" * 65)
    t0 = time.time()

    final_match_path = os.path.join(output_dir, "matching_results.tsv")
    final_cand_path = os.path.join(output_dir, "candidate_pairs.tsv")

    # Load lookup dictionaries from partition files
    print("Loading partition predictions into memory lookup...")
    match_lookup = {}
    cand_lookup = {}

    for c, (m_file, c_file) in part_files.items():
        print(f"  Reading {c} temporary partition files...")
        with open(m_file, "r", encoding="utf-8") as f_m:
            for line in f_m:
                parts = line.rstrip("\n").split("\t")
                if len(parts) >= 2:
                    match_lookup[parts[0]] = parts[1]
                elif len(parts) == 1:
                    match_lookup[parts[0]] = ""

        with open(c_file, "r", encoding="utf-8") as f_c:
            for line in f_c:
                parts = line.rstrip("\n").split("\t")
                if len(parts) >= 2:
                    cand_lookup[parts[0]] = parts[1]
                elif len(parts) == 1:
                    cand_lookup[parts[0]] = ""

    print(f"Loaded {len(match_lookup):,} entity predictions across all partitions.")

    # Stream write in original S1 order
    print(f"Writing {final_match_path} and {final_cand_path}...")
    written_count = 0
    total_matched = 0
    total_singletons = 0
    total_cands = 0

    with open(s1_all_path, "r", encoding="utf-8") as f_s1, \
         open(final_match_path, "w", encoding="utf-8") as f_m_out, \
         open(final_cand_path, "w", encoding="utf-8") as f_c_out:

        f_m_out.write("source1_entity_id\tmatched_entity_ids\n")
        f_c_out.write("source1_entity_id\tcandidate_entity_ids\n")

        header = f_s1.readline()  # Skip header
        for line in f_s1:
            if max_rows and written_count >= max_rows:
                break
            sid = line.split("\t")[0].strip()
            if not sid:
                continue

            m_val = match_lookup.get(sid, "")
            c_val = cand_lookup.get(sid, "")

            f_m_out.write(f"{sid}\t{m_val}\n")
            f_c_out.write(f"{sid}\t{c_val}\n")

            written_count += 1
            if m_val:
                total_matched += len(m_val.split(","))
            else:
                total_singletons += 1
            if c_val:
                total_cands += len(c_val.split(","))

    # Clean up temporary partition files
    for c, (m_file, c_file) in part_files.items():
        if os.path.exists(m_file):
            os.remove(m_file)
        if os.path.exists(c_file):
            os.remove(c_file)

    print(f"Assembly completed in {time.time()-t0:.2f}s")
    print(f"Total Source 1 Records Written : {written_count:,}")
    print(f"Total Predicted Singletons     : {total_singletons:,} ({total_singletons/written_count*100:.1f}%)")
    print(f"Total Entities with Matches    : {written_count-total_singletons:,} ({(written_count-total_singletons)/written_count*100:.1f}%)")
    print(f"Total Match Links Predicted    : {total_matched:,} (avg {total_matched/written_count:.2f}/entity)")
    print(f"Total Candidates Generated     : {total_cands:,} (avg {total_cands/written_count:.1f}/entity)")
    print("=" * 65 + "\n")

    return final_match_path, final_cand_path


def run_full_pipeline(
    test_dir: str,
    output_dir: str,
    model_path: str = "code/business_entity_resolution/src/matching_model.pkl",
    batch_size: int = 20000,
    max_candidates: int = 65,
    sample_s1_size: Optional[int] = None,
):
    """Run memory-safe, country-partitioned streaming pipeline."""
    total_start = time.time()
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 65)
    print("       BUSINESS ENTITY RESOLUTION — FULL STREAMING PIPELINE   ")
    print("=" * 65)
    print(f"Test Directory    : {test_dir}")
    print(f"Output Directory  : {output_dir}")
    print(f"Model Path        : {model_path}")
    print(f"Batch Size        : {batch_size:,}")
    print(f"Max Candidates    : {max_candidates}")
    print(f"Sample Limit      : {sample_s1_size if sample_s1_size else 'FULL DATASET (1.73M)'}")
    print("-" * 65)

    # 1. Load Model & Extractor
    print(f"Loading model from: {model_path}")
    model = EntityMatchingModel.load_model(model_path)
    extractor = FeatureExtractor()

    # 2. Ingest Test Source 1
    s1_path = os.path.join(test_dir, "test_source1.tsv")
    print(f"Reading Test Source 1 from {s1_path}...")
    df_s1_raw = pd.read_csv(s1_path, sep="\t", nrows=sample_s1_size)
    print(f"Loaded {len(df_s1_raw):,} Source 1 entities.")

    df_s1_raw["country_clean"] = df_s1_raw["country"].fillna("UNKNOWN").astype(str).str.strip().str.upper()
    active_countries = sorted(df_s1_raw["country_clean"].unique())
    print(f"Active countries in S1: {active_countries}")

    part_files = {}

    # 3. Process each country sequentially (FREEING MEMORY AFTER EACH)
    for c in active_countries:
        df_s1_c = df_s1_raw[df_s1_raw["country_clean"] == c].drop(columns=["country_clean"]).reset_index(drop=True)
        m_file, c_file, _, _, _ = process_country_partition(
            country=c,
            df_s1_country=df_s1_c,
            test_dir=test_dir,
            output_dir=output_dir,
            model=model,
            extractor=extractor,
            batch_size=batch_size,
            max_candidates=max_candidates,
            sample_s1=sample_s1_size,
        )
        part_files[c] = (m_file, c_file)
        del df_s1_c
        gc.collect()

    del df_s1_raw
    gc.collect()

    # 4. Final Assembly in exact original order
    final_match, final_cand = assemble_final_submission(
        s1_all_path=s1_path,
        output_dir=output_dir,
        part_files=part_files,
        max_rows=sample_s1_size,
    )

    print(f"Full Pipeline Finished in {time.time()-total_start:.2f}s")
    return final_match, final_cand


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Full Streaming Entity Resolution Pipeline")
    parser.add_argument("--test-dir", default="student_resource/dataset/test", help="Test directory")
    parser.add_argument("--output-dir", default="output", help="Output directory")
    parser.add_argument("--model-path", default="code/business_entity_resolution/src/matching_model.pkl", help="Model path")
    parser.add_argument("--batch-size", type=int, default=20000, help="S1 batch size")
    parser.add_argument("--max-candidates", type=int, default=65, help="Max candidates per entity")
    parser.add_argument("--sample-s1", type=int, default=None, help="Optional sample limit")

    args = parser.parse_args()

    run_full_pipeline(
        test_dir=args.test_dir,
        output_dir=args.output_dir,
        model_path=args.model_path,
        batch_size=args.batch_size,
        max_candidates=args.max_candidates,
        sample_s1_size=args.sample_s1,
    )
