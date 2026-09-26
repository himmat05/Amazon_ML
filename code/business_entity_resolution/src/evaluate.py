"""
Benchmark evaluation script for local validation.
Reads ground truth and prediction TSV files, computes the official Macro F_0.5 score,
and prints diagnostic metrics.
"""

import argparse
import sys
import pandas as pd
from metric import compute_macro_f05


def load_mapping_from_tsv(path: str, id_col: str, list_col: str):
    """Read a tab-separated file and return dict of id -> set(matched_ids)."""
    df = pd.read_csv(path, sep="\t", dtype=str)
    mapping = {}
    for _, row in df.iterrows():
        key = str(row[id_col]).strip()
        val_str = row[list_col]
        if pd.isna(val_str) or not str(val_str).strip():
            mapping[key] = set()
        else:
            mapping[key] = {x.strip() for x in str(val_str).split(",") if x.strip()}
    return mapping


def evaluate(ground_truth_path: str, predictions_path: str):
    print(f"Loading ground truth from: {ground_truth_path}")
    gt_map = load_mapping_from_tsv(ground_truth_path, "source1_entity_id", "matched_entity_ids")

    print(f"Loading predictions from: {predictions_path}")
    pred_map = load_mapping_from_tsv(predictions_path, "source1_entity_id", "matched_entity_ids")

    # Check for missing entities
    missing_entities = set(gt_map.keys()) - set(pred_map.keys())
    if missing_entities:
        print(f"WARNING: {len(missing_entities)} entities in ground truth are missing from predictions!")

    results = compute_macro_f05(gt_map, pred_map, beta=0.5)

    print("\n" + "=" * 50)
    print("         BENCHMARK EVALUATION RESULTS (F_0.5)      ")
    print("=" * 50)
    print(f"Total Evaluated Entities : {results['num_entities']:,}")
    print(f"Total Singletons (0 match): {results['num_singletons']:,} ({results['num_singletons']/results['num_entities']*100:.1f}%)")
    print(f"Singleton Accuracy       : {results['singleton_accuracy']*100:.2f}%")
    print(f"Non-Singleton F_0.5      : {results['non_singleton_f05']:.4f}")
    print("-" * 50)
    print(f"OVERALL MACRO F_0.5 SCORE: {results['macro_f05']:.4f}")
    print("=" * 50 + "\n")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate prediction F_0.5 score against ground truth.")
    parser.add_argument("--gt", required=True, help="Path to ground truth TSV")
    parser.add_argument("--pred", required=True, help="Path to predictions TSV")
    args = parser.parse_args()

    evaluate(args.gt, args.pred)
