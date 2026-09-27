import sys, os, time, pandas as pd, numpy as np
sys.path.insert(0, 'code/business_entity_resolution/src')
from preprocess import preprocess_dataframe
from blocking import ScalableBlocker
from features import FeatureExtractor
from model import EntityMatchingModel
from metric import compute_macro_f05
from postprocess import apply_cross_source_transitivity

val_dir = 'student_resource/dataset/val'
s1 = pd.read_csv(f'{val_dir}/val_source1_clean.tsv', sep='\t')
s2 = pd.read_csv(f'{val_dir}/val_source2_clean.tsv', sep='\t')
s3 = pd.read_csv(f'{val_dir}/val_source3_clean.tsv', sep='\t')
pool = pd.concat([s2, s3], ignore_index=True)
gt = pd.read_csv(f'{val_dir}/val_ground_truth.tsv', sep='\t')
gt_map = {r['source1_entity_id']: set(r['matched_entity_ids'].split(',')) if pd.notna(r['matched_entity_ids']) and r['matched_entity_ids'] else set() for _, r in gt.iterrows()}

model = EntityMatchingModel.load_model('code/business_entity_resolution/src/matching_model.pkl')
extractor = FeatureExtractor()

blocker = ScalableBlocker(max_candidates_per_entity=65)
blocker.fit_pool(pool)
cand_map = blocker.generate_candidates(s1)

X, y, pair_ids = extractor.extract_features_for_candidates(cand_map, s1, pool, gt_map)
probs = model.predict_probabilities(X)
s1_ids = s1['entity_id'].tolist()

from collections import defaultdict
s1_to_pairs = defaultdict(list)
for (sid, cid), p in zip(pair_ids, probs):
    s1_to_pairs[sid].append((cid, p))

print('--- SWEEPING DIRECT THRESHOLDS WITHOUT TRANSITIVITY ---')
for t in [0.70, 0.75, 0.78, 0.80, 0.82, 0.85, 0.88, 0.90, 0.92, 0.95]:
    pred_map = {sid: set(cid for cid, p in s1_to_pairs[sid] if p >= t) for sid in s1_ids}
    res = compute_macro_f05(gt_map, pred_map)
    n_matches = [len(m) for m in pred_map.values()]
    print("Thresh %.2f -> Macro F0.5: %.4f | Singletons: %.1f%% | Non-Sing: %.4f | Mean matches: %.2f" % 
          (t, res['macro_f05'], res['singleton_accuracy']*100, res['non_singleton_f05'], np.mean(n_matches)))

# Fast pool dictionary lookup for similarity
pool_records = {
    str(r["entity_id"]): (
        str(r["core_name"]) if pd.notna(r.get("core_name")) else "",
        str(r["clean_address"]) if pd.notna(r.get("clean_address")) else "",
    )
    for _, r in pool.iterrows()
}

print('\n--- TESTING WITH TRANSITIVITY AT VARIOUS RECOVERY THRESHOLDS ---')
for rec_offset in [0.05, 0.10, 0.12, 0.15]:
    for base_t in [0.78, 0.82, 0.85, 0.88]:
        post_matches = apply_cross_source_transitivity(
            candidate_scores=s1_to_pairs,
            pool_records=pool_records,
            high_conf_thresh=base_t,
            recovery_thresh=base_t - rec_offset,
        )
        post_pred_map = {sid: set(m) for sid, m in post_matches.items()}
        res = compute_macro_f05(gt_map, post_pred_map)
        n_matches = [len(m) for m in post_pred_map.values()]
        print("Base %.2f, Rec %.2f -> Macro F0.5: %.4f | Singletons: %.1f%% | Non-Sing: %.4f | Mean matches: %.2f" %
              (base_t, base_t - rec_offset, res['macro_f05'], res['singleton_accuracy']*100, res['non_singleton_f05'], np.mean(n_matches)))
