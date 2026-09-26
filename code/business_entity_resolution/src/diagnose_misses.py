import sys
import os
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from evaluate import load_mapping_from_tsv

gt = load_mapping_from_tsv("student_resource/output/val_gt_subset.tsv", "source1_entity_id", "matched_entity_ids")
p1 = load_mapping_from_tsv("student_resource/output/val_matching_results.tsv", "source1_entity_id", "matched_entity_ids")

missed = [(s, g - p1[s]) for s, g in gt.items() if len(g - p1[s]) > 0]
print(f"Entities with missed matches: {len(missed)}")
print(f"Total missed links: {sum(len(m[1]) for m in missed)}")

s1_df = pd.read_csv("student_resource/dataset/val/val_source1_clean.tsv", sep="\t").set_index("entity_id")
s2_df = pd.read_csv("student_resource/dataset/val/val_source2_clean.tsv", sep="\t").set_index("entity_id")
s3_df = pd.read_csv("student_resource/dataset/val/val_source3_clean.tsv", sep="\t").set_index("entity_id")
pool = pd.concat([s2_df, s3_df])

print("\n--- Diagnostic: Why Matches Were Missed ---")
for i, (sid, m_set) in enumerate(missed[:6]):
    s1_r = s1_df.loc[sid]
    print(f"\n[Case {i+1}] S1 ({sid}):")
    print(f"   Name   : {s1_r['business_name']}")
    print(f"   Address: {s1_r['business_address']}")
    for mid in list(m_set)[:2]:
        if mid in pool.index:
            pr = pool.loc[mid]
            if isinstance(pr, pd.DataFrame):
                pr = pr.iloc[0]
            print(f"   --> MISSED MATCH ({mid}):")
            print(f"       Name   : {pr['business_name']}")
            print(f"       Address: {pr['business_address']}")
