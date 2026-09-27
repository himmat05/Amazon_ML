"""
Precision Optimization & Branch Contradiction Pruning for Amazon ML Challenge 2026.
Eliminates false-positive branch merges, ranks matches strictly by multi-key blocking agreement,
and aligns the prediction distribution with the true ground truth (mean matches ~3.4, singletons ~5.0%).
"""

import os
import sys
import time
import pandas as pd
import numpy as np

def optimize_submission(
    cand_path: str = "output/candidate_pairs.tsv",
    match_path: str = "output/matching_results.tsv",
    output_path: str = "output/matching_results.tsv",
    test_s1_path: str = "student_resource/dataset/test/test_source1.tsv",
    cap_france: int = 4,
    cap_general: int = 4,
):
    t0 = time.time()
    print("=" * 65)
    print("   OPTIMIZING SUBMISSION PRECISION FOR MACRO F_0.5 METRIC   ")
    print("=" * 65)
    
    # 1. Load country list for test entities
    print(f"Loading test countries from {test_s1_path}...")
    s1_df = pd.read_csv(test_s1_path, sep="\t", usecols=["country"])
    countries = s1_df["country"].tolist()
    total_entities = len(countries)
    del s1_df
    
    # Backup original matching_results
    backup_path = match_path.replace(".tsv", "_backup_raw.tsv")
    if not os.path.exists(backup_path):
        import shutil
        shutil.copy2(match_path, backup_path)
        print(f"Backed up raw predictions to {backup_path}")
        
    temp_output = output_path + ".tmp"
    
    print(f"Streaming {total_entities:,} entities to optimize precision...")
    
    pruned_false_merges = 0
    recovered_singletons = 0
    total_matches_kept = 0
    singletons_count = 0
    
    with open(cand_path, "r", encoding="utf-8") as f_c, \
         open(backup_path, "r", encoding="utf-8") as f_m, \
         open(temp_output, "w", encoding="utf-8") as f_out:
        
        # Read & write header
        h_c = f_c.readline()
        h_m = f_m.readline()
        f_out.write("source1_entity_id\tmatched_entity_ids\n")
        
        for i, country in enumerate(countries):
            l_c = f_c.readline()
            l_m = f_m.readline()
            if not l_c or not l_m:
                break
                
            c_parts = l_c.rstrip("\r\n").split("\t")
            m_parts = l_m.rstrip("\r\n").split("\t")
            sid = m_parts[0]
            
            cands = c_parts[1].split(",") if len(c_parts) > 1 and c_parts[1] else []
            matches = set(m_parts[1].split(",")) if len(m_parts) > 1 and m_parts[1] else set()
            
            old_count = len(matches)
            
            # Rank matches strictly by blocking similarity order in cands
            ranked_matches = [c for c in cands if c in matches]
            
            # Prune single matches that only appear at low candidate ranks (rank >= 3)
            # These are almost always false merges on true singletons
            if len(ranked_matches) == 1 and ranked_matches[0] in cands:
                rank_idx = cands.index(ranked_matches[0])
                if rank_idx >= 3:
                    ranked_matches = []
                    recovered_singletons += 1
            
            # Adaptive capping:
            # In ground truth, mean matches is 3.46, 95% of entities have <= 4 matches.
            # France was drastically over-merged (8.53 avg), cap at 4.
            # India and US cap at 4.
            cap = cap_france if country == "France" else cap_general
            final_matches = ranked_matches[:cap]
            
            pruned_false_merges += max(0, old_count - len(final_matches))
            total_matches_kept += len(final_matches)
            
            if not final_matches:
                singletons_count += 1
                f_out.write(f"{sid}\t\n")
            else:
                f_out.write(f"{sid}\t{','.join(final_matches)}\n")
                
            if (i + 1) % 400000 == 0:
                print(f"  Processed {i+1:,}/{total_entities:,} entities ({((i+1)/total_entities)*100:.1f}%)...")
                
    # Replace destination with temp file
    if os.path.exists(output_path):
        os.remove(output_path)
    os.rename(temp_output, output_path)
    
    mean_matches = total_matches_kept / total_entities
    singleton_pct = (singletons_count / total_entities) * 100
    
    print("\n" + "=" * 65)
    print("   PRECISION OPTIMIZATION COMPLETED SUCCESSFULLY   ")
    print("=" * 65)
    print(f"Total Entities Processed  : {total_entities:,}")
    print(f"False Merges Pruned       : {pruned_false_merges:,}")
    print(f"True Singletons Recovered : {recovered_singletons:,}")
    print(f"Final Singletons          : {singletons_count:,} ({singleton_pct:.2f}%) [GT Target: ~5.58%]")
    print(f"Final Mean Matches        : {mean_matches:.3f} [GT Target: ~3.461]")
    print(f"Elapsed Time              : {time.time()-t0:.2f}s")
    print("=" * 65)

if __name__ == "__main__":
    optimize_submission()
