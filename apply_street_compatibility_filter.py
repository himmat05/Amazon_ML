"""
Production Street-Number Compatibility & Branch Conflict Filter.
Prunes contradictory branch matches, orders matches by multi-key blocking agreement,
and recovers true singletons across France, India, and the US.
"""

import os
import sys
import time
import re
import pandas as pd
import numpy as np

def run_filter(
    cand_path: str = "output/candidate_pairs.tsv",
    raw_match_path: str = "output/matching_results_backup_raw.tsv",
    output_path: str = "output/matching_results.tsv",
    test_dir: str = "student_resource/dataset/test",
):
    t0 = time.time()
    print("=" * 65)
    print("   APPLYING STREET NUMBER COMPATIBILITY & BRANCH PRUNING   ")
    print("=" * 65)

    num_re = re.compile(r"\b(\d{1,4})\b")

    # 1. Build Street Number Index for all 8.5M pool records
    street_num_map = {}
    print("Step 1: Indexing pool street numbers from test_source2 & test_source3...")
    for fn in ["test_source2.tsv", "test_source3.tsv"]:
        fpath = os.path.join(test_dir, fn)
        for chunk in pd.read_csv(fpath, sep="\t", usecols=["entity_id", "business_address"], chunksize=500000):
            for eid, addr in zip(chunk["entity_id"], chunk["business_address"]):
                nums = num_re.findall(str(addr))
                if nums:
                    street_num_map[eid] = set(int(n) for n in nums if int(n) > 0)

    print(f"Indexed {len(street_num_map):,} pool records with street numbers in {time.time()-t0:.2f}s")

    # 2. Ingest Test Source 1 metadata
    print("Step 2: Loading Test Source 1 records...")
    s1_path = os.path.join(test_dir, "test_source1.tsv")
    s1_df = pd.read_csv(s1_path, sep="\t", usecols=["entity_id", "business_address", "country"])
    s1_addrs = s1_df["business_address"].tolist()
    countries = s1_df["country"].tolist()
    s1_ids = s1_df["entity_id"].tolist()
    total_entities = len(s1_ids)
    del s1_df

    # 3. Stream & Filter Matches
    print(f"Step 3: Streaming and filtering {total_entities:,} entities...")
    temp_output = output_path + ".tmp"
    
    pruned_conflicts = 0
    recovered_singletons = 0
    total_matches_kept = 0
    singletons_count = 0
    
    fr_counts = []
    in_counts = []
    us_counts = []

    with open(cand_path, "r", encoding="utf-8") as f_c, \
         open(raw_match_path, "r", encoding="utf-8") as f_m, \
         open(temp_output, "w", encoding="utf-8") as f_out:
        
        f_c.readline()
        f_m.readline()
        f_out.write("source1_entity_id\tmatched_entity_ids\n")

        for i in range(total_entities):
            l_c = f_c.readline()
            l_m = f_m.readline()
            if not l_c or not l_m:
                break

            sid = s1_ids[i]
            country = countries[i]

            c_parts = l_c.rstrip("\r\n").split("\t")
            m_parts = l_m.rstrip("\r\n").split("\t")

            cands = c_parts[1].split(",") if len(c_parts) > 1 and c_parts[1] else []
            matches = set(m_parts[1].split(",")) if len(m_parts) > 1 and m_parts[1] else set()

            # Rank matches strictly by candidate blocking order
            ranked = [c for c in cands if c in matches]

            # S1 street numbers
            s1_addr = str(s1_addrs[i])
            s1_nums = set(int(n) for n in num_re.findall(s1_addr) if int(n) > 0)

            filtered = []
            for cid in ranked:
                c_nums = street_num_map.get(cid, None)
                if s1_nums and c_nums is not None:
                    # If both S1 and candidate have street numbers, they must share at least one
                    if len(s1_nums & c_nums) == 0:
                        pruned_conflicts += 1
                        continue
                filtered.append(cid)

            # Recover true singletons from isolated low-rank matches
            if len(filtered) == 1 and filtered[0] in cands:
                if cands.index(filtered[0]) >= 3:
                    filtered = []
                    recovered_singletons += 1

            # Adaptive ceiling: 4 for France, 5 for India and US
            cap = 4 if country == "France" else 5
            final_matches = filtered[:cap]

            n_m = len(final_matches)
            total_matches_kept += n_m

            if country == "France":
                fr_counts.append(n_m)
            elif country == "India":
                in_counts.append(n_m)
            else:
                us_counts.append(n_m)

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

    # Copy to root folder as well
    import shutil
    shutil.copy2(output_path, "Amazon_ML_matching_results.tsv")

    print("\n" + "=" * 65)
    print("   FILTERING AND PRECISION OPTIMIZATION COMPLETE   ")
    print("=" * 65)
    print(f"Total Entities Processed  : {total_entities:,}")
    print(f"Contradictory Merges Cut  : {pruned_conflicts:,}")
    print(f"True Singletons Recovered : {recovered_singletons:,}")
    print(f"France  : Singletons = {np.mean([x==0 for x in fr_counts])*100:.2f}% | Mean matches = {np.mean(fr_counts):.3f}")
    print(f"India   : Singletons = {np.mean([x==0 for x in in_counts])*100:.2f}% | Mean matches = {np.mean(in_counts):.3f}")
    print(f"US      : Singletons = {np.mean([x==0 for x in us_counts])*100:.2f}% | Mean matches = {np.mean(us_counts):.3f}")
    all_m = fr_counts + in_counts + us_counts
    print(f"Overall : Singletons = {np.mean([x==0 for x in all_m])*100:.2f}% | Mean matches = {np.mean(all_m):.3f} [GT: ~3.46]")
    print(f"Elapsed Time              : {time.time()-t0:.2f}s")
    print("=" * 65)

if __name__ == "__main__":
    run_filter()
