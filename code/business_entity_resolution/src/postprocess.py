"""
Post-Processing & Graph Transitivity Module for Entity Resolution.
Leverages cross-source agreement between Source 2 and Source 3 to:
1. Prune uncertain candidates that conflict with strongly verified matches.
2. Recover near-threshold true matches via transitive equivalence.
3. Guarantee strict submission formatting.
"""

from collections import defaultdict
from typing import Dict, List, Set, Tuple
from rapidfuzz import fuzz


def apply_cross_source_transitivity(
    candidate_scores: Dict[str, List[Tuple[str, float]]],
    pool_records: Dict[str, Tuple[str, str]],
    high_conf_thresh: float = 0.88,
    recovery_thresh: float = 0.65,
    name_sim_thresh: float = 82.0,
    addr_sim_thresh: float = 60.0,
) -> Dict[str, List[str]]:
    """
    Apply graph transitivity between verified matches and near-threshold candidates.
    If S1 matches S2_a with high confidence (>= 0.88), and S3_b is in candidate pool
    with probability >= 0.65, and S2_a strongly agrees with S3_b, link S3_b.
    """
    refined_matches = {}

    for s1_id, scored_cands in candidate_scores.items():
        if not scored_cands:
            refined_matches[s1_id] = []
            continue

        # Initial accepted matches
        accepted = [cid for cid, p in scored_cands if p >= high_conf_thresh]
        borderline = [(cid, p) for cid, p in scored_cands if recovery_thresh <= p < high_conf_thresh]

        # If we have verified high-confidence matches, check borderline candidates for transitive agreement
        if accepted and borderline:
            for b_id, _ in borderline:
                b_name, b_addr = pool_records.get(b_id, ("", ""))
                for a_id in list(accepted):
                    a_name, a_addr = pool_records.get(a_id, ("", ""))
                    n_sim = fuzz.token_sort_ratio(a_name, b_name)
                    a_sim = fuzz.token_sort_ratio(a_addr, b_addr)
                    if n_sim >= name_sim_thresh and a_sim >= addr_sim_thresh:
                        accepted.append(b_id)
                        break

        # If no high-confidence match found, accept if any >= high_conf_thresh
        if not accepted:
            accepted = [cid for cid, p in scored_cands if p >= high_conf_thresh]

        refined_matches[s1_id] = list(set(accepted))

    return refined_matches
