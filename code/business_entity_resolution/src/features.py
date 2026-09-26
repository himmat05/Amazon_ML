"""
Pairwise Feature Engineering Pipeline for Business Entity Resolution.
Computes string metrics, address alignment, numeric token matching,
and structural compatibility using C++ SIMD accelerated RapidFuzz.
"""

from typing import Dict, List, Optional, Set, Tuple
import numpy as np
import pandas as pd
from rapidfuzz import fuzz
from rapidfuzz.distance import JaroWinkler, Levenshtein


from blocking import soundex

GENERIC_ADDR_WORDS = {
    "street", "road", "floor", "suite", "avenue", "boulevard", "lane", "drive",
    "court", "plaza", "building", "cross", "main", "first", "second", "third",
    "block", "phase", "sector", "near", "opp", "opposite", "behind", "beside",
    "above", "below", "dist", "taluk", "post", "village", "india", "state", "city",
    "tower", "room", "dept", "highway", "expressway", "circle"
}

FEATURE_NAMES = [
    "name_jaro_winkler",
    "name_levenshtein",
    "name_token_sort_ratio",
    "name_token_set_ratio",
    "name_partial_ratio",
    "name_exact_match",
    "name_len_diff",
    "name_len_ratio",
    "soundex_match",
    "acronym_match",
    "legal_suffix_match",
    "addr_token_jaccard",
    "addr_token_overlap",
    "addr_token_sort_ratio",
    "rare_addr_overlap",
    "street_num_match",
    "num_discrepancy",
    "postal_code_match",
    "full_text_token_set_ratio",
    "is_source_2",
]


def compute_pair_features(
    s1_core: str,
    s1_addr: str,
    s1_suffix: str,
    s1_nums: Set[str],
    s1_postal: str,
    s1_full: str,
    cand_id: str,
    cand_core: str,
    cand_addr: str,
    cand_suffix: str,
    cand_nums: Set[str],
    cand_postal: str,
    cand_full: str,
) -> List[float]:
    """Compute high-signal pairwise features between Source 1 and a candidate record."""
    # 1. Name Similarities
    jw = JaroWinkler.similarity(s1_core, cand_core)
    lev = Levenshtein.normalized_similarity(s1_core, cand_core)
    tsort = fuzz.token_sort_ratio(s1_core, cand_core) / 100.0
    tset = fuzz.token_set_ratio(s1_core, cand_core) / 100.0
    partial = fuzz.partial_ratio(s1_core, cand_core) / 100.0
    exact = 1.0 if s1_core and s1_core == cand_core else 0.0

    len1 = len(s1_core)
    len2 = len(cand_core)
    len_diff = abs(len1 - len2)
    len_ratio = (min(len1, len2) / max(len1, len2)) if max(len1, len2) > 0 else 1.0

    # Phonetic & Acronym Alignment
    s1_words = [w for w in s1_core.split() if len(w) >= 3]
    c_words = [w for w in cand_core.split() if len(w) >= 3]
    if s1_words and c_words:
        sx_match = 1.0 if soundex(s1_words[0]) == soundex(c_words[0]) else 0.0
    else:
        sx_match = 0.0

    s1_acr = "".join(w[0] for w in s1_core.split() if w and w[0].isalnum()).upper()
    c_acr = "".join(w[0] for w in cand_core.split() if w and w[0].isalnum()).upper()
    clean_s1 = "".join(ch for ch in s1_core if ch.isalnum()).upper()
    clean_c = "".join(ch for ch in cand_core if ch.isalnum()).upper()
    acr_match = 1.0 if (s1_acr and s1_acr == clean_c) or (c_acr and c_acr == clean_s1) or (len(s1_acr) >= 2 and s1_acr == c_acr) else 0.0

    # 2. Legal Suffix Agreement
    if s1_suffix and cand_suffix:
        suffix_match = 1.0 if s1_suffix == cand_suffix else -1.0
    else:
        suffix_match = 0.0

    # 3. Address Similarities
    s1_tokens = set(s1_addr.split())
    cand_tokens = set(cand_addr.split())
    overlap = len(s1_tokens & cand_tokens)
    union = len(s1_tokens | cand_tokens)
    addr_jaccard = (overlap / union) if union > 0 else 0.0
    addr_tsort = fuzz.token_sort_ratio(s1_addr, cand_addr) / 100.0

    # Rare / Landmark address word overlap
    s1_rare = {w for w in s1_tokens if len(w) >= 6 and w not in GENERIC_ADDR_WORDS and not w.isdigit()}
    c_rare = {w for w in cand_tokens if len(w) >= 6 and w not in GENERIC_ADDR_WORDS and not w.isdigit()}
    rare_overlap = float(len(s1_rare & c_rare))

    # 4. Street Number Match Score & Contradiction
    if s1_nums and cand_nums:
        if len(s1_nums & cand_nums) > 0:
            num_match = 1.0
            num_discrepancy = 0.0
        else:
            num_match = -1.0
            num_discrepancy = 1.0
    else:
        num_match = 0.0
        num_discrepancy = 0.0

    # 5. Postal / PIN Code Match Score
    if s1_postal and cand_postal:
        postal_match = 1.0 if s1_postal == cand_postal else -1.0
    else:
        postal_match = 0.0

    # 6. Composite Text Similarity
    full_tset = fuzz.token_set_ratio(s1_full, cand_full) / 100.0

    # 7. Source Origin Indicator (S2 vs S3)
    is_s2 = 1.0 if cand_id.startswith("S2-") else 0.0

    return [
        jw,
        lev,
        tsort,
        tset,
        partial,
        exact,
        len_diff,
        len_ratio,
        sx_match,
        acr_match,
        suffix_match,
        addr_jaccard,
        float(overlap),
        addr_tsort,
        rare_overlap,
        num_match,
        num_discrepancy,
        postal_match,
        full_tset,
        is_s2,
    ]


class FeatureExtractor:
    """Batch Feature Extractor for Candidate Pairs."""

    def __init__(self):
        self.feature_names = FEATURE_NAMES
        self._cached_pool_signature = None
        self._cached_pool_lookup = None

    def extract_features_for_candidates(
        self,
        candidate_map: Dict[str, List[str]],
        df_s1: pd.DataFrame,
        df_pool: pd.DataFrame,
        gt_map: Optional[Dict[str, Set[str]]] = None,
    ) -> Tuple[np.ndarray, Optional[np.ndarray], List[Tuple[str, str]]]:
        """Convert candidate pairs into feature matrix X, binary targets y, and pair index list."""
        # Fast S1 lookup dict using vectorized arrays
        s1_ids = df_s1["entity_id"].astype(str).values
        s1_cores = df_s1["core_name"].fillna("").astype(str).values
        s1_addrs = df_s1["clean_address"].fillna("").astype(str).values
        s1_sufs = df_s1["legal_suffix"].fillna("").astype(str).values
        s1_nums = [set(n.split(",")) if n else set() for n in df_s1["address_numbers"].fillna("").astype(str).values]
        s1_posts = df_s1["postal_code"].fillna("").astype(str).values
        s1_fulls = df_s1["blocking_text"].fillna("").astype(str).values

        s1_lookup = {
            s1_ids[i]: (s1_cores[i], s1_addrs[i], s1_sufs[i], s1_nums[i], s1_posts[i], s1_fulls[i])
            for i in range(len(s1_ids))
        }

        # Check if Pool lookup is already cached for this pool DataFrame
        pool_sig = (len(df_pool), id(df_pool))
        if self._cached_pool_signature == pool_sig and self._cached_pool_lookup is not None:
            pool_lookup = self._cached_pool_lookup
        else:
            print(f"Preparing lookup tables for {len(df_pool):,} candidate pool records...")
            p_ids = df_pool["entity_id"].astype(str).values
            p_cores = df_pool["core_name"].fillna("").astype(str).values
            p_addrs = df_pool["clean_address"].fillna("").astype(str).values
            p_sufs = df_pool["legal_suffix"].fillna("").astype(str).values
            p_nums = [set(n.split(",")) if n else set() for n in df_pool["address_numbers"].fillna("").astype(str).values]
            p_posts = df_pool["postal_code"].fillna("").astype(str).values
            p_fulls = df_pool["blocking_text"].fillna("").astype(str).values

            pool_lookup = {
                p_ids[i]: (p_cores[i], p_addrs[i], p_sufs[i], p_nums[i], p_posts[i], p_fulls[i])
                for i in range(len(p_ids))
            }
            self._cached_pool_signature = pool_sig
            self._cached_pool_lookup = pool_lookup

        total_pairs = sum(len(cands) for cands in candidate_map.values())
        print(f"Extracting features for {total_pairs:,} candidate pairs...")

        X_list = []
        y_list = [] if gt_map is not None else None
        pair_ids = []

        for s1_id, cands in candidate_map.items():
            if s1_id not in s1_lookup:
                continue
            s1_c, s1_a, s1_suf, s1_num, s1_post, s1_full = s1_lookup[s1_id]
            true_set = gt_map.get(s1_id, set()) if gt_map is not None else None

            for cand_id in cands:
                if cand_id not in pool_lookup:
                    continue
                c_c, c_a, c_suf, c_num, c_post, c_full = pool_lookup[cand_id]

                feats = compute_pair_features(
                    s1_core=s1_c,
                    s1_addr=s1_a,
                    s1_suffix=s1_suf,
                    s1_nums=s1_num,
                    s1_postal=s1_post,
                    s1_full=s1_full,
                    cand_id=cand_id,
                    cand_core=c_c,
                    cand_addr=c_a,
                    cand_suffix=c_suf,
                    cand_nums=c_num,
                    cand_postal=c_post,
                    cand_full=c_full,
                )

                X_list.append(feats)
                pair_ids.append((s1_id, cand_id))

                if y_list is not None:
                    y_list.append(1 if cand_id in true_set else 0)

        X = np.array(X_list, dtype=np.float32)
        y = np.array(y_list, dtype=np.int32) if y_list is not None else None

        print(f"Extracted feature matrix shape: {X.shape}")
        if y is not None:
            pos_count = int(np.sum(y))
            print(f"Class balance: {pos_count:,} positive ({pos_count/len(y)*100:.2f}%), {len(y)-pos_count:,} negative")

        return X, y, pair_ids
