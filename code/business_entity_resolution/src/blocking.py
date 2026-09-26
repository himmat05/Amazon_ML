"""
Scalable Candidate Generation (Blocking) Engine for Business Entity Resolution.
Constructs a country-partitioned multi-key inverted index to slash the search space
from billions of pairs down to 15-35 high-recall candidates per Source 1 entity.
Produces official candidate_pairs.tsv.
"""

from collections import defaultdict
import time
from typing import Dict, List, Set, Tuple
import pandas as pd
from rapidfuzz import fuzz
from preprocess import ocr_fold_text


STOP_WORDS = {
    "the", "and", "for", "all", "inc", "corp", "llc", "ltd", "pvt", "sarl", "sas",
    "services", "solutions", "group", "india", "usa", "enterprises", "company"
}


def soundex(token: str) -> str:
    """Fast, zero-dependency Soundex phonetic algorithm for transliteration invariance."""
    token = token.upper()
    if not token or not token[0].isalpha():
        return ""
    first = token[0]
    mapping = {
        'B': '1', 'F': '1', 'P': '1', 'V': '1',
        'C': '2', 'G': '2', 'J': '2', 'K': '2', 'Q': '2', 'S': '2', 'X': '2', 'Z': '2',
        'D': '3', 'T': '3',
        'L': '4',
        'M': '5', 'N': '5',
        'R': '6',
    }
    encoded = [first]
    prev = mapping.get(first, '0')
    for ch in token[1:]:
        code = mapping.get(ch, '0')
        if code != '0' and code != prev:
            encoded.append(code)
        prev = code
    res = "".join(encoded).replace('0', '')
    return (res + "000")[:4]


GENERIC_ADDR_WORDS = {
    "street", "road", "floor", "suite", "avenue", "boulevard", "lane", "drive",
    "court", "plaza", "building", "cross", "main", "first", "second", "third",
    "block", "phase", "sector", "near", "opp", "opposite", "behind", "beside",
    "above", "below", "dist", "taluk", "post", "village", "india", "state", "city",
    "tower", "room", "dept", "highway", "expressway", "circle"
}


def extract_blocking_keys(
    country: str,
    core_name: str,
    clean_address: str,
    num_list: List[str],
    postal: str,
) -> List[str]:
    """Generate high-recall multi-angle blocking keys for a single entity record."""
    keys = []
    c = country.strip().upper()

    clean_core = "".join(ch for ch in core_name if ch.isalnum())
    words = [w for w in core_name.split() if len(w) >= 3 and w not in STOP_WORDS]

    # 1. Core Name N-Gram Prefixes (Length 3 and 4)
    if len(clean_core) >= 3:
        keys.append(f"{c}_P3:{clean_core[:3]}")
        if len(clean_core) >= 4:
            keys.append(f"{c}_P4:{clean_core[:4]}")

    # 2. Significant Word Tokens in Business Name (Up to 6 words)
    for w in words[:6]:
        keys.append(f"{c}_W:{w}")

    # 3. Phonetic Soundex Key (Transliteration / spelling invariance)
    if words:
        sx = soundex(words[0])
        if sx:
            keys.append(f"{c}_SX:{sx}")

    # 4. Sorted Word-Pair Prefix Key (handles swapped, transposed, or dropped middle words)
    if len(words) >= 2:
        top_words = sorted(words[:3])
        keys.append(f"{c}_SWP:{top_words[0][:3]}_{top_words[1][:3]}")

    # 5. Acronym & Initialism Keys (e.g., "KFC" <-> "Kentucky Fried Chicken")
    raw_tokens = [w for w in core_name.split() if w and w not in STOP_WORDS]
    if len(raw_tokens) >= 2:
        acr = "".join(w[0] for w in raw_tokens if w[0].isalnum()).upper()
        if 2 <= len(acr) <= 6:
            keys.append(f"{c}_ACR:{acr}")
    elif len(clean_core) >= 2 and len(clean_core) <= 5 and clean_core.isupper():
        keys.append(f"{c}_ACR:{clean_core}")

    # 6. Numeric Anchor + Street Token (combines street number with first street word)
    addr_tokens = [w for w in clean_address.split() if len(w) >= 4 and w not in STOP_WORDS and not w.isdigit()]
    if num_list and addr_tokens:
        keys.append(f"{c}_ST:{num_list[0]}_{addr_tokens[0]}")

    # 7. Postal / PIN Code Anchor + Initial
    if postal and len(postal) >= 4 and clean_core:
        keys.append(f"{c}_POST:{postal}_{clean_core[0]}")

    # 8. Distinctive Address Landmark / Locality Word Key (len >= 6, non-generic)
    rare_addr_tokens = [
        w for w in clean_address.split()
        if len(w) >= 6 and w not in STOP_WORDS and w not in GENERIC_ADDR_WORDS and not w.isdigit()
    ]
    for aw in rare_addr_tokens[:3]:
        keys.append(f"{c}_AW:{aw}")

    # 9. House / Building Number Anchor + First Letter of Name
    if num_list and clean_core:
        keys.append(f"{c}_NUM:{num_list[0]}_{clean_core[0]}")

    # 10. OCR-Folded Visual Invariance Key (resolves l vs I, 0 vs O, 1 vs i)
    ocr_core = ocr_fold_text(clean_core)
    if len(ocr_core) >= 3:
        keys.append(f"{c}_OCR3:{ocr_core[:3]}")

    # 11. Rare Distinctive Name Token + Locality / Postal / Number Key
    rare_name_tokens = [w for w in words if len(w) >= 5 and w not in STOP_WORDS]
    if rare_name_tokens:
        rw = rare_name_tokens[0]
        if postal and len(postal) >= 3:
            keys.append(f"{c}_NPOST:{rw}_{postal[:3]}")
        elif num_list:
            keys.append(f"{c}_NNUM:{rw}_{num_list[0]}")

    # 12. Character 3-Gram MinHash LSH Invariance Key
    if len(clean_core) >= 5:
        trigrams = sorted(list(set(clean_core[j:j+3] for j in range(len(clean_core)-2) if clean_core[j:j+3].isalnum())))
        if len(trigrams) >= 2:
            keys.append(f"{c}_LSH:{trigrams[0]}_{trigrams[1]}")
            if len(trigrams) >= 4:
                keys.append(f"{c}_LSH:{trigrams[-2]}_{trigrams[-1]}")

    return keys


class ScalableBlocker:
    """Scalable Multi-Key Inverted Index with Country Partitioning."""

    def __init__(self, max_candidates_per_entity: int = 75):
        self.max_candidates = max_candidates_per_entity
        self.index = defaultdict(list)
        self.pool_records: Dict[str, Tuple[str, str, str]] = {}  # id -> (core_name, clean_address, country)

    def fit_pool(self, df_pool: pd.DataFrame):
        """Index all Source 2 and Source 3 records into country-partitioned keys."""
        t0 = time.time()
        print(f"Indexing {len(df_pool):,} candidate records into inverted index...")

        ids = df_pool["entity_id"].astype(str).values
        countries = df_pool["country"].fillna("").astype(str).values
        cores = df_pool["core_name"].fillna("").astype(str).values
        addresses = df_pool["clean_address"].fillna("").astype(str).values
        postals = df_pool["postal_code"].fillna("").astype(str).values
        raw_nums = df_pool["address_numbers"].fillna("").astype(str).values
        num_lists = [n.split(",") if n else [] for n in raw_nums]

        for i in range(len(df_pool)):
            rec_id = ids[i]
            c = countries[i]
            core = cores[i]
            addr = addresses[i]
            post = postals[i]
            nums = num_lists[i]

            self.pool_records[rec_id] = (core, addr, c)

            keys = extract_blocking_keys(c, core, addr, nums, post)
            for k in keys:
                self.index[k].append(rec_id)

        print(f"Indexed {len(df_pool):,} records across {len(self.index):,} keys in {time.time()-t0:.2f}s")

    def retrieve_candidates_for_entity(
        self,
        country: str,
        core_name: str,
        clean_address: str,
        num_list: List[str],
        postal: str,
    ) -> List[str]:
        """Retrieve and rank candidates for a single Source 1 entity."""
        keys = extract_blocking_keys(country, core_name, clean_address, num_list, postal)

        candidate_counts = defaultdict(int)
        for k in keys:
            cand_list = self.index.get(k, ())
            # Prune mega-blocks (> 5000 records) to maintain high precision
            if len(cand_list) > 5000:
                continue
            for cand_id in cand_list:
                candidate_counts[cand_id] += 1

        if not candidate_counts:
            # Fallback if all keys were mega-blocks: use the smallest mega-block
            min_block = None
            min_len = float("inf")
            for k in keys:
                cand_list = self.index.get(k, ())
                if cand_list and len(cand_list) < min_len:
                    min_len = len(cand_list)
                    min_block = cand_list
            if min_block:
                for cand_id in min_block[:80]:
                    candidate_counts[cand_id] += 1
            else:
                return []

        # If candidates are within max_candidates, return them directly
        if len(candidate_counts) <= self.max_candidates:
            return list(candidate_counts.keys())

        # If candidate pool is large, pre-filter by key_freq before fuzzy scoring
        items = list(candidate_counts.items())
        if len(items) > 180:
            items.sort(key=lambda x: x[1], reverse=True)
            items = items[:180]

        scored_candidates = []
        for cand_id, key_freq in items:
            cand_core, cand_addr, _ = self.pool_records[cand_id]
            name_sim = fuzz.token_sort_ratio(core_name, cand_core)
            addr_sim = fuzz.token_sort_ratio(clean_address, cand_addr) if clean_address and cand_addr else 0.0
            rank_score = name_sim + (0.4 * addr_sim) + (key_freq * 12)
            scored_candidates.append((rank_score, cand_id))

        scored_candidates.sort(key=lambda x: x[0], reverse=True)
        return [cid for _, cid in scored_candidates[:self.max_candidates]]

    def generate_candidates(self, df_s1: pd.DataFrame) -> Dict[str, List[str]]:
        """Run candidate generation for all entities in df_s1."""
        t0 = time.time()
        print(f"Generating candidate sets for {len(df_s1):,} Source 1 entities...")

        s1_ids = df_s1["entity_id"].astype(str).values
        countries = df_s1["country"].fillna("").astype(str).values
        cores = df_s1["core_name"].fillna("").astype(str).values
        addresses = df_s1["clean_address"].fillna("").astype(str).values
        postals = df_s1["postal_code"].fillna("").astype(str).values
        raw_nums = df_s1["address_numbers"].fillna("").astype(str).values
        num_lists = [n.split(",") if n else [] for n in raw_nums]

        candidate_map = {}
        for i in range(len(df_s1)):
            s1_id = s1_ids[i]
            cands = self.retrieve_candidates_for_entity(
                country=countries[i],
                core_name=cores[i],
                clean_address=addresses[i],
                num_list=num_lists[i],
                postal=postals[i],
            )
            candidate_map[s1_id] = cands

        print(f"Candidate generation complete in {time.time()-t0:.2f}s")
        return candidate_map


def save_candidate_pairs(
    candidate_map: Dict[str, List[str]],
    output_path: str,
):
    """Save candidate_pairs.tsv according to official format requirements."""
    rows = []
    for s1_id, cands in candidate_map.items():
        rows.append({
            "source1_entity_id": s1_id,
            "candidate_entity_ids": ",".join(cands) if cands else "",
        })
    df_out = pd.DataFrame(rows)
    df_out.to_csv(output_path, sep="\t", index=False)
    print(f"Saved {len(df_out):,} rows to candidate file: {output_path}")


def evaluate_blocking(
    candidate_map: Dict[str, List[str]],
    gt_map: Dict[str, Set[str]],
) -> Dict[str, float]:
    """Calculate Recall Ceiling and Reduction metrics for the candidate set."""
    total_true = 0
    found_true = 0
    candidate_lengths = []

    for s1_id, true_set in gt_map.items():
        if not true_set:
            continue
        total_true += len(true_set)
        cands = set(candidate_map.get(s1_id, []))
        found_true += len(true_set & cands)
        candidate_lengths.append(len(cands))

    recall_ceiling = found_true / total_true if total_true > 0 else 1.0
    avg_cands = sum(candidate_lengths) / len(candidate_lengths) if candidate_lengths else 0.0

    print("\n" + "=" * 50)
    print("         BLOCKING STAGE QUALITY AUDIT             ")
    print("=" * 50)
    print(f"Total True Positive Links in GT : {total_true:,}")
    print(f"Captured in Candidate Sets      : {found_true:,}")
    print(f"Recall Ceiling (Upper Bound)    : {recall_ceiling * 100:.2f}%")
    print(f"Average Candidate Set Size      : {avg_cands:.1f} per entity")
    print("=" * 50 + "\n")

    return {
        "recall_ceiling": recall_ceiling,
        "avg_candidates": avg_cands,
        "total_true": total_true,
        "found_true": found_true,
    }
