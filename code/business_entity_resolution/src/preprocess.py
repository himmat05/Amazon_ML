"""
Text and Address Normalization Engine for Business Entity Resolution.
Handles multi-country noise, Unicode errors, legal suffix standardization,
and numeric token extraction for US, India, and France.
"""

import re
import unicodedata
from typing import Dict, List, Set, Tuple
import pandas as pd


# 1. Unicode & Character Normalization
RE_REPLACEMENT_CHAR = re.compile(r"[\ufffd\x7f-\x9f]")
RE_AMPERSAND = re.compile(r"\s*&\s*|\s*\+\s*")
RE_PUNCT = re.compile(r"['\"\`\(\)\[\]\{\}\<\>\\\/\|\:\;\,\.\?\!\*\#\_\~]")
RE_MULTISPACE = re.compile(r"\s+")

# 2. Multi-Country Legal Suffixes (US, India, France, International)
# Standardized mapping: alias -> canonical label
LEGAL_SUFFIX_MAP = {
    # US & General
    "limited liability company": "llc",
    "limited liability co": "llc",
    "limited company": "ltd",
    "incorporation": "inc",
    "incorporated": "inc",
    "corporation": "corp",
    "company": "co",
    "limited": "ltd",
    "corp": "corp",
    "inc": "inc",
    "llc": "llc",
    "ltd": "ltd",
    "co": "co",
    "llp": "llp",
    "pllc": "pllc",
    "dba": "dba",
    
    # India
    "private limited": "pvt ltd",
    "pvt limited": "pvt ltd",
    "pvt ltd": "pvt ltd",
    "private ltd": "pvt ltd",
    "privatelimited": "pvt ltd",
    "pvtltd": "pvt ltd",
    
    # France
    "societe a responsabilite limitee": "sarl",
    "societe par actions simplifiee": "sas",
    "societe par actions simplifiee unipersonnelle": "sasu",
    "entreprise unipersonnelle a responsabilite limitee": "eurl",
    "societe anonyme": "sa",
    "societe en nom collectif": "snc",
    "societe civile immobiliere": "sci",
    "sarl": "sarl",
    "sasu": "sasu",
    "sas": "sas",
    "eurl": "eurl",
    "snc": "snc",
    "sci": "sci",
    "sa": "sa",
    "fils": "fils",
}

# Regex to detect and strip legal suffixes from the end of business names
_legal_pattern_str = r"\b(" + "|".join(
    sorted([re.escape(k) for k in LEGAL_SUFFIX_MAP.keys()], key=len, reverse=True)
) + r")\b\s*$"
RE_LEGAL_SUFFIX = re.compile(_legal_pattern_str, re.IGNORECASE)

# 3. Address Keyword Standardizations
ADDRESS_ABBREVIATIONS = {
    r"\bst\b": "street",
    r"\bstr\b": "street",
    r"\brd\b": "road",
    r"\bave\b": "avenue",
    r"\bav\b": "avenue",
    r"\bdr\b": "drive",
    r"\bln\b": "lane",
    r"\bblvd\b": "boulevard",
    r"\bct\b": "court",
    r"\bpl\b": "place",
    r"\bhwy\b": "highway",
    r"\bpkwy\b": "parkway",
    r"\bsq\b": "square",
    r"\bste\b": "suite",
    r"\bapt\b": "apt",
    r"\bfl\b": "floor",
    r"\bflr\b": "floor",
    r"\bbldg\b": "building",
    r"\bopp\b": "opposite",
    r"\bnr\b": "near",
    r"\bb\/h\b": "behind",
    r"\bbd\b": "boulevard",  # French boulevard abbr
}

RE_ADDRESS_STANDARDS = [
    (re.compile(pattern, re.IGNORECASE), repl)
    for pattern, repl in ADDRESS_ABBREVIATIONS.items()
]

# Numeric patterns
RE_NUMBERS = re.compile(r"\b\d+\b")
RE_PIN_CODE_INDIA = re.compile(r"\b([1-9]\d{5})\b")  # 6-digit Indian PIN
RE_POSTAL_US_FR = re.compile(r"\b(\d{5})\b")        # 5-digit US ZIP or French Postal Code

# URL & Domain Patterns
RE_URL_PREFIX = re.compile(r"^(?:https?:\/\/)?(?:www\.)+", re.IGNORECASE)
RE_DOMAIN_SUFFIX = re.compile(r"\.(?:com|co\.in|org|net|in|fr|io|biz|info|gov|edu|co|us|eu)\b", re.IGNORECASE)

# OCR Glyph Folding Table (maps visually ambiguous characters together)
OCR_FOLD_TRANS = str.maketrans({
    "l": "i",
    "1": "i",
    "0": "o",
    "5": "s",
    "w": "v",
})


def ocr_fold_text(text: str) -> str:
    """Fold visually ambiguous characters (OCR confusion like l vs I, 0 vs O)."""
    return text.translate(OCR_FOLD_TRANS)


def clean_text(text: str) -> str:
    """Enhanced sanitization: Indic transliteration to Latin, unicode normalization, lowercasing, punctuation removal."""
    if not text or pd.isna(text):
        return ""
    text_str = str(text)

    # 1. Transliterate non-Latin scripts (Hindi/Devanagari, Tamil, Bengali, French accents) to Latin ASCII
    try:
        import anyascii
        text_str = anyascii.anyascii(text_str)
    except Exception:
        pass

    # 2. Normalize unicode (decompose accents, then strip non-spacing marks)
    text_str = unicodedata.normalize("NFKD", text_str)
    text_str = "".join(c for c in text_str if not unicodedata.combining(c))

    # 3. Remove replacement chars & unprintable
    text_str = RE_REPLACEMENT_CHAR.sub(" ", text_str)

    # 4. Strip URL web prefixes (www., http://) and domain extensions (.com, .co.in)
    text_str = RE_URL_PREFIX.sub("", text_str)
    text_str = RE_DOMAIN_SUFFIX.sub("", text_str)

    # 5. Replace & and + with ' and '
    text_str = RE_AMPERSAND.sub(" and ", text_str)

    # 6. Lowercase
    text_str = text_str.lower()

    # 7. Remove punctuation
    text_str = RE_PUNCT.sub(" ", text_str)

    # 8. Collapse multiple spaces
    text_str = RE_MULTISPACE.sub(" ", text_str).strip()
    return text_str


RE_PREFIX_LEGAL = re.compile(r"^(?:(?:inc|corp|llc|ltd|pvt|sa|sas|sarl|the)\s+)+", re.IGNORECASE)
RE_DBA = re.compile(r"^.*?\bdba\s+", re.IGNORECASE)


def normalize_name(raw_name: str) -> Tuple[str, str, str]:
    """Clean business name, extract legal suffix, handle DBA and leading legal prefixes,
    and produce stripped core name.

    Returns:
        (clean_name, core_name, canonical_legal_suffix)
    """
    clean = clean_text(raw_name)
    if not clean:
        return "", "", ""

    # Strip DBA prefix if present: "Brand A dba Brand B" -> "Brand B"
    clean = RE_DBA.sub("", clean).strip()

    # Strip leading legal prefix if present: "inc empire express" -> "empire express"
    clean = RE_PREFIX_LEGAL.sub("", clean).strip()

    # Look for trailing legal suffix
    match = RE_LEGAL_SUFFIX.search(clean)
    if match:
        raw_suffix = match.group(1).strip()
        canonical_suffix = LEGAL_SUFFIX_MAP.get(raw_suffix, raw_suffix)
        core = clean[:match.start()].strip()
        if not core:
            core = clean
        return clean, core, canonical_suffix

    return clean, clean, ""


def normalize_address(raw_addr: str) -> Tuple[str, List[str], str]:
    """Clean address, standardize street/locality keywords, and extract numeric tokens.

    Returns:
        (clean_address, numbers_list, postal_code)
    """
    clean = clean_text(raw_addr)
    if not clean:
        return "", [], ""

    # Standardize street/address tokens
    for pattern, repl in RE_ADDRESS_STANDARDS:
        clean = pattern.sub(repl, clean)

    clean = RE_MULTISPACE.sub(" ", clean).strip()

    # Extract all numeric tokens (house numbers, units, etc.)
    numbers = RE_NUMBERS.findall(clean)

    # Detect postal code (India 6-digit or US/France 5-digit)
    pin_match = RE_PIN_CODE_INDIA.search(clean)
    if pin_match:
        postal = pin_match.group(1)
    else:
        post_match = RE_POSTAL_US_FR.search(clean)
        postal = post_match.group(1) if post_match else ""

    return clean, numbers, postal


def preprocess_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Preprocess an entire DataFrame of entities efficiently.

    Adds columns:
    - clean_name
    - core_name
    - legal_suffix
    - clean_address
    - address_numbers (comma-separated numbers for easy filtering)
    - postal_code
    - blocking_text (core_name + ' ' + clean_address)
    """
    df = df.copy()

    # Preprocess names
    name_results = [normalize_name(name) for name in df["business_name"]]
    df["clean_name"] = [r[0] for r in name_results]
    df["core_name"] = [r[1] for r in name_results]
    df["legal_suffix"] = [r[2] for r in name_results]

    # Preprocess addresses
    addr_results = [normalize_address(addr) for addr in df["business_address"]]
    df["clean_address"] = [r[0] for r in addr_results]
    df["address_numbers"] = [",".join(r[1]) for r in addr_results]
    df["postal_code"] = [r[2] for r in addr_results]

    # Create composite text for indexing / blocking
    df["blocking_text"] = df["core_name"] + " " + df["clean_address"]
    df["country"] = df["country"].fillna("").astype(str).str.strip().str.upper()

    return df


if __name__ == "__main__":
    # Test sample cases across US, India, and France
    test_cases = [
        ("ON BRKLEY LLC", "1708-1710 PATRICIA CT, MECAHNICSBURG, PA", "US"),
        ("ON Berkley LLC", "1708 Patricia Court, Upper Allen Township, PA", "US"),
        ("Smart Healthcare Private Limited", "303, 3Rd Floor Sakar 5 B/H Natraj Cinema Ashram Road, Ahmedabad, Gujarat", "India"),
        ("<< Team Ecole", "175 Boulevard du Prsident Franklin Roosevelt, Bordeaux", "France"),
        ("ZNB Club SARL", "5 bis Rue Pierre Dignac, La Teste-de-Buch", "France"),
    ]

    df_test = pd.DataFrame(test_cases, columns=["business_name", "business_address", "country"])
    df_out = preprocess_dataframe(df_test)
    
    print("--- Preprocessed Test Output ---")
    for _, r in df_out.iterrows():
        print(f"Original: {r['business_name']} | {r['business_address']}")
        print(f"Core Name: '{r['core_name']}' | Suffix: '{r['legal_suffix']}'")
        print(f"Clean Addr: '{r['clean_address']}' | Numbers: {r['address_numbers']} | Postal: {r['postal_code']}")
        print("-" * 60)
