"""
Helper script to package the final submission zip archive
strictly following the Amazon ML Challenge 2026 guidelines:

<team_name>_submission.zip
├── output/
│   ├── matching_results.tsv        # final matches (same file uploaded to leaderboard)
│   └── candidate_pairs.tsv         # blocking candidate set
├── code/
│   └── business_entity_resolution/
│       ├── src/                    # all source code & model artifact
│       ├── README.md               # reproduction instructions
│       └── requirements.txt        # pinned dependencies
└── Documentation_template.md       # completed methodology write-up
"""

import os
import sys
import time
import zipfile

def build_package(team_name: str = "Team"):
    t0 = time.time()
    zip_filename = f"{team_name}_submission.zip"
    print(f"Creating submission package: {zip_filename}...")

    # Files to include
    files_to_zip = [
        ("output/matching_results.tsv", "output/matching_results.tsv"),
        ("output/candidate_pairs.tsv", "output/candidate_pairs.tsv"),
        ("code/business_entity_resolution/README.md", "code/business_entity_resolution/README.md"),
        ("code/business_entity_resolution/requirements.txt", "code/business_entity_resolution/requirements.txt"),
        ("Documentation_template.md", "Documentation_template.md"),
    ]

    # Add all files in code/business_entity_resolution/src/
    src_dir = "code/business_entity_resolution/src"
    for fname in sorted(os.listdir(src_dir)):
        if fname == "__pycache__" or fname.endswith(".pyc"):
            continue
        fpath = os.path.join(src_dir, fname)
        if os.path.isfile(fpath):
            files_to_zip.append((fpath, f"code/business_entity_resolution/src/{fname}"))

    with zipfile.ZipFile(zip_filename, "w", compression=zipfile.ZIP_DEFLATED) as zipf:
        for local_path, arc_name in files_to_zip:
            if not os.path.exists(local_path):
                print(f"ERROR: Missing required file: {local_path}")
                sys.exit(1)
            file_size_mb = os.path.getsize(local_path) / (1024 * 1024)
            print(f"  Adding: {arc_name:52s} ({file_size_mb:6.1f} MB)")
            zipf.write(local_path, arcname=arc_name)

    zip_size_mb = os.path.getsize(zip_filename) / (1024 * 1024)
    print("\n" + "=" * 60)
    print(f"Package successfully created: {zip_filename}")
    print(f"Total Archive Size: {zip_size_mb:.2f} MB")
    print(f"Completed in {time.time()-t0:.2f}s")
    print("=" * 60)

if __name__ == "__main__":
    name = sys.argv[1] if len(sys.argv) > 1 else "Amazon_ML"
    build_package(name)
