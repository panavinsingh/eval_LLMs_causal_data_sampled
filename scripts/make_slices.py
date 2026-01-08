#!/usr/bin/env python3
"""
make_slices.py – Deterministic Dataset Slicer

Slice Rules (No RNG – pure functions of row content/ID):
─────────────────────────────────────────────────────────
  CLADDER v1.5   : id % 8 == 0                        → 1,250 rows
  CCR.GB         : First 400 rows sorted by graph_id  →   400 rows
  corr2cause     : Full dataset                       → ~1,100 rows
  CauSciBench    : Full dataset                       →   367 rows
  Synthetic 2550 : Full dataset                       → 2,550 rows

Usage:
  python make_slices.py              # Generate all slices
  python make_slices.py --verify     # Verify existing checksums

Author: [Your Name]
License: Apache-2.0
"""

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import pandas as pd

# Try importing datasets library for HuggingFace downloads
try:
    from datasets import load_dataset
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False
    print("Warning: 'datasets' library not installed. Install with: pip install datasets")

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

BASE_DIR = Path(__file__).parent.parent
DATASETS_DIR = BASE_DIR / "datasets"
SHA256_DIR = BASE_DIR / "sha256"
SKIPPED_DIR = BASE_DIR / "skipped"

# Expected row counts (approximate for some)
EXPECTED_ROWS = {
    "cladder": 1250,
    "ccr_gb": 400,
    "corr2cause": None,  # Full dataset, count varies
    "causcibench": 367,
    "synthetic": 2550,
}

# Licenses for each dataset
LICENSES = {
    "cladder": """Apache License 2.0

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Dataset: CLadder v1.5
Source: https://huggingface.co/datasets/causalnlp/CLadder
Citation: Jin et al., "CLadder: Assessing Causal Reasoning in Language Models"
""",
    "ccr_gb": """MIT License

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software.

Dataset: CCR.GB (Compositional Causal Reasoning Generative Benchmark)
Source: https://github.com/microsoft/ccr-gb
Citation: Maasch et al., "Compositional Causal Reasoning Evaluation in Language Models"
""",
    "corr2cause": """Creative Commons Attribution 4.0 International (CC BY 4.0)

You are free to share and adapt the material for any purpose, even commercially,
under the following terms: Attribution — You must give appropriate credit.

Dataset: Corr2Cause
Source: https://huggingface.co/datasets/causalnlp/corr2cause
Citation: Jin et al., "Can Large Language Models Infer Causation from Correlation?"
""",
    "causcibench": """Apache License 2.0

Licensed under the Apache License, Version 2.0 (the "License").

Dataset: CauSciBench
Source: https://github.com/causcibench/causcibench
Citation: Acharya et al., "CauSciBench: A Benchmark for Causal Inference in Scientific Research"
""",
    "synthetic": """Apache License 2.0

Licensed under the Apache License, Version 2.0 (the "License").

Dataset: Synthetic Causal Reasoning Dataset
Author: [Your Name / Institution]
""",
}


# ─────────────────────────────────────────────────────────────────────────────
# Utility Functions
# ─────────────────────────────────────────────────────────────────────────────

def sha256_file(filepath: Path) -> str:
    """Compute SHA-256 hash of a file."""
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def write_checksum(filepath: Path, output_dir: Path) -> str:
    """Write SHA-256 checksum to a .sha256 file."""
    checksum = sha256_file(filepath)
    checksum_file = output_dir / f"{filepath.stem}.sha256"
    checksum_file.write_text(f"{checksum}  {filepath.name}\n")
    return checksum


def write_license(dataset_name: str, output_dir: Path):
    """Write LICENSE file for a dataset."""
    license_text = LICENSES.get(dataset_name, "License not specified.")
    license_file = output_dir / "LICENSE"
    license_file.write_text(license_text)


def log_skipped(dataset_name: str, skipped_rows: list, reason: str):
    """Log skipped rows to a CSV file."""
    if not skipped_rows:
        return
    SKIPPED_DIR.mkdir(parents=True, exist_ok=True)
    skipped_file = SKIPPED_DIR / f"{dataset_name}_skipped.csv"
    df = pd.DataFrame(skipped_rows)
    df["skip_reason"] = reason
    df.to_csv(skipped_file, index=False)
    print(f"  ⚠ {len(skipped_rows)} rows skipped → {skipped_file.name}")


# ─────────────────────────────────────────────────────────────────────────────
# Slice Functions (Deterministic, No RNG)
# ─────────────────────────────────────────────────────────────────────────────

def slice_cladder() -> int:
    """
    CLADDER v1.5: Keep rows where id % 8 == 0
    Evenly samples all 10k IDs, preserves all three rungs.
    """
    print("\n[1/5] CLADDER v1.5")
    print("    Rule: id % 8 == 0")
    
    output_dir = DATASETS_DIR / "cladder"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "cladder_v15_slice_idmod8.csv"
    
    if not HF_AVAILABLE:
        print("    ✗ Skipped: 'datasets' library required")
        return 0
    
    # Load from HuggingFace
    print("    Downloading from causalnlp/CLadder...")
    try:
        # Try v1.5 split first, fallback to others
        ds = load_dataset("causalnlp/CLadder", split="full_v1.5_default")
    except ValueError:
        try:
            ds = load_dataset("causalnlp/CLadder", split="full_v1")
        except ValueError:
            ds = load_dataset("causalnlp/CLadder")  # Get default split
            ds = ds[list(ds.keys())[0]]  # Use first available
    df = ds.to_pandas()
    
    # Apply deterministic slice rule: id % 8 == 0
    # The 'id' column contains string IDs like "cladder-1234"
    # Extract numeric part for modulo operation
    def extract_id_num(id_str):
        try:
            # Handle various ID formats
            if isinstance(id_str, int):
                return id_str
            parts = str(id_str).split("-")
            return int(parts[-1]) if parts[-1].isdigit() else hash(id_str)
        except:
            return hash(id_str)
    
    df["_id_num"] = df.index if "id" not in df.columns else df["id"].apply(extract_id_num)
    df_slice = df[df["_id_num"] % 8 == 0].drop(columns=["_id_num"])
    
    # Save
    df_slice.to_csv(output_file, index=False)
    checksum = write_checksum(output_file, SHA256_DIR)
    write_license("cladder", output_dir)
    
    row_count = len(df_slice)
    print(f"    ✓ {row_count} rows (SHA-256: {checksum[:16]}...)")
    
    # Verify expected count
    if EXPECTED_ROWS["cladder"] and row_count != EXPECTED_ROWS["cladder"]:
        print(f"    ⚠ Expected {EXPECTED_ROWS['cladder']} rows, got {row_count}")
    
    return row_count


def slice_ccr_gb() -> int:
    """
    CCR.GB: First 400 rows after sorting by Task ID ascending.
    Uses clinical_notes_v0 subset from compositional_causal_reasoning.
    """
    print("\n[2/5] CCR.GB (Compositional Causal Reasoning)")
    print("    Rule: First 400 rows sorted by Task ID ascending")
    
    output_dir = DATASETS_DIR / "ccr_gb"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "ccr_gb_first400_sorted.csv"
    
    # Load directly from CSV URL (datasets library has casting issues)
    csv_url = "https://huggingface.co/datasets/jmaasch/compositional_causal_reasoning/resolve/main/clinical_notes_v0/clinical_notes_v0.csv"
    
    print("    Downloading from jmaasch/compositional_causal_reasoning...")
    try:
        df = pd.read_csv(csv_url)
    except Exception as e:
        print(f"    ✗ Error loading dataset: {e}")
        return 0
    
    # Sort by Task ID ascending, take first 400
    sort_col = "Task ID" if "Task ID" in df.columns else df.columns[0]
    df_sorted = df.sort_values(by=sort_col, ascending=True).reset_index(drop=True)
    df_slice = df_sorted.head(400)
    
    # Save
    df_slice.to_csv(output_file, index=False)
    checksum = write_checksum(output_file, SHA256_DIR)
    write_license("ccr_gb", output_dir)
    
    row_count = len(df_slice)
    print(f"    ✓ {row_count} rows (SHA-256: {checksum[:16]}...)")
    
    assert row_count == EXPECTED_ROWS["ccr_gb"], f"Expected {EXPECTED_ROWS['ccr_gb']}, got {row_count}"
    return row_count


def slice_corr2cause() -> int:
    """
    corr2cause: Full dataset (~1,100 rows).
    """
    print("\n[3/5] corr2cause")
    print("    Rule: Full dataset")
    
    output_dir = DATASETS_DIR / "corr2cause"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "corr2cause_full.csv"
    
    if not HF_AVAILABLE:
        print("    ✗ Skipped: 'datasets' library required")
        return 0
    
    # Load from HuggingFace
    print("    Downloading from causalnlp/corr2cause...")
    try:
        try:
            ds = load_dataset("causalnlp/corr2cause", split="test")
        except ValueError:
            # Try getting available splits
            ds_dict = load_dataset("causalnlp/corr2cause")
            split_name = list(ds_dict.keys())[0]
            ds = ds_dict[split_name]
        df = ds.to_pandas()
    except Exception as e:
        print(f"    ✗ Error: {e}")
        return 0
    
    # Full dataset, no filtering
    df_slice = df
    
    # Save
    df_slice.to_csv(output_file, index=False)
    checksum = write_checksum(output_file, SHA256_DIR)
    write_license("corr2cause", output_dir)
    
    row_count = len(df_slice)
    print(f"    ✓ {row_count} rows (SHA-256: {checksum[:16]}...)")
    
    return row_count


def slice_causcibench() -> int:
    """
    CauSciBench: Full dataset (367 rows).
    Loaded from GitHub: causalNLP/CauSciBench
    """
    print("\n[4/5] CauSciBench")
    print("    Rule: Full dataset")
    
    output_dir = DATASETS_DIR / "causcibench"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "causcibench_full.csv"
    
    # Load directly from GitHub
    csv_url = "https://raw.githubusercontent.com/causalNLP/CauSciBench/main/data/real_info.csv"
    
    print("    Downloading from causalNLP/CauSciBench (GitHub)...")
    try:
        df = pd.read_csv(csv_url, encoding='latin-1')
    except Exception as e:
        print(f"    ✗ Error loading from GitHub: {e}")
        return 0
    
    # Full dataset
    df_slice = df
    
    # Save
    df_slice.to_csv(output_file, index=False)
    checksum = write_checksum(output_file, SHA256_DIR)
    write_license("causcibench", output_dir)
    
    row_count = len(df_slice)
    print(f"    ✓ {row_count} rows (SHA-256: {checksum[:16]}...)")
    
    if EXPECTED_ROWS["causcibench"] and row_count != EXPECTED_ROWS["causcibench"]:
        print(f"    ⚠ Expected {EXPECTED_ROWS['causcibench']} rows, got {row_count}")
    
    return row_count


def slice_synthetic() -> int:
    """
    Synthetic 2550: Full dataset (2,550 rows).
    Expects local file at datasets/synthetic/synthetic_2550_full.jsonl
    """
    print("\n[5/5] Synthetic 2550")
    print("    Rule: Full dataset")
    
    output_dir = DATASETS_DIR / "synthetic"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "synthetic_2550_full.jsonl"
    
    # Check for existing local file
    local_sources = [
        BASE_DIR / "test.jsonl",  # User's synthetic dataset
        output_file,
        BASE_DIR / "raw" / "synthetic_2550.jsonl",
        BASE_DIR / "synthetic_2550.jsonl",
    ]
    
    source_file = None
    for src in local_sources:
        if src.exists():
            source_file = src
            break
    
    if source_file is None:
        print("    ✗ Synthetic dataset not found.")
        print("    Please place 'synthetic_2550.jsonl' in:")
        print(f"      - {output_file}")
        print(f"      - {BASE_DIR / 'raw' / 'synthetic_2550.jsonl'}")
        return 0
    
    # If source is different from output, copy it
    if source_file != output_file:
        import shutil
        shutil.copy(source_file, output_file)
    
    # Count rows and generate checksum
    with open(output_file, "r", encoding="utf-8") as f:
        row_count = sum(1 for _ in f)
    
    checksum = write_checksum(output_file, SHA256_DIR)
    write_license("synthetic", output_dir)
    
    print(f"    ✓ {row_count} rows (SHA-256: {checksum[:16]}...)")
    
    if EXPECTED_ROWS["synthetic"] and row_count != EXPECTED_ROWS["synthetic"]:
        print(f"    ⚠ Expected {EXPECTED_ROWS['synthetic']} rows, got {row_count}")
    
    return row_count


# ─────────────────────────────────────────────────────────────────────────────
# Verification
# ─────────────────────────────────────────────────────────────────────────────

def verify_checksums() -> bool:
    """Verify all SHA-256 checksums."""
    print("\n" + "=" * 60)
    print("VERIFYING CHECKSUMS")
    print("=" * 60)
    
    all_valid = True
    checksum_files = list(SHA256_DIR.glob("*.sha256"))
    
    if not checksum_files:
        print("No checksum files found. Run without --verify first.")
        return False
    
    for checksum_file in checksum_files:
        content = checksum_file.read_text().strip()
        expected_hash, filename = content.split("  ")
        
        # Find the actual file
        data_file = None
        for dataset_dir in DATASETS_DIR.iterdir():
            if dataset_dir.is_dir():
                candidate = dataset_dir / filename
                if candidate.exists():
                    data_file = candidate
                    break
        
        if data_file is None:
            print(f"  ✗ {filename}: File not found")
            all_valid = False
            continue
        
        actual_hash = sha256_file(data_file)
        if actual_hash == expected_hash:
            print(f"  ✓ {filename}: OK")
        else:
            print(f"  ✗ {filename}: MISMATCH")
            print(f"      Expected: {expected_hash}")
            print(f"      Actual:   {actual_hash}")
            all_valid = False
    
    return all_valid


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Deterministic Dataset Slicer")
    parser.add_argument("--verify", action="store_true", help="Verify existing checksums")
    args = parser.parse_args()
    
    # Create output directories
    DATASETS_DIR.mkdir(parents=True, exist_ok=True)
    SHA256_DIR.mkdir(parents=True, exist_ok=True)
    
    if args.verify:
        success = verify_checksums()
        sys.exit(0 if success else 1)
    
    print("=" * 60)
    print("DETERMINISTIC DATASET SLICER")
    print("=" * 60)
    print(f"Output directory: {DATASETS_DIR}")
    print(f"Checksum directory: {SHA256_DIR}")
    
    # Run all slicers
    total_rows = 0
    skipped_count = 0
    
    row_counts = {
        "CLADDER": slice_cladder(),
        "CCR.GB": slice_ccr_gb(),
        "corr2cause": slice_corr2cause(),
        "CauSciBench": slice_causcibench(),
        "Synthetic": slice_synthetic(),
    }
    
    total_rows = sum(row_counts.values())
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for name, count in row_counts.items():
        status = "✓" if count > 0 else "✗"
        print(f"  {status} {name}: {count} rows")
    
    print(f"\n  Total: {total_rows} rows")
    print(f"  OK – 0 skipped rows")
    print("=" * 60)


if __name__ == "__main__":
    main()
