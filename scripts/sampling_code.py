#!/usr/bin/env python3
"""
sampling_code.py – Stratified Sampling for Evaluation

Sampling Strategy:
──────────────────
  CLADDER      : Full dataset (1,278 rows)
  CCR.GB       : Full dataset (400 rows)
  corr2cause   : Stratified sample 400 by label (0/1 balance)
  CauSciBench  : Full dataset (155 rows)
  P3SA-Synthetic: Stratified sample 600 by rung × binary_answer

Total: ~2,833 samples

Seed: 42 (fixed for reproducibility)
"""

import hashlib
import json
import os
from collections import Counter
from pathlib import Path

import pandas as pd

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

SEED = 42
BASE_DIR = Path(__file__).parent.parent
DATASETS_DIR = BASE_DIR / "datasets"
EVAL_DIR = BASE_DIR / "eval_samples"

# Sample sizes
CORR2CAUSE_SAMPLE = 400
SYNTHETIC_SAMPLE = 600


# ─────────────────────────────────────────────────────────────────────────────
# Utility Functions
# ─────────────────────────────────────────────────────────────────────────────

def sha256_string(s: str) -> str:
    """Compute SHA-256 hash of a string."""
    return hashlib.sha256(s.encode()).hexdigest()


def stratified_sample(df: pd.DataFrame, strata_col: str, n: int, seed: int) -> pd.DataFrame:
    """
    Stratified sampling maintaining proportions.
    """
    # Calculate proportions
    counts = df[strata_col].value_counts()
    total = len(df)
    
    samples = []
    remaining = n
    
    for i, (stratum, count) in enumerate(counts.items()):
        # Proportional allocation
        if i == len(counts) - 1:
            # Last stratum gets remaining
            stratum_n = remaining
        else:
            stratum_n = int(round(n * count / total))
        
        stratum_n = min(stratum_n, count)  # Can't sample more than available
        remaining -= stratum_n
        
        stratum_df = df[df[strata_col] == stratum]
        samples.append(stratum_df.sample(n=stratum_n, random_state=seed))
    
    return pd.concat(samples, ignore_index=True)


def multi_stratified_sample(df: pd.DataFrame, strata_cols: list, n: int, seed: int) -> pd.DataFrame:
    """
    Multi-attribute stratified sampling.
    Creates joint strata from multiple columns.
    """
    # Create joint stratum key
    df = df.copy()
    df['_stratum'] = df[strata_cols].astype(str).agg('_'.join, axis=1)
    
    result = stratified_sample(df, '_stratum', n, seed)
    result = result.drop(columns=['_stratum'])
    
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Dataset Sampling Functions
# ─────────────────────────────────────────────────────────────────────────────

def sample_cladder() -> pd.DataFrame:
    """CLADDER: Use full dataset."""
    print("[1/5] CLADDER: Full dataset")
    df = pd.read_csv(DATASETS_DIR / "cladder" / "cladder_v15_slice_idmod8.csv")
    print(f"      → {len(df)} rows")
    return df


def sample_ccr_gb() -> pd.DataFrame:
    """CCR.GB: Use full dataset."""
    print("[2/5] CCR.GB: Full dataset")
    df = pd.read_csv(DATASETS_DIR / "ccr_gb" / "ccr_gb_first400_sorted.csv")
    print(f"      → {len(df)} rows")
    return df


def sample_corr2cause() -> pd.DataFrame:
    """corr2cause: Stratified sample 400 by label."""
    print(f"[3/5] corr2cause: Stratified sample {CORR2CAUSE_SAMPLE} by label")
    
    df = pd.read_csv(DATASETS_DIR / "corr2cause" / "corr2cause_full.csv")
    
    # Original distribution
    orig_dist = df['label'].value_counts().to_dict()
    print(f"      Original distribution: {orig_dist}")
    
    # Stratified sample
    sampled = stratified_sample(df, 'label', CORR2CAUSE_SAMPLE, SEED)
    
    # Verify distribution maintained
    new_dist = sampled['label'].value_counts().to_dict()
    print(f"      Sampled distribution: {new_dist}")
    print(f"      → {len(sampled)} rows")
    
    return sampled


def sample_causcibench() -> pd.DataFrame:
    """CauSciBench: Use full dataset."""
    print("[4/5] CauSciBench: Full dataset")
    df = pd.read_csv(DATASETS_DIR / "causcibench" / "causcibench_full.csv")
    print(f"      → {len(df)} rows")
    return df


def sample_synthetic() -> pd.DataFrame:
    """P3SA-Synthetic: Stratified sample 600 by rung × binary_answer."""
    print(f"[5/5] P3SA-Synthetic: Stratified sample {SYNTHETIC_SAMPLE} by rung × binary_answer")
    
    # Load JSONL
    data = []
    with open(DATASETS_DIR / "synthetic" / "synthetic_2550_full.jsonl", 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    
    df = pd.DataFrame(data)
    
    # Extract stratification columns
    df['rung'] = df['query'].apply(lambda q: q.get('rung', 1) if isinstance(q, dict) else 1)
    
    # Original distribution
    orig_rung = df['rung'].value_counts().to_dict()
    orig_answer = df['binary_answer'].value_counts().to_dict() if 'binary_answer' in df.columns else {}
    print(f"      Original rung distribution: {orig_rung}")
    print(f"      Original answer distribution: {orig_answer}")
    
    # Multi-attribute stratified sample
    strata_cols = ['rung']
    if 'binary_answer' in df.columns:
        strata_cols.append('binary_answer')
    
    sampled = multi_stratified_sample(df, strata_cols, SYNTHETIC_SAMPLE, SEED)
    
    # Verify distributions
    new_rung = sampled['rung'].value_counts().to_dict()
    new_answer = sampled['binary_answer'].value_counts().to_dict() if 'binary_answer' in sampled.columns else {}
    print(f"      Sampled rung distribution: {new_rung}")
    print(f"      Sampled answer distribution: {new_answer}")
    print(f"      → {len(sampled)} rows")
    
    return sampled


# ─────────────────────────────────────────────────────────────────────────────
# Verification Functions
# ─────────────────────────────────────────────────────────────────────────────

def verify_no_leakage(eval_df: pd.DataFrame, train_dag_ids: set = None) -> bool:
    """Verify no DAG overlap with training data."""
    if train_dag_ids is None:
        print("  ⚠ No training DAG IDs provided - skipping leakage check")
        return True
    
    if 'dag_id' not in eval_df.columns:
        print("  ⚠ No dag_id column in eval data")
        return True
    
    eval_dag_ids = set(eval_df['dag_id'].unique())
    overlap = eval_dag_ids & train_dag_ids
    
    if overlap:
        print(f"  ✗ LEAKAGE DETECTED: {len(overlap)} overlapping DAGs")
        return False
    else:
        print(f"  ✓ No leakage: {len(eval_dag_ids)} eval DAGs, 0 overlap with training")
        return True


def compute_manifest(samples: dict) -> dict:
    """Compute evaluation manifest with hashes."""
    manifest = {
        "seed": SEED,
        "n_samples": sum(len(df) for df in samples.values()),
        "datasets": {},
        "stratification": {}
    }
    
    # Compute hash of all sample IDs
    all_ids = []
    for name, df in samples.items():
        # Use index as ID if no explicit ID column
        if 'id' in df.columns:
            ids = df['id'].astype(str).tolist()
        elif 'dag_id' in df.columns:
            ids = df['dag_id'].astype(str).tolist()
        else:
            ids = [f"{name}_{i}" for i in range(len(df))]
        
        all_ids.extend(ids)
        manifest["datasets"][name] = {
            "n_samples": len(df),
            "sample_ids_hash": sha256_string(",".join(ids))
        }
    
    manifest["eval_set_hash"] = sha256_string(",".join(all_ids))
    
    # Add stratification info
    if 'corr2cause' in samples:
        df = samples['corr2cause']
        if 'label' in df.columns:
            manifest["stratification"]["corr2cause"] = df['label'].value_counts().to_dict()
    
    if 'synthetic' in samples:
        df = samples['synthetic']
        if 'rung' in df.columns:
            manifest["stratification"]["synthetic_rung"] = df['rung'].value_counts().to_dict()
        if 'binary_answer' in df.columns:
            manifest["stratification"]["synthetic_answer"] = df['binary_answer'].value_counts().to_dict()
    
    return manifest


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("STRATIFIED SAMPLING")
    print(f"Seed: {SEED}")
    print("=" * 60)
    
    # Create output directory
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    
    # Sample all datasets
    samples = {
        "cladder": sample_cladder(),
        "ccr_gb": sample_ccr_gb(),
        "corr2cause": sample_corr2cause(),
        "causcibench": sample_causcibench(),
        "synthetic": sample_synthetic(),
    }
    
    # Verify no leakage for synthetic
    print("\n" + "-" * 60)
    print("LEAKAGE VERIFICATION")
    print("-" * 60)
    # Note: Add training DAG IDs here if available
    verify_no_leakage(samples["synthetic"], train_dag_ids=None)
    
    # Compute manifest
    print("\n" + "-" * 60)
    print("GENERATING MANIFEST")
    print("-" * 60)
    manifest = compute_manifest(samples)
    
    # Save manifest
    manifest_path = EVAL_DIR / "eval_manifest.json"
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)
    print(f"  Saved: {manifest_path}")
    
    # Save sampled datasets
    for name, df in samples.items():
        if name == "synthetic":
            # Save as JSONL
            output_path = EVAL_DIR / f"{name}_eval.jsonl"
            with open(output_path, 'w', encoding='utf-8') as f:
                for _, row in df.iterrows():
                    f.write(json.dumps(row.to_dict(), default=str) + "\n")
        else:
            # Save as CSV
            output_path = EVAL_DIR / f"{name}_eval.csv"
            df.to_csv(output_path, index=False)
        print(f"  Saved: {output_path}")
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for name, df in samples.items():
        print(f"  {name}: {len(df)} samples")
    print(f"\n  Total: {manifest['n_samples']} samples")
    print(f"  Hash: {manifest['eval_set_hash'][:16]}...")
    print("=" * 60)


if __name__ == "__main__":
    main()
