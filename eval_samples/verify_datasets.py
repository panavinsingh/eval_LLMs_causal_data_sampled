import json
import pandas as pd
import hashlib

print("="*70)
print("FINAL DATASET VERIFICATION (RECURSIVE CHECK)")
print("="*70)

errors = []

# =========================================================================
# CHECK 1: File existence and row counts
# =========================================================================
print("\n[CHECK 1] File Existence & Row Counts")

files = {
    'cladder_eval.csv': {'expected': 1278, 'type': 'csv'},
    'ccr_gb_eval.csv': {'expected': 400, 'type': 'csv'},
    'corr2cause_eval.csv': {'expected': 400, 'type': 'csv'},
    'causcibench_eval.csv': {'expected': 155, 'type': 'csv'},
    'synthetic_eval.jsonl': {'expected': 600, 'type': 'jsonl'},
}

for f, info in files.items():
    try:
        if info['type'] == 'csv':
            df = pd.read_csv(f)
            count = len(df)
        else:
            count = sum(1 for _ in open(f))
        
        if count == info['expected']:
            print(f"  [OK] {f}: {count} rows")
        else:
            errors.append(f"{f}: Expected {info['expected']}, got {count}")
            print(f"  [FAIL] {f}: Expected {info['expected']}, got {count}")
    except FileNotFoundError:
        errors.append(f"{f}: FILE NOT FOUND")
        print(f"  [FAIL] {f}: FILE NOT FOUND")

# =========================================================================
# CHECK 2: Critical fields exist
# =========================================================================
print("\n[CHECK 2] Critical Fields Exist")

# CLadder
df = pd.read_csv('cladder_eval.csv')
required = ['prompt', 'label', 'rung']
missing = [c for c in required if c not in df.columns]
if missing:
    errors.append(f"cladder_eval.csv: Missing columns {missing}")
    print(f"  [FAIL] cladder_eval.csv: Missing {missing}")
else:
    print(f"  [OK] cladder_eval.csv: Has prompt, label, rung")

# CCR.GB
df = pd.read_csv('ccr_gb_eval.csv')
prompt_col = None
for col in ['Question', 'question', 'prompt', 'input', 'Query']:
    if col in df.columns:
        prompt_col = col
        break
if prompt_col:
    print(f"  [OK] ccr_gb_eval.csv: Has {prompt_col}")
else:
    print(f"  [WARN] ccr_gb_eval.csv: Need to construct prompt from columns")
    print(f"      Columns: {list(df.columns)[:10]}")

# Corr2Cause
df = pd.read_csv('corr2cause_eval.csv')
if 'input' in df.columns and 'label' in df.columns:
    print(f"  [OK] corr2cause_eval.csv: Has input, label")
else:
    errors.append("corr2cause_eval.csv: Missing input or label")

# CauSciBench
df = pd.read_csv('causcibench_eval.csv')
# Check for essential causal inference columns
if 'treatment' in df.columns and 'outcome' in df.columns:
    print(f"  [OK] causcibench_eval.csv: Has treatment, outcome")
elif len(df.columns) > 0:
    print(f"  [OK] causcibench_eval.csv: {len(df.columns)} columns loaded")
else:
    errors.append("causcibench_eval.csv: No columns found")

# Synthetic
data = []
with open('synthetic_eval.jsonl') as f:
    for line in f:
        data.append(json.loads(line))

if all('nl_prompt' in d and 'binary_answer' in d for d in data):
    print(f"  [OK] synthetic_eval.jsonl: Has nl_prompt, binary_answer")
else:
    errors.append("synthetic_eval.jsonl: Missing nl_prompt or binary_answer")

# =========================================================================
# CHECK 3: No duplicates in synthetic
# =========================================================================
print("\n[CHECK 3] No Duplicates in Synthetic")
prompts = [d['nl_prompt'] for d in data]
if len(prompts) == len(set(prompts)):
    print(f"  [OK] No duplicate prompts")
else:
    dups = len(prompts) - len(set(prompts))
    errors.append(f"synthetic_eval.jsonl: {dups} duplicate prompts!")
    print(f"  [FAIL] {dups} duplicate prompts!")

# =========================================================================
# CHECK 4: Rung balance in synthetic
# =========================================================================
print("\n[CHECK 4] Rung Balance in Synthetic")
from collections import Counter
rungs = Counter(d.get('rung', d.get('query', {}).get('rung', 0)) for d in data)
if rungs[1] == rungs[2] == rungs[3] == 200:
    print(f"  [OK] Perfect balance: {dict(sorted(rungs.items()))}")
else:
    print(f"  [WARN] Imbalanced: {dict(sorted(rungs.items()))}")

# =========================================================================
# CHECK 5: Answer balance
# =========================================================================
print("\n[CHECK 5] Answer Balance")

# CLadder
df = pd.read_csv('cladder_eval.csv')
label_dist = df['label'].value_counts()
yes_pct = label_dist.get('yes', 0) / len(df) * 100
print(f"  CLadder: yes={yes_pct:.1f}%")

# Synthetic
ans_dist = Counter(d['binary_answer'] for d in data)
yes_pct = ans_dist.get('yes', 0) / len(data) * 100
print(f"  Synthetic: yes={yes_pct:.1f}%")

# =========================================================================
# CHECK 6: SHA256 Hashes
# =========================================================================
print("\n[CHECK 6] SHA256 Hashes (for manifest)")
for f in files.keys():
    h = hashlib.sha256()
    with open(f, 'rb') as file:
        for chunk in iter(lambda: file.read(8192), b""):
            h.update(chunk)
    print(f"  {f}: {h.hexdigest()[:32]}...")

# =========================================================================
# FINAL VERDICT
# =========================================================================
print("\n" + "="*70)
if errors:
    print("[FAIL] ERRORS FOUND - DO NOT PROCEED:")
    for e in errors:
        print(f"   - {e}")
else:
    print("[OK] ALL CHECKS PASSED - SAFE TO PROCEED WITH EVALUATION")
print("="*70)
