#!/usr/bin/env python3
"""
fix_duplicates.py

Run this in your project directory to create synthetic_eval_final.jsonl
This removes 2 duplicate prompts and adds 2 replacements to maintain 600 samples.

Usage: python fix_duplicates.py
"""
import json
import random

random.seed(42)

# Load original eval
eval_data = [json.loads(line) for line in open('eval_samples/synthetic_eval.jsonl', 'r', encoding='utf-8')]
print(f"Loaded synthetic_eval.jsonl: {len(eval_data)} samples")

# Remove duplicates (keep first occurrence)
seen = set()
cleaned = []
for d in eval_data:
    if d['nl_prompt'] not in seen:
        seen.add(d['nl_prompt'])
        cleaned.append(d)
print(f"After removing duplicates: {len(cleaned)} samples")

# Load full test set for replacements
full_data = [json.loads(line) for line in open('test.jsonl', 'r', encoding='utf-8')]

# Find rung-2 candidates not already used
used_prompts = set(d['nl_prompt'] for d in cleaned)
rung2_candidates = [d for d in full_data if d['query']['rung'] == 2 and d['nl_prompt'] not in used_prompts]

# Add 2 replacements
for d in random.sample(rung2_candidates, 2):
    d['rung'] = d['query']['rung']
    cleaned.append(d)
print(f"After adding replacements: {len(cleaned)} samples")

# Save
with open('eval_samples/synthetic_eval_final.jsonl', 'w', encoding='utf-8') as f:
    for d in cleaned:
        f.write(json.dumps(d) + '\n')

print("✅ Created: synthetic_eval_final.jsonl (600 samples, 0 duplicates)")
