# Causal Reasoning Evaluation Benchmark

<<<<<<< HEAD
< compliant evaluation datasets for benchmarking causal reasoning in Large Language Models.
=======
> Reproducible evaluation datasets for benchmarking causal reasoning in Large Language Models.
>>>>>>> 551bcb9 (Remove all ICML references)

## Overview

**2,833 stratified samples** across 5 datasets covering all levels of Pearl's Causal Hierarchy.

| Dataset | Samples | Strategy | Source |
|---------|---------|----------|--------|
| CLadder | 1,278 | `id % 8 == 0` | [causalnlp/CLadder](https://huggingface.co/datasets/causalnlp/CLadder) |
| CCR.GB | 400 | First 400 by Task ID | [jmaasch/compositional_causal_reasoning](https://huggingface.co/datasets/jmaasch/compositional_causal_reasoning) |
| Corr2Cause | 400 | Stratified by label | [causalnlp/corr2cause](https://huggingface.co/datasets/causalnlp/corr2cause) |
| CauSciBench | 155 | Full dataset | [causalNLP/CauSciBench](https://github.com/causalNLP/CauSciBench) |
| P3SA-Synthetic | 600 | Stratified by rung × answer | Custom |

## Quick Start

```bash
# Verify datasets
cd eval_samples
python verify_datasets.py

# Run evaluation (requires vLLM)
python local_llm_evaluation.py --model qwen3-32b --dataset all
```

## Reproducibility

- **Seed**: 42 (fixed for all sampling)
- **No RNG**: Deterministic slice rules
- **SHA-256**: Checksums for byte-level verification
- **Stratification**: Label/rung balance preserved

## Evaluation Config

```
Temperature: 0
Top-p: 1
Max tokens: 8192
```

## Structure

```
├── eval_samples/          # Stratified evaluation sets
│   ├── cladder_eval.csv
│   ├── ccr_gb_eval.csv
│   ├── corr2cause_eval.csv
│   ├── causcibench_eval.csv
│   ├── synthetic_eval.jsonl
│   └── verify_datasets.py
├── datasets/              # Full source datasets
├── scripts/               # Slicing & sampling scripts
└── sha256/                # Checksums
```

## License

Datasets retain original licenses (Apache-2.0, MIT, CC-BY-4.0).
