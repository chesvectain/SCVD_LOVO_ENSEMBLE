# Detecting Unknown Vulnerabilities in Smart Contracts Using Opcode-Based Ensemble Learning with LOVO Evaluation

This repository contains the experiment code and result data for the paper:

> **"Detecting unknown vulnerabilities in smart contracts using opcode-based ensemble learning with LOVO evaluation"**  

---

## Overview

We propose a framework for detecting **unknown** smart contract vulnerabilities (i.e., vulnerability types unseen during training) using:
- **Dual feature representation**: LiWP n-gram sparse features + Skip-gram dense embeddings
- **Two-stage ensemble strategy**: within-feature soft voting (Best2, All, Fast2) + cross-feature late fusion
- **Strengthened LOVO evaluation**: Leave-One-Vulnerability-Out with 5-fold CV (40 evaluations total)
- **Adaptive threshold selection**: validation-set F1 maximization per fold

---

## Repository Structure

```
.
├── data/
│   ├── dataset_txhash_label.csv     # Transaction hashes + vulnerability labels (2.5 MB)
│   └── README_data.md               # Dataset description and reproduction instructions
│
├── code/
│   ├── preprocess.py                # Opcode sequence preprocessing & normalization
│   ├── 20260115_threshold_fixed.py  # Main experiment: single models + within-feature ensembles
│   ├── cross_feature_fusion.py      # Cross-feature late fusion experiment
│   ├── compare_thresholds.py        # Fixed vs. adaptive threshold comparison
│   └── summarize_all_results.py     # Final summary across all systems
│
└── results/
    ├── single_within/
    │   ├── results_raw.csv          # Per-fold results for all single models & within-feature ensembles
    │   └── ALL_RESULTS_COMBINED.csv # Aggregated summary (mean F1, ROC-AUC, PR-AUC per system)
    ├── cross_feature/
    │   ├── results_cross_feature.csv  # Per-fold cross-feature fusion results
    │   └── summary_cross_feature.csv  # Cross-feature aggregated summary
    └── threshold_comparison/
        ├── overall_summary.csv      # Fixed-0.5 vs. adaptive threshold comparison summary
        └── per_target_f1.csv        # Per-vulnerability-type F1 under each threshold mode
```

---

## Dataset

The full dataset consists of **33,202 Ethereum mainnet transaction samples** across 9 classes (S0–S8).

| Label | Vulnerability Type | Samples |
|-------|--------------------|---------|
| S0 | Normal (Benign) | 5,624 |
| S1 | Re-entrancy | 1,949 |
| S2 | Unexpected Function Invocation | 6,000 |
| S3 | No Check After Invocation | 748 |
| S4 | Missing Transfer Event | 5,997 |
| S5 | Strict Check for Balance | 710 |
| S6 | Timestamp & Block Number Dependency | 1,951 |
| S7 | Incorrect Auth. Check | 9,384 |
| S8 | Invalid Input Data / Failed Send | 839 |
| **Total** | | **33,202** |

The full dataset (12 GB, including opcode sequences) cannot be distributed directly due to size constraints.  
`data/dataset_txhash_label.csv` provides transaction hashes and labels to enable reproduction.  
See [`data/README_data.md`](data/README_data.md) for full reproduction instructions.

---

## Experiment Reproduction

### Requirements

```bash
Python >= 3.10
scikit-learn >= 1.3
gensim >= 4.3
numpy, pandas, scipy, joblib, matplotlib
# Optional GPU acceleration:
cuml (RAPIDS)
```

### Step 1: Reproduce the Dataset

Follow the instructions in [`data/README_data.md`](data/README_data.md) to collect opcode sequences from Ethereum mainnet using TxSpector and SODA.

### Step 2: Preprocess

```bash
python code/preprocess.py
```

### Step 3: Run Main Experiments (Single Models + Within-Feature Ensembles)

```bash
python code/20260115_threshold_fixed.py
```

Output: `runs/v5_fixed/`

### Step 4: Run Cross-Feature Late Fusion

```bash
python code/cross_feature_fusion.py
```

Output: `runs/cross_feature/`

### Step 5: Compare Threshold Strategies

```bash
python code/compare_thresholds.py
```

Output: `runs/threshold_comparison/`

### Step 6: Generate Final Summary

```bash
python code/summarize_all_results.py
```

Output: `runs/final_summary/`

---

## Key Results

| System | Mean F1 | ROC-AUC | PR-AUC |
|--------|---------|---------|--------|
| LiWP tri-gram + SVM (single, best) | 0.927 | 0.920 | 0.942 |
| LiWP tri-gram + Best2 ensemble | **0.945** | **0.939** | **0.951** |
| SkipGram + Best2 ensemble | 0.931 | 0.909 | 0.928 |
| Same ensembles with Fixed-0.5 threshold | 0.476–0.577 | — | — |

*All results use LOVO with 5-fold CV (40 evaluations). Adaptive threshold = validation-set F1 maximization.*

---

## Data Availability Statement

The transaction hash list and experiment code are publicly available at this repository.  
The full opcode sequence dataset can be reproduced from Ethereum mainnet transaction data using the tools and instructions provided in [`data/README_data.md`](data/README_data.md).

---

## License

Code: MIT License  
Data: CC BY 4.0
