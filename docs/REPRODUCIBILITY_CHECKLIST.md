# TIMELY-Bench Reproducibility Checklist

This checklist ensures that all experiments and results in TIMELY-Bench can be reproduced.

---

## ✅ Data Availability

- [x] **MIMIC-IV v2.2** - Available via PhysioNet (requires credentialed access)
- [x] **Predefined splits** - `data/processed/predefined_splits.csv` (fixed train/val/test)
- [x] **Episode JSONs** - Prebuilt in `episodes/episodes_enhanced/`

---

## ✅ Environment

| Requirement | Version |
|-------------|---------|
| Python | ≥ 3.9 |
| pandas | ≥ 1.4 |
| numpy | ≥ 1.21 |
| scikit-learn | ≥ 1.0 |
| xgboost | ≥ 1.6 |
| torch | ≥ 1.12 |
| transformers | ≥ 4.20 |

### Install
```bash
pip install -r requirements.txt
```

---

## ✅ Code Structure

```
TIMELY-Bench_Final/
├── code/
│   ├── baselines/          # Training scripts
│   ├── data_processing/    # Data preparation
│   └── config.py           # Configuration
├── data/processed/         # Processed data
├── episodes/               # Episode JSONs
├── results/                # Experiment results
├── Makefile                # Automation
└── Snakefile               # Pipeline (alternative)
```

---

## ✅ Reproduction Steps

### Quick Start (5 min)
```bash
# Run all baseline experiments
make all
```

### Full Pipeline (2-4 hours)
```bash
# 1. Generate data splits
make splits

# 2. Run tabular baselines
make baselines

# 3. Run fusion experiments
make fusion

# 4. Run GRU models
make gru

# 5. Evaluate
make eval
```

---

## ✅ Verification

### Data Integrity
```bash
make verify
make check-leakage
```

### Expected Outputs

| Task | Model | Expected AUROC |
|------|-------|----------------|
| Mortality | XGBoost | ~0.80 |
| Mortality | Full Fusion | ~0.84 |
| LOS | XGBoost | ~0.74 |
| LOS | Full Fusion | ~0.84 |

---

## ✅ Random Seeds

All experiments use fixed random seeds for reproducibility:

| Parameter | Value |
|-----------|-------|
| RANDOM_STATE | 42 |
| N_FOLDS | 5 |
| TEST_SIZE | 0.15 |

Defined in `code/config.py`.

---

## ✅ Hardware Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| RAM | 16 GB | 32 GB |
| GPU | Not required | NVIDIA 8GB+ |
| Storage | 100 GB | 200 GB |

---

## ✅ Files to Check

Before submission, verify these files exist:

| File | Purpose |
|------|---------|
| `data/processed/predefined_splits.csv` | Fixed data splits |
| `data/processed/cohorts/cohort_with_conditions.csv` | Cohort definition |
| `results/tabular_baselines/tabular_results.csv` | Baseline results |
| `results/fusion_baselines/fusion_results.csv` | Fusion results |
| `docs/DATA_CARD.md` | Data documentation |
| `docs/MODEL_CARD.md` | Model documentation |
| `docs/ALIGNMENT_PROTOCOL_CARD.md` | Alignment documentation |

---

## ✅ Known Issues

1. **Large file sizes**: Episode JSONs total ~50GB; use `rsync` for transfers
2. **Memory usage**: Full dataset training requires 16GB+ RAM
3. **MedCAT model**: Uses simplified keyword matching (not full MedCAT)

---

## Contact

For reproducibility issues, contact the project maintainers.
