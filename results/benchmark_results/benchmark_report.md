# TIMELY-Bench Benchmark Results

## Overview

This report summarizes the benchmark results across multiple:
- **Time Windows**: 6h, 12h, 24h
- **Prediction Tasks**: Mortality, Prolonged LOS, 30-day Readmission  
- **Patient Cohorts**: All, Sepsis, AKI, Sepsis+AKI
- **Models**: Logistic Regression, XGBoost

## Results Summary


### Mortality

| Cohort | Model | 6h | 12h | 24h |
|--------|-------|-----|------|-----|
| all | LogisticRegression | 0.7783 | 0.8104 | 0.8418 |
| all | XGBoost | 0.7998 | 0.8331 | 0.8613 |
| sepsis | LogisticRegression | 0.7348 | 0.7626 | 0.7912 |
| sepsis | XGBoost | 0.7437 | 0.7759 | 0.8098 |
| aki | LogisticRegression | 0.7660 | 0.7968 | 0.8258 |
| aki | XGBoost | 0.7842 | 0.8137 | 0.8427 |
| sepsis_aki | LogisticRegression | 0.7244 | 0.7539 | 0.7798 |
| sepsis_aki | XGBoost | 0.7322 | 0.7624 | 0.7970 |

### Prolonged Los

| Cohort | Model | 6h | 12h | 24h |
|--------|-------|-----|------|-----|
| all | LogisticRegression | 0.6967 | 0.7351 | 0.7772 |
| all | XGBoost | 0.7222 | 0.7564 | 0.7985 |
| sepsis | LogisticRegression | 0.6677 | 0.7074 | 0.7473 |
| sepsis | XGBoost | 0.6910 | 0.7316 | 0.7722 |
| aki | LogisticRegression | 0.6751 | 0.7150 | 0.7534 |
| aki | XGBoost | 0.6991 | 0.7359 | 0.7784 |
| sepsis_aki | LogisticRegression | 0.6526 | 0.6898 | 0.7259 |
| sepsis_aki | XGBoost | 0.6747 | 0.7107 | 0.7529 |

### Readmission

| Cohort | Model | 6h | 12h | 24h |
|--------|-------|-----|------|-----|
| all | LogisticRegression | 0.6027 | 0.6092 | 0.6168 |
| all | XGBoost | 0.5981 | 0.6070 | 0.6218 |
| sepsis | LogisticRegression | 0.5923 | 0.5973 | 0.6070 |
| sepsis | XGBoost | 0.5860 | 0.5928 | 0.6041 |
| aki | LogisticRegression | 0.5888 | 0.6005 | 0.6110 |
| aki | XGBoost | 0.5877 | 0.5933 | 0.6116 |
| sepsis_aki | LogisticRegression | 0.5819 | 0.5905 | 0.5996 |
| sepsis_aki | XGBoost | 0.5690 | 0.5748 | 0.5868 |


## Key Findings

1. **Window Effect**: [To be filled based on results]
2. **Task Difficulty**: [To be filled based on results]
3. **Cohort Differences**: [To be filled based on results]

## Reproducibility

All experiments were conducted using:
- 5-fold GroupKFold cross-validation (grouped by subject_id)
- StandardScaler normalization within each fold
- Balanced class weights for imbalanced tasks

