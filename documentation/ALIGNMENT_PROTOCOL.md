# TIMELY-Bench Alignment Protocol Card

## Overview

This document describes the time-alignment protocols used in TIMELY-Bench for fusing clinical time-series with notes.

---

## Time Reference Point

| Element | Definition |
|---------|------------|
| **T0** | ICU admission time (`intime` from `icustays`) |
| **Observation Window** | [T0, T0 + W hours] |
| **Prediction Target** | Events after observation window |

---

## Alignment Windows

| Window ID | Hours | Description |
|-----------|-------|-------------|
| W6 | 6h | Early warning (first 6 hours) |
| W12 | 12h | Standard observation |
| W24 | 24h | Full first-day observation |

---

## Time-Series Alignment

### Vital Signs & Labs

```
Data Source: chartevents, labevents
Alignment: charttime relative to T0
Aggregation: Hourly buckets [0, 1, 2, ..., W-1]
```

### Handling Missing Hours

1. **Forward Fill**: Carry last observation forward
2. **Zero Imputation**: Fill remaining NaN with 0
3. **Missingness Flags**: Binary indicator if feature ever observed

---

## Text Alignment

### Radiology Reports

```
Data Source: noteevents (category='Radiology')
Time Field: charttime
Alignment: hour_offset = floor((charttime - T0) / 3600)
```

### LLM Feature Injection

For each note at hour `h`:
- Extract 5 binary features using LLM
- Inject features at hour `h` and propagate to end of window
- If multiple notes exist, use logical OR

```python
# Injection logic
X[patient, h:, llm_features] = extracted_values
```

---

## Fusion Strategies

### 1. Early Fusion (Concatenation)

```
X_fused = concat(X_tabular, X_llm)
Model: XGBoost on concatenated features
```

### 2. Late Fusion (Probability Average)

```
p_tab = TabularModel(X_tabular)
p_text = TextModel(X_llm)
p_fused = (p_tab + p_text) / 2
```

### 3. Late Fusion (Weighted)

```
p_fused = 0.7 * p_tab + 0.3 * p_text
```

### 4. Temporal Fusion (GRU)

```
X_temporal[t, :] = concat(X_tabular[t, :], X_llm[t, :])
Model: GRU with final hidden state -> prediction
```

---

## Data Leakage Prevention

| Risk | Mitigation |
|------|------------|
| Future information | Strict time filtering: only data before T0+W |
| Patient overlap | GroupKFold by subject_id |
| Label leakage | Labels computed from data after observation window |
| Scaling leakage | StandardScaler fit only on training fold |

---

## Validation Protocol

1. **5-fold GroupKFold**: Grouped by subject_id
2. **Metrics**: AUROC (primary), AUPRC, Brier Score
3. **Reporting**: Mean ± Std across 5 folds

---

## Reproducibility Checklist

- [ ] Use provided data splits (or GroupKFold with same random_state=42)
- [ ] Apply StandardScaler within each fold
- [ ] Use identical time windows (6h/12h/24h from ICU admission)
- [ ] Report mean and std across folds
- [ ] Cite MIMIC-IV v3.1 and this benchmark

