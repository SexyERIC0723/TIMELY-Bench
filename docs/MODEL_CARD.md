# Model Card: TIMELY-Bench Baselines v2.0

## Overview

This document describes the baseline models implemented in TIMELY-Bench for benchmarking multimodal EHR fusion. Version 2.0 adds enhanced reasoning features including syndrome detection, reasoning chains, and LLM-generated disease timelines.

---

## Models Summary

| Model | Type | Input | Test AUROC (Mortality) |
|-------|------|-------|------------------------|
| **Full Feature Fusion** | Tree-based | TS + Annotations + BERT + Concepts | **0.844** |
| Enhanced Reasoning | Tree-based | TS + Syndrome + Timeline + State | 0.816 |
| BERT + Annotation | Tree-based | TS + Annotations + BERT | 0.840 |
| Enhanced GRU | Deep Learning | Time-series + LLM + Annotations | 0.831 |
| Temporal GRU | Deep Learning | Time-series + LLM | 0.824 |
| XGBoost (Tabular) | Tree-based | Time-series stats + Annotations | 0.804 |

### Prediction Task Results

| Task | Best Model | AUROC |
|------|------------|-------|
| **Mortality** | Full Feature Fusion | 0.844 |
| **Prolonged LOS** | Full Feature Fusion | 0.844 |
| **30-Day Readmission** | XGBoost | 0.632 |

### Disease-Stratified Models (5-fold CV)

| Condition | N Samples | Mortality | GB AUROC |
|-----------|-----------|-----------|----------|
| AKI | 57,263 | 14.5% | 0.820 ± 0.002 |
| Sepsis | 34,152 | 18.2% | 0.807 ± 0.006 |
| ARDS | 822 | 39.9% | 0.676 ± 0.015 |

---

## Model Details

### 1. Enhanced Reasoning Model (NEW in v2.0)

| Parameter | Value |
|-----------|-------|
| Architecture | XGBoost |
| Feature Count | 46 |
| Key Features | syndrome_detection, disease_timeline, patient_state_space |
| Test AUROC (Mortality) | 0.816 |
| Test AUROC (LOS) | 0.793 |

**New Reasoning Features:**
- `sepsis_detected`, `aki_detected`, `ards_detected`
- `dt_has_sepsis`, `dt_onset_hour`, `dt_deteriorating`
- `rc_evidence_count`, `rc_confidence`
- `pss_hours`, `pss_last_severity`

### 2. Full Feature Fusion (Best Overall)

| Parameter | Value |
|-----------|-------|
| Architecture | XGBoost |
| n_estimators | 100 |
| max_depth | 6 |
| Feature Sets | Vitals + BERT + Concepts + Annotations |
| Test AUROC | 0.844 |

### 3. Enhanced GRU

| Parameter | Value |
|-----------|-------|
| Architecture | GRU + Static Feature Fusion |
| Hidden Dim | 128 |
| Num Layers | 2 |
| Dropout | 0.2 |
| Input Dim (Seq) | 30 (25 physio + 5 LLM) |
| Input Dim (Static) | 4 (annotation features) |

**Training:**
- Optimizer: Adam (lr=0.001)
- Loss: Binary Cross-Entropy
- Early Stopping: patience=10
- 5-Fold GroupKFold CV (by subject_id)

### 4. XGBoost (Tabular)

| Parameter | Value |
|-----------|-------|
| n_estimators | 100 |
| max_depth | 6 |
| learning_rate | 0.1 |
| Features | 35 (time-series mean/std + annotations) |

---

## Feature Sets

### Time-Series Features (per vital)

| Feature | Description |
|---------|-------------|
| `{vital}_mean` | Mean value over 24h |
| `{vital}_std` | Standard deviation |
| `{vital}_min` | Minimum value |
| `{vital}_max` | Maximum value |
| `{vital}_last` | Last recorded value |

### Annotation Features

| Feature | Description |
|---------|-------------|
| `n_supportive` | Count of SUPPORTIVE annotations |
| `n_contradictory` | Count of CONTRADICTORY annotations |
| `n_patterns` | Count of detected patterns |
| `n_conditions` | Count of associated conditions |

### Enhanced Reasoning Features (NEW in v2.0)

| Feature | Description |
|---------|-------------|
| `sepsis_detected` | Syndrome detection result |
| `aki_detected` | AKI detection result |
| `aki_stage` | AKI severity stage (1-3) |
| `dt_has_sepsis` | Disease timeline includes sepsis |
| `dt_onset_hour` | Predicted disease onset hour |
| `dt_deteriorating` | Patient prognosis is deteriorating |
| `rc_evidence_count` | Number of reasoning evidence |
| `rc_confidence` | Reasoning chain confidence |
| `pss_hours` | Hours in patient state-space |

### ClinicalBERT Embedding Features

| Feature | Description |
|---------|-------------|
| `bert_0` to `bert_49` | Top 50 PCA dimensions from 768-dim ClinicalBERT embeddings |
| **Model** | emilyalsentzer/Bio_ClinicalBERT |
| **Pooling** | Mean pooling across notes per stay |

### NER Medical Concept Features

| Feature | Description |
|---------|-------------|
| `concept_DISEASE` | Count of disease entities |
| `concept_DRUG` | Count of drug/medication entities |
| `concept_PROCEDURE` | Count of procedure entities |
| **Model** | spaCy en_core_sci_lg |

---

## Evaluation Metrics

### Discrimination

| Model | AUROC | AUPRC |
|-------|-------|-------|
| Full Feature Fusion | 0.844 | 0.486 |
| Enhanced Reasoning | 0.816 | 0.418 |
| Enhanced GRU | 0.831 | 0.468 |
| XGBoost | 0.804 | 0.409 |

### Calibration

| Model | ECE ↓ | HL p-value |
|-------|-------|------------|
| EarlyFusion_XGBoost | 0.0067 | 0.0001 |
| Tabular_XGBoost | 0.0065 | 0.0382 |
| TextOnly_XGBoost | 0.0015 | 0.0002 |

**Note**: All models show good calibration (ECE < 0.01).

---

## Ablation Studies

### Time Window Sensitivity

| Window | AUROC |
|--------|-------|
| ±6h | 0.777 |
| ±12h | 0.800 |
| ±24h (default) | 0.833 |

### Note Category Importance

| Category | Importance |
|----------|------------|
| **Nursing** | Highest (AUROC 0.638 alone) |
| Radiology | Moderate (AUROC 0.545 alone) |
| Others | Minimal contribution |

---

## Intended Use

- **Primary**: Benchmark for multimodal EHR fusion research
- **Secondary**: Educational demonstration of fusion strategies
- **New**: Clinical reasoning and disease progression modeling

## Limitations

1. **Single-task optimization**: Models optimized for mortality only
2. **Fixed architecture**: No hyperparameter tuning for GRU
3. **Limited fusion strategies**: Only early/late fusion tested
4. **Syndrome Detection**: High Sepsis Recall, Lower AKI Recall

## Ethical Considerations

- Models trained on de-identified data
- Not validated for clinical deployment
- Should not be used for individual patient decisions

---

## Reproducibility

```bash
# Train all baselines
python code/baselines/train_tabular_baselines.py
python code/baselines/train_text_only.py
python code/baselines/train_enhanced_gru.py
python code/baselines/train_fusion.py
python code/baselines/train_enhanced_reasoning.py  # NEW

# Evaluate
python code/baselines/eval_calibration.py
python code/baselines/eval_note_ablation.py
```

---

## Version History

| Version | Date | Changes |
|---------|------|---------|
| v1.0 | 2025-12 | Initial release |
| **v2.0** | **2026-01** | Added enhanced reasoning model, syndrome detection features, disease timeline features |

---

## Citation

```bibtex
@misc{timely-bench-2026,
  title={TIMELY-Bench: A Unified Framework for Multimodal Clinical Reasoning at Scale},
  author={[Author Names]},
  year={2026},
  institution={King's College London, LOPPN Department}
}
```
