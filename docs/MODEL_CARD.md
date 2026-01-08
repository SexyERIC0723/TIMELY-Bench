# Model Card: TIMELY-Bench Baselines

## Overview

This document describes the baseline models implemented in TIMELY-Bench for benchmarking multimodal EHR fusion.

---

## Models Summary

| Model | Type | Input | Test AUROC (Mortality) |
|-------|------|-------|------------------------|
| **Full Feature Fusion** | Tree-based | TS + Annotations + BERT + Concepts | **0.844** |
| BERT + Annotation | Tree-based | TS + Annotations + BERT | 0.840 |
| Enhanced GRU | Deep Learning | Time-series + LLM + Annotations | 0.831 |
| Temporal GRU | Deep Learning | Time-series + LLM | 0.824 |
| XGBoost (Tabular) | Tree-based | Time-series stats + Annotations | 0.804 |
| Early Fusion | Tree-based | All features concatenated | 0.779 |
| Text-only | Tree-based | Annotation features only | 0.759 |

### Disease-Stratified Models (5-fold CV)

| Condition | N Samples | Mortality | GB AUROC |
|-----------|-----------|-----------|----------|
| AKI | 57,263 | 14.5% | 0.820 ± 0.002 |
| Sepsis | 34,152 | 18.2% | 0.807 ± 0.006 |
| ARDS | 822 | 39.9% | 0.676 ± 0.015 |

---

## Model Details

### 1. Enhanced GRU (Best Performance)

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

### 2. Temporal GRU

| Parameter | Value |
|-----------|-------|
| Architecture | Standard GRU |
| Hidden Dim | 128 |
| Num Layers | 2 |
| Dropout | 0.2 |
| Input Dim | 30 (25 physio + 5 LLM) |

### 3. XGBoost (Tabular)

| Parameter | Value |
|-----------|-------|
| n_estimators | 100 |
| max_depth | 6 |
| learning_rate | 0.1 |
| Features | 35 (time-series mean/std + annotations) |

### 4. Logistic Regression

| Parameter | Value |
|-----------|-------|
| max_iter | 1000 |
| Regularization | L2 (default) |

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

### ClinicalBERT Embedding Features (New)

| Feature | Description |
|---------|-------------|
| `bert_0` to `bert_49` | Top 50 PCA dimensions from 768-dim ClinicalBERT embeddings |
| **Model** | emilyalsentzer/Bio_ClinicalBERT |
| **Pooling** | Mean pooling across notes per stay |
| **Original dim** | 768 → 50 (dimensionality reduction) |

### NER Medical Concept Features (New)

| Feature | Description |
|---------|-------------|
| `concept_DISEASE` | Count of disease entities |
| `concept_DRUG` | Count of drug/medication entities |
| `concept_PROCEDURE` | Count of procedure entities |
| `concept_ANATOMICAL` | Count of anatomical entities |
| `concept_*` (40 dim) | Various medical concept categories |
| **Model** | spaCy en_core_sci_lg |


---

## Evaluation Metrics

### Discrimination

| Model | AUROC | AUPRC |
|-------|-------|-------|
| Enhanced GRU | 0.831 | 0.468 |
| Temporal GRU | 0.824 | 0.455 |
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

## Limitations

1. **Single-task optimization**: Models optimized for mortality only
2. **Fixed architecture**: No hyperparameter tuning for GRU
3. **Limited fusion strategies**: Only early/late fusion tested
4. **No cross-modal attention**: Missing attention mechanisms

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
python code/baselines/train_aligner_comparison.py

# Evaluate
python code/baselines/eval_calibration.py
python code/baselines/eval_note_ablation.py
```

---

## Citation

```bibtex
@misc{timely-bench-2025,
  title={TIMELY-Bench: A Benchmark for Time-Aligned Fusion of Clinical Time-Series and Notes},
  author={[Author Names]},
  year={2025},
  institution={King's College London}
}
```
