# TIMELY-Bench Model Card

**Version**: 2.0
**Created**: 2025-12
**Framework**: scikit-learn, XGBoost, PyTorch

---

## Model Overview

### Purpose
TIMELY-Bench provides baseline models for benchmarking multimodal fusion of clinical time-series and text data for ICU outcome prediction.

### Models Included

| Model | Type | Description |
|-------|------|-------------|
| XGBoost (Tabular) | Gradient Boosting | Baseline for structured features |
| Logistic Regression | Linear | Baseline for text features |
| Early Fusion (XGB) | Hybrid | Concatenated features |
| Late Fusion (Avg) | Ensemble | Average probability fusion |
| Late Fusion (Wt) | Ensemble | Weighted probability fusion (0.7/0.3) |
| Temporal GRU | Deep Learning | Sequence model for time-series |

---

## Training Data

### Source
- **Database**: MIMIC-IV v3.1
- **Cohort**: 3,000 core episodes
- **Split**: 70% train / 15% validation / 15% test
- **Stratification**: Subject-level grouping

### Features

| Feature Type | Count | Description |
|--------------|-------|-------------|
| Vital Signs | 7 | heart_rate, sbp, dbp, mbp, resp_rate, temperature, spo2 |
| Lab Values | 15+ | creatinine, lactate, wbc, hemoglobin, etc. |
| Aggregations | 6 | min, max, mean, first, last, std |
| LLM-Extracted | 5 | pneumonia, edema, pleural_effusion, pneumothorax, tubes_lines |

### Time Windows
- 6 hours (early warning)
- 12 hours (standard)
- 24 hours (comprehensive)

---

## Prediction Tasks

| Task | Label | Prevalence | Clinical Relevance |
|------|-------|------------|-------------------|
| **Mortality** | In-hospital death | ~10% | Primary outcome |
| **Prolonged LOS** | ICU stay ≥ 7 days | ~35% | Resource planning |
| **Readmission** | 30-day ICU return | ~15% | Care quality indicator |

---

## Performance Metrics

### Discrimination
- **AUROC** (Area Under ROC Curve): Primary metric for discrimination ability
- **AUPRC** (Area Under Precision-Recall Curve): Important for imbalanced classes

### Calibration
- **Brier Score**: Mean squared error of probability predictions
- **ECE** (Expected Calibration Error): Alignment of predicted and actual probabilities

### Example Results (24h Window, Mortality, All Cohort)

| Model | AUROC | AUPRC | Brier | ECE |
|-------|-------|-------|-------|-----|
| Tabular Only | 0.81 | 0.38 | 0.08 | 0.05 |
| Text Only | 0.62 | 0.18 | 0.10 | 0.08 |
| Early Fusion | 0.82 | 0.40 | 0.07 | 0.04 |
| Late Fusion | 0.81 | 0.39 | 0.08 | 0.05 |

*Note: Actual values depend on training run.*

---

## Model Hyperparameters

### XGBoost
```python
{
    "n_estimators": 100,
    "max_depth": 5,
    "learning_rate": 0.1,
    "scale_pos_weight": "auto (class imbalance)",
    "eval_metric": "logloss"
}
```

### Logistic Regression
```python
{
    "max_iter": 1000,
    "class_weight": "balanced",
    "solver": "lbfgs"
}
```

### Temporal GRU
```python
{
    "hidden_size": 128,
    "num_layers": 2,
    "dropout": 0.3,
    "learning_rate": 0.001,
    "batch_size": 64,
    "epochs": 50
}
```

---

## Intended Use

### Primary Use Case
- Benchmarking new multimodal fusion methods
- Comparing alignment strategies
- Evaluating clinical NLP approaches

### Users
- Machine learning researchers
- Clinical informatics specialists
- Healthcare AI developers

### Out-of-Scope Uses
- **NOT for clinical decision-making** without validation
- **NOT validated** for real-time patient monitoring
- **NOT generalizable** to non-ICU settings without evaluation

---

## Limitations and Biases

### Data Limitations
1. **Single Center**: MIMIC-IV is from Beth Israel Deaconess Medical Center only
2. **Temporal Scope**: Data from 2008-2019
3. **Text Coverage**: Only ~65% of episodes have radiology notes
4. **Label Noise**: ICD codes may have coding errors

### Model Limitations
1. **No Uncertainty Quantification**: Point estimates only
2. **Static Features**: Aggregated features lose temporal dynamics
3. **Limited Text Representation**: 5-dimensional LLM features
4. **No External Validation**: Performance on other hospitals unknown

### Known Biases
1. **Age Distribution**: ICU patients skew older
2. **Severity Bias**: More severe cases have more data
3. **Documentation Bias**: Clinical notes vary by provider

---

## Ethical Considerations

### Privacy
- All data from MIMIC-IV (de-identified)
- No re-identification possible
- Compliant with PhysioNet data use agreement

### Fairness
- Models trained on specific demographic distribution
- Performance may vary across subgroups
- Recommend subgroup analysis before deployment

### Clinical Safety
- **High stakes application**: mortality prediction
- Miscalibrated predictions can lead to harm
- ECE evaluation essential for clinical trust

---

## Evaluation Recommendations

### Before Deployment
1. Evaluate on local hospital data
2. Perform subgroup analysis (age, race, sex)
3. Assess calibration (ECE < 0.10 recommended)
4. Review with clinical experts

### Monitoring
1. Track prediction drift over time
2. Monitor calibration degradation
3. Collect feedback from clinicians

---

## Citation

If you use TIMELY-Bench models, please cite:

```bibtex
@misc{timely-bench-2025,
  title={TIMELY-Bench: A Temporal-Textual Alignment Benchmark for Clinical AI},
  author={TIMELY-Bench Team},
  year={2025},
  note={Version 2.0}
}
```

---

## Model Governance

### Version History
| Version | Date | Changes |
|---------|------|---------|
| 1.0 | 2024-11 | Initial release |
| 2.0 | 2025-12 | Added calibration metrics, data splits, alignment protocols |

### Maintenance
- **Contact**: [Project Repository]
- **Updates**: Quarterly reviews
- **Issues**: GitHub Issues

---

## Technical Specifications

### Requirements
```
python >= 3.8
scikit-learn >= 1.0
xgboost >= 1.7
pytorch >= 1.12
pandas >= 1.5
numpy >= 1.21
```

### Reproducibility
- Random seed: 42
- Cross-validation: 5-fold GroupKFold
- Train/Val/Test split: Subject-level stratified

---

*Document Version: 1.0 | Last Updated: 2025-12*
