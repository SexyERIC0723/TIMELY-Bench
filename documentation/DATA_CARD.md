# TIMELY-Bench Data Card

## Dataset Identity

| Field | Value |
|-------|-------|
| **Name** | TIMELY-Bench |
| **Version** | 2.0 |
| **Created** | 2025-12 |
| **Source** | MIMIC-IV v3.1 |
| **License** | PhysioNet Credentialed Health Data License |

---

## Dataset Overview

### Purpose
TIMELY-Bench is designed to benchmark multimodal fusion methods that combine structured EHR time-series with clinical notes for ICU outcome prediction.

### Cohort Definition

| Criterion | Value |
|-----------|-------|
| Age | > 18 years |
| ICU Stay | > 24 hours |
| Total Patients | ~74,000 |

### Disease Subcohorts

| Cohort | Definition | Size |
|--------|------------|------|
| All | All eligible ICU admissions | ~74,000 |
| Sepsis | ICD codes OR Sepsis-3 criteria | ~34,000 |
| AKI | ICD codes OR KDIGO criteria | ~57,000 |
| ARDS | ICD codes (J80) | ~800 |

---

## Features

### Structured Features (Tabular)

| Category | Features | Examples |
|----------|----------|----------|
| Vitals | 7 | heart_rate, sbp, dbp, temperature, resp_rate, spo2 |
| Labs | 15+ | lactate, creatinine, bun, wbc, hemoglobin, platelet |
| Scores | 3 | gcs_min, urineoutput, charlson |
| Aggregations | 6 | min, max, mean, first, last, std |
| Missingness | Yes | _missing flags for each feature |

### Text Features (LLM-extracted)

| Feature | Description | Values |
|---------|-------------|--------|
| pneumonia | Presence of pneumonia | 0, 1 |
| edema | Presence of pulmonary edema | 0, 1 |
| pleural_effusion | Presence of pleural effusion | 0, 1 |
| pneumothorax | Presence of pneumothorax | 0, 1 |
| tubes_lines | Presence of tubes/lines | 0, 1 |

**Source**: Radiology reports within observation window
**Extraction**: DeepSeek V3 with structured prompting

---

## Prediction Tasks

| Task | Label | Positive Rate | Difficulty |
|------|-------|---------------|------------|
| Mortality | In-hospital death | ~10% | Medium |
| Prolonged LOS | ICU stay ≥ 7 days | ~35% | Medium |
| Readmission | 30-day ICU readmission | ~15% | Hard |

---

## Time Windows

| Window | Hours | Use Case |
|--------|-------|----------|
| 6h | 0-6 | Early warning |
| 12h | 0-12 | Standard |
| 24h | 0-24 | Comprehensive |

---

## Data Splits

- **Method**: 5-fold GroupKFold
- **Grouping**: By subject_id (prevents same patient in train/test)
- **Stratification**: Not applied (GroupKFold limitation)

---

## Known Limitations

1. **Single Center**: Data from Beth Israel Deaconess Medical Center only
2. **Text Coverage**: Not all patients have radiology reports in observation window
3. **Label Noise**: ICD codes may have coding errors
4. **Class Imbalance**: Mortality is relatively rare (~10%)

---

## Ethical Considerations

- Data is de-identified per HIPAA Safe Harbor
- Access requires PhysioNet credentialing and CITI training
- No individual patient can be re-identified from released features

---

## Updates

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | 2024-10 | Initial release (24h window, mortality only) |
| 2.0 | 2025-01 | Multi-window, multi-task, disease subcohorts |

