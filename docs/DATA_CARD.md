# TIMELY-Bench Data Card

## Dataset Overview

| Field | Value |
|-------|-------|
| **Name** | TIMELY-Bench v1.0 |
| **Source** | MIMIC-IV v2.2 |
| **Access** | PhysioNet Credentialed Access |
| **License** | PhysioNet Credentialed Health Data License |

---

## Cohort Statistics

| Metric | Value |
|--------|-------|
| **Total ICU Stays** | 74,829 |
| **Unique Patients** | 52,417 |
| **Observation Window** | First 24 hours of ICU stay |
| **Time Period** | 2008-2019 |

---

## Prediction Tasks

### Task 1: In-Hospital Mortality

| Metric | Value |
|--------|-------|
| **Definition** | Death during hospital admission |
| **Positive Rate** | ~12.4% |
| **Evaluation Metric** | AUROC, AUPRC |

### Task 2: Prolonged Length of Stay (LOS)

| Metric | Value |
|--------|-------|
| **Definition** | ICU stay > 7 days |
| **Positive Rate** | ~15.2% |
| **Evaluation Metric** | AUROC, AUPRC |

---

## Data Modalities

### 1. Time-Series Data

| Feature Type | Variables | Frequency |
|--------------|-----------|-----------|
| **Vitals** | heart_rate, sbp, dbp, mbp, resp_rate, temperature, spo2 | Hourly |
| **Labs** | glucose, lactate, creatinine, etc. | Event-based |

### 2. Clinical Notes

| Note Type | Description | Avg Count/Stay |
|-----------|-------------|----------------|
| **Nursing** | Nursing progress notes | ~15 |
| **Radiology** | Imaging reports | ~3 |
| **Physician** | Progress notes | ~5 |
| **Discharge Summary** | Summary at discharge | 1 |

### 3. Alignment Annotations

| Annotation Type | Description |
|-----------------|-------------|
| **SUPPORTIVE** | Text evidence supports physiological pattern |
| **CONTRADICTORY** | Text evidence contradicts pattern |
| **UNRELATED** | No meaningful relationship |

---

## Data Splits

| Split | Subjects | Episodes | Ratio |
|-------|----------|----------|-------|
| **Train** | 36,693 | 52,380 | 70% |
| **Validation** | 5,241 | 7,483 | 10% |
| **Test** | 10,483 | 14,966 | 20% |

**Note**: Splits are by patient (subject_id) to prevent data leakage.

---

## Episode JSON Schema

```json
{
  "episode_id": "TIMELY_v2_{stay_id}",
  "stay_id": 12345678,
  "patient": {
    "subject_id": 12345,
    "gender": "M",
    "age": 65
  },
  "timeseries": {
    "vitals": [...],
    "labs": [...]
  },
  "clinical_text": {
    "notes": [...],
    "n_notes": 15
  },
  "reasoning": {
    "detected_patterns": [...],
    "pattern_annotations": [...],
    "n_supportive": 10,
    "n_contradictory": 2
  },
  "labels": {
    "outcome": {
      "mortality": 0,
      "prolonged_los": 1
    }
  }
}
```

---

## Known Limitations

1. **Temporal Coverage**: Limited to first 24 hours
2. **Missing Data**: ~15% missing rate for some vitals
3. **Annotation Quality**: 96% of annotations are rule-based, only 4% are LLM-verified
4. **Population Bias**: Single-center data (Beth Israel Deaconess Medical Center)

---

## Ethical Considerations

- Data is de-identified according to HIPAA Safe Harbor
- PhysioNet Credentialed Access required
- No individual patient re-identification possible

---

## Version History

| Version | Date | Changes |
|---------|------|---------|
| v1.0 | 2025-12 | Initial release |
