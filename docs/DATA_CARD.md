# TIMELY-Bench Data Card

## Dataset Overview

| Field | Value |
|-------|-------|
| **Name** | TIMELY-Bench v2.0 |
| **Source** | MIMIC-IV v2.2 |
| **Access** | PhysioNet Credentialed Access |
| **License** | PhysioNet Credentialed Health Data License |
| **Last Updated** | January 2026 |

## Cohort Statistics

| Metric | Value |
|--------|-------|
| **Total ICU Stays** | 74,829 |
| **Enhanced Episodes** | 74,711 |
| **Unique Patients** | 52,417 |
| **Observation Window** | First 24 hours of ICU stay |
| **Time Period** | 2008-2019 |

## Prediction Tasks

### Task 1: In-Hospital Mortality

| Metric | Value |
|--------|-------|
| **Definition** | Death during hospital admission |
| **Positive Rate** | ~12.4% |
| **Best AUROC** | 0.844 |

### Task 2: Prolonged Length of Stay (LOS)

| Metric | Value |
|--------|-------|
| **Definition** | ICU stay > 7 days |
| **Positive Rate** | ~15.2% |
| **Best AUROC** | 0.844 |

### Task 3: 30-Day Readmission

| Metric | Value |
|--------|-------|
| **Definition** | Readmission within 30 days |
| **Positive Rate** | ~8.5% |
| **Best AUROC** | 0.632 |

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
| **Lab Comments** | Laboratory comments | ~5 |
| **Discharge Summary** | Summary at discharge | 1 |

### 3. Alignment Annotations

| Annotation Type | Description |
|-----------------|-------------|
| **SUPPORTIVE** | Text evidence supports physiological pattern |
| **CONTRADICTORY** | Text evidence contradicts pattern |
| **UNRELATED** | No meaningful relationship |

### 4. Enhanced Reasoning Features (NEW in v2.0)

| Feature | Description |
|---------|-------------|
| **syndrome_detection** | Sepsis/AKI/ARDS detection based on clinical criteria |
| **reasoning_chain** | Diagnostic evidence chain with confidence |
| **disease_timeline** | LLM-generated disease progression timeline |
| **patient_state_space** | 48-hour state vectors for each patient |

## Data Splits

| Split | Subjects | Episodes | Ratio |
|-------|----------|----------|-------|
| **Train** | 36,693 | 52,380 | 70% |
| **Validation** | 5,241 | 7,483 | 10% |
| **Test** | 10,483 | 14,966 | 20% |

**Note**: Splits are by patient (subject_id) to prevent data leakage.

## Episode JSON Schema (v2.0)

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
  "conditions": ["sepsis", "aki"],
  "patient_state_space": [...],
  "reasoning": {
    "detected_patterns": [...],
    "pattern_annotations": [...],
    "n_supportive": 10,
    "n_contradictory": 2,
    "syndrome_detection": {
      "sepsis": {"detected": true, "sirs_count": 3},
      "aki": {"detected": true, "stage": 2},
      "ards": {"detected": false}
    },
    "reasoning_chain": {
      "evidence": [...],
      "confidence": 0.85
    },
    "disease_timeline": {
      "primary_disease": "sepsis",
      "onset_hour": 4,
      "phases": [...],
      "prognosis": "deteriorating"
    }
  },
  "labels": {
    "outcome": {
      "mortality": 0,
      "prolonged_los": 1
    },
    "has_sepsis": true,
    "has_aki": true,
    "has_ards": false
  }
}
```

## Disease Distribution

| Disease | Count | Percentage |
|---------|-------|------------|
| **AKI** | 28,344 | 37.9% |
| **Sepsis** | 18,759 | 25.1% |
| **Sepsis + AKI** | 15,338 | 20.5% |
| **None** | 12,241 | 16.4% |

## Syndrome Detection Performance

| Disease | Precision | Recall | F1 Score |
|---------|-----------|--------|----------|
| **Sepsis** | 75.8% | 97.4% | 85.3% |
| **AKI** | 97.7% | 52.6% | 68.4% |

## Known Limitations

1. **Temporal Coverage**: Limited to first 24 hours
2. **Missing Data**: ~15% missing rate for some vitals
3. **Annotation Quality**: 96% rule-based, 4% LLM-verified
4. **Population Bias**: Single-center data (Beth Israel Deaconess)
5. **Syndrome Detection**: High Sepsis Recall, Lower AKI Recall

## Ethical Considerations

- Data is de-identified according to HIPAA Safe Harbor
- PhysioNet Credentialed Access required
- No individual patient re-identification possible

## Version History

| Version | Date | Changes |
|---------|------|---------|
| v1.0 | 2025-12 | Initial release |
| **v2.0** | **2026-01** | Added syndrome_detection, reasoning_chain, disease_timeline, patient_state_space |
