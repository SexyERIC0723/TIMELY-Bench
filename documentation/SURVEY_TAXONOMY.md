# Survey & Taxonomy: Temporal-Textual Alignment and Fusion in Clinical AI

**TIMELY-Bench v2.0 | D1 Deliverable**

---

## 1. Introduction

### 1.1 Motivation

Electronic Health Records (EHRs) contain rich multimodal information: structured time-series data (vital signs, lab values) and unstructured clinical text (physician notes, radiology reports). Effective fusion of these modalities is crucial for clinical decision support, yet presents significant challenges due to:

- **Temporal misalignment**: Notes are written asynchronously with physiological measurements
- **Semantic heterogeneity**: Different information granularities and representations
- **Missing data**: Incomplete coverage across modalities

### 1.2 Scope

This survey covers:
1. **Temporal alignment strategies** between time-series and text
2. **Fusion architectures** for multimodal clinical data
3. **Benchmark evaluation** methodologies
4. **Gap analysis** and research opportunities

---

## 2. Taxonomy of Alignment Protocols

### 2.1 Time Window-Based Alignment

| Protocol | Window | Description | Use Case |
|----------|--------|-------------|----------|
| **D0 (Same-Day)** | Calendar day | Match notes to same calendar day | Daily summaries |
| **±6h** | -6h to +6h | Tight temporal window | Acute events |
| **±12h** | -12h to +12h | Medium window | Shift-based alignment |
| **±24h** | -24h to +24h | Broad context | Comprehensive analysis |
| **Asymmetric** | -6h to +2h | Predictive alignment | Causal modeling |

### 2.2 Alignment Quality Metrics

```
Quality Categories:
├── EXACT    : |Δt| ≤ 1 hour   (highest relevance)
├── CLOSE    : |Δt| ≤ 3 hours  (high relevance)
├── MODERATE : |Δt| ≤ 12 hours (moderate relevance)
└── DISTANT  : |Δt| > 12 hours (contextual only)
```

### 2.3 Event-Based Alignment

| Strategy | Description | Reference |
|----------|-------------|-----------|
| **Pattern-Triggered** | Align text to detected physiological patterns | TIMELY-Bench |
| **Diagnosis-Anchored** | Align to ICD diagnosis time | Harutyunyan et al., 2019 |
| **Intervention-Based** | Align to treatment initiation | MIMIC-Extract |

### 2.4 Time Unit Granularity

| Unit | Field | Precision | Use Case |
|------|-------|-----------|----------|
| **charttime** | Timestamp | Minute-level | Vital signs, lab results |
| **chartdate** | Date only | Day-level | D0 daily alignment |
| **storetime** | Storage timestamp | Minute-level | Note entry time |

> **Note**: MIMIC-IV uses `charttime` for precise temporal alignment. For D0 (daily) alignment, we aggregate events by calendar date derived from `charttime`.

---

## 3. Taxonomy of Fusion Architectures

### 3.1 Fusion Timing

```
┌─────────────────────────────────────────────────────────┐
│                    FUSION STRATEGIES                     │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  ┌─────────┐    ┌─────────┐    ┌─────────┐             │
│  │  EARLY  │    │  JOINT  │    │  LATE   │             │
│  │ FUSION  │    │ FUSION  │    │ FUSION  │             │
│  └────┬────┘    └────┬────┘    └────┬────┘             │
│       │              │              │                   │
│       ▼              ▼              ▼                   │
│  Feature-Level   Embedding    Decision-Level           │
│  Concatenation   Alignment    Ensemble                 │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

### 3.2 Architecture Categories

| Category | Method | Description | Key Papers |
|----------|--------|-------------|------------|
| **Early Fusion** | Feature Concatenation | Combine raw/extracted features | Rajkomar et al., 2018 |
| | Cross-Modal Attention | Attend across modalities | Xu et al., 2021 |
| **Joint Fusion** | Multimodal Transformer | Joint encoding | BEHRT, Med-BERT |
| | Graph Neural Networks | Relation-aware fusion | Clinical Knowledge Graph |
| **Late Fusion** | Ensemble Voting | Combine predictions | Baseline approach |
| | Stacking | Meta-learner on predictions | TIMELY-Bench |

### 3.3 Text Representation Methods

| Method | Dimension | Source | Pros | Cons |
|--------|-----------|--------|------|------|
| **Bag-of-Words** | Variable | Local | Interpretable | Sparse, no semantics |
| **TF-IDF** | Variable | Local | Better weighting | Still sparse |
| **Word2Vec** | 100-300 | Pre-trained | Dense | No context |
| **BERT** | 768 | Pre-trained | Contextual | Compute-heavy |
| **Clinical BERT** | 768 | Domain-specific | Medical knowledge | Fine-tuning needed |
| **LLM Extraction** | 5-30 | GPT/DeepSeek | Structured, interpretable | API cost |

---

## 4. Survey of Existing Approaches

### 4.1 Clinical Multimodal Benchmarks

| Benchmark | Modalities | Tasks | Size | Alignment |
|-----------|------------|-------|------|-----------|
| **MIMIC-Extract** | Time-series only | 5 tasks | 34K | N/A |
| **eICU Benchmark** | Time-series only | Mortality | 139K | N/A |
| **ClinicalBERT** | Text only | Various | 2M notes | N/A |
| **HAIM** | Image + Tabular | 12 tasks | 45K | Admission-level |
| **TIMELY-Bench** | Time-series + Text | 3 tasks | 3K core | Multi-window |

### 4.2 Key Publications

#### Time-Series Processing
1. **Harutyunyan et al. (2019)** - Multitask learning on MIMIC-III
   - 17 clinical features, 48h observation
   - Benchmarked LSTM, GRU, channel-wise LSTM

2. **MIMIC-Extract (Wang et al., 2020)** - Standardized extraction pipeline
   - Hourly aggregation
   - Imputation and normalization protocols

#### Clinical Text Processing
3. **ClinicalBERT (Alsentzer et al., 2019)** - BERT pre-trained on clinical notes
   - 2M clinical notes from MIMIC-III
   - Showed gains on NER, relation extraction

4. **Med-BERT (Rasmy et al., 2021)** - EHR-specific pre-training
   - Structured diagnosis codes
   - Hierarchical attention

#### Multimodal Fusion
5. **HAIM (Soenksen et al., 2022)** - Holistic AI in Medicine
   - Images + tabular data
   - Shapley-based feature importance

6. **Khadanga et al. (2019)** - Multimodal ICU prediction
   - RNN for time-series + CNN for text
   - Early concatenation fusion

---

## 5. TIMELY-Bench Contribution

### 5.1 Novel Elements

| Aspect | Previous Work | TIMELY-Bench |
|--------|--------------|--------------|
| **Alignment** | Single fixed window | Multi-protocol (5 options) |
| **Validation** | Black-box | LLM-annotated concordance |
| **Patterns** | Ad-hoc thresholds | Evidence-based (PMID refs) |
| **Fusion** | Single method | Early + Late + Ablation |
| **Transparency** | Limited | Full data lineage |

### 5.2 Alignment Protocol Innovation

```
TIMELY-Bench Alignment Pipeline:

Time-Series ──► Pattern Detection ──► Temporal Window ──► Text Matching
     │              (SIRS, KDIGO,        (±6h/±12h/        (Note retrieval)
     │               Berlin)              ±24h)                 │
     │                                                          │
     └──────────────────────────────────────────────────────────┘
                                │
                                ▼
                    LLM Concordance Annotation
                    (SUPPORTIVE / CONTRADICTORY / UNRELATED)
```

### 5.3 Physiology-Grounded Patterns

| Disease | Standard | Key Patterns | PMID |
|---------|----------|--------------|------|
| **Sepsis** | Sepsis-3, SIRS | Fever, hypotension, tachycardia | 26903338 |
| **AKI** | KDIGO | Creatinine rise, oliguria | 25018915 |
| **ARDS** | Berlin Definition | P/F ratio, SpO2 | 22797452 |
| **Critical** | ICU Standards | GCS, severe vitals | Various |

---

## 6. Evaluation Framework

### 6.1 Prediction Metrics

| Metric | Formula | Use Case |
|--------|---------|----------|
| **AUROC** | Area under ROC curve | Discrimination |
| **AUPRC** | Area under PR curve | Imbalanced classes |
| **Brier Score** | Mean squared error | Calibration |
| **F1-Score** | Harmonic mean P/R | Balance |

### 6.2 Calibration Assessment

- **Reliability Diagram**: Predicted vs. actual probability
- **Expected Calibration Error (ECE)**: Weighted bin deviation
- **Hosmer-Lemeshow Test**: Statistical calibration test

### 6.3 Fusion Effectiveness Metrics

| Metric | Description |
|--------|-------------|
| **Unimodal Baseline Gap** | Improvement over single modality |
| **Fusion Gain** | (Multimodal - Best Unimodal) / Best Unimodal |
| **Complementarity Score** | Unique information from each modality |

---

## 7. Gap Analysis and Future Directions

### 7.1 Current Limitations

| Gap | Description | Impact |
|-----|-------------|--------|
| **Dynamic Alignment** | Fixed windows, no adaptive alignment | Suboptimal matching |
| **Cross-Modal Attention** | Limited attention mechanisms | Information loss |
| **Temporal Reasoning** | No explicit time modeling | Causality unclear |
| **Uncertainty Quantification** | Point estimates only | Overconfidence |

### 7.2 Research Opportunities

1. **Learnable Alignment Windows**
   - Attention-based window selection
   - Patient-specific optimal windows

2. **Causal Fusion**
   - Interventional rather than observational
   - Counterfactual predictions

3. **Continual Learning**
   - Adapt to distribution shift
   - Online model updates

4. **Explainable Multimodal AI**
   - Cross-modal attribution
   - Clinical reasoning chains

---

## 8. Recommended Best Practices

### 8.1 Alignment Protocol Selection

```python
def select_alignment_protocol(task_type: str) -> str:
    """Recommended alignment protocol by task type."""
    recommendations = {
        "acute_prediction": "±6h",      # Sepsis onset, AKI
        "daily_assessment": "D0_daily", # Daily rounds
        "comprehensive": "±24h",        # Full context
        "causal_analysis": "asymmetric" # -6h, +2h
    }
    return recommendations.get(task_type, "±12h")
```

### 8.2 Fusion Strategy Selection

| Scenario | Recommended Fusion | Rationale |
|----------|-------------------|-----------|
| Limited text coverage | Late fusion | Robust to missing modality |
| High text quality | Early fusion | Better feature integration |
| Interpretability needed | Late fusion | Modality-specific explanations |
| Maximum performance | Ensemble | Best of both |

---

## 9. Conclusion

This survey establishes a systematic taxonomy for temporal-textual alignment and fusion in clinical AI. Key contributions include:

1. **Five-protocol alignment taxonomy** with evidence-based quality metrics
2. **Three-tier fusion architecture classification** (Early/Joint/Late)
3. **Comprehensive evaluation framework** including calibration
4. **Gap analysis** identifying future research directions

TIMELY-Bench v2.0 implements this taxonomy with:
- Multi-window alignment (±6h, ±12h, ±24h)
- LLM-annotated concordance validation
- Evidence-based physiological patterns (32 templates)
- Standardized train/validation/test splits

---

## References

1. Singer M, et al. (2016). The Third International Consensus Definitions for Sepsis and Septic Shock (Sepsis-3). JAMA. PMID: 26903338
2. Kidney Disease: Improving Global Outcomes (KDIGO). (2012). KDIGO Clinical Practice Guideline for Acute Kidney Injury. PMID: 25018915
3. Ranieri VM, et al. (2012). Acute Respiratory Distress Syndrome: The Berlin Definition. JAMA. PMID: 22797452
4. Harutyunyan H, et al. (2019). Multitask learning and benchmarking with clinical time series data. Scientific Data.
5. Alsentzer E, et al. (2019). Publicly Available Clinical BERT Embeddings. NAACL.
6. Soenksen LR, et al. (2022). Integrated multimodal artificial intelligence framework for healthcare applications. npj Digital Medicine.
7. Wang S, et al. (2020). MIMIC-Extract: A Data Extraction, Preprocessing, and Representation Pipeline for MIMIC-III. MLHC.

---

*Document Version: 1.0 | Created: 2025-12 | TIMELY-Bench v2.0*
