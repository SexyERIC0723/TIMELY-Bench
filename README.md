# TIMELY-Bench

Multimodal EHR fusion benchmark for clinical reasoning.

[![Python](https://img.shields.io/badge/Python-3.12+-green.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org)

English | [中文](README_zh.md)

## Current Status

Total Episodes: **74,829** | Enhanced: **74,711** | LLM Timelines: **74,711**

Last updated: January 2026

## What it does

TIMELY-Bench is a reproducible benchmark for multimodal EHR fusion. It:

- Curates benchmark-ready cohorts from MIMIC-IV with transparent alignment protocols
- Implements LLM-guided probabilistic disease timelines and clinical reasoning
- Provides evaluation suite for clinical prediction tasks
- Releases data schemas, code, and documentation for the community

## Results

**Prediction Tasks**

| Task | AUROC |
|------|-------|
| Mortality | 0.844 |
| Prolonged LOS | 0.844 |
| 30-Day Readmission | 0.632 |

**Disease-Stratified**

| Disease | AUROC |
|---------|-------|
| AKI | 0.820 |
| Sepsis | 0.807 |
| ARDS | 0.676 |

Alignment window ±24h gives best performance (AUROC 0.833).

---

## New Features

**LLM-Guided Disease Timelines**
- 74,711 episodes processed with DeepSeek API
- Probabilistic disease progression tracking
- Onset hour prediction and prognosis assessment

**Reasoning Chain**
- Syndrome detection (Sepsis F1: 85.3%, AKI F1: 68.4%)
- Rule-based diagnostic reasoning
- Patient state-space reconstruction (48-hour vectors)

**Enhanced Episode Structure**
- `patient_state_space`: Hourly state vectors
- `reasoning.syndrome_detection`: Clinical criteria detection
- `reasoning.reasoning_chain`: Diagnostic evidence chain
- `reasoning.disease_timeline`: LLM-generated progression

## Project Structure

```
TIMELY-Bench_Final/
├── code/
│   ├── baselines/              # Model training scripts
│   │   ├── train_tabular_baselines.py
│   │   ├── train_los_baselines.py
│   │   ├── train_readmission_baselines.py
│   │   ├── train_differential_diagnosis.py
│   │   └── ...
│   ├── data_processing/        # Data processing pipeline
│   │   ├── generate_disease_timeline.py
│   │   ├── generate_state_space.py
│   │   ├── generate_reasoning_chain.py
│   │   ├── syndrome_detector.py
│   │   └── ...
│   └── config.py
├── data/
│   └── processed/
│       ├── disease_timelines/   # LLM-generated timelines
│       ├── hidden_features/     # Latent diagnostic features
│       └── medcat_umls/         # UMLS concept extraction
├── episodes/
│   └── episodes_enhanced/       # 74,829 Enhanced Episodes
├── results/                     # Training results
└── docs/                        # Documentation
```

---

## Quick Start

### Prerequisites

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install pandas numpy scikit-learn xgboost torch tqdm openai aiohttp
```

### Run Baselines

```bash
cd code/baselines

# Train mortality prediction
python train_los_baselines.py

# Train readmission prediction
python train_readmission_baselines.py

# Train differential diagnosis
python train_differential_diagnosis.py
```

### Generate Disease Timelines (requires API key)

```bash
export DEEPSEEK_API_KEY='your-api-key'
python code/data_processing/generate_timeline_concurrent.py
```

---

## Benchmark Tasks

| Task | Definition | Positive Rate |
|------|------------|---------------|
| **In-Hospital Mortality** | Death during hospital stay | ~12.4% |
| **Prolonged LOS** | ICU stay > 7 days | ~15.2% |
| **30-Day Readmission** | Readmission within 30 days | ~8.5% |

---

## Documentation

- [Data Card](docs/DATA_CARD.md) - Dataset description and statistics
- [Alignment Protocol Card](docs/ALIGNMENT_PROTOCOL_CARD.md) - Time alignment details
- [Model Card](docs/MODEL_CARD.md) - Baseline model specifications
- [Results Summary](docs/RESULTS_SUMMARY.md) - Comprehensive results

---

## Citation

```bibtex
@misc{timely-bench-2026,
  title={TIMELY-Bench: A Unified Framework for Multimodal 
         Clinical Reasoning at Scale},
  author={[Author Name]},
  year={2026},
  institution={King's College London}
}
```

---

## License

This project uses MIMIC-IV data, which requires PhysioNet Credentialed Access.

---

## Acknowledgments

- MIMIC-IV Database (PhysioNet)
- King's College London, LOPPN Department
