# TIMELY-Bench

**A Benchmark for Time-Aligned Fusion of Clinical Time-Series and Notes in MIMIC**

[![License](https://img.shields.io/badge/License-PhysioNet-blue.svg)](https://physionet.org/)
[![Python](https://img.shields.io/badge/Python-3.12+-green.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org)

---

## 🎯 Overview

TIMELY-Bench is a reproducible benchmark for multimodal EHR fusion that:

1. **Curates** benchmark-ready cohorts from MIMIC-IV with transparent alignment protocols
2. **Implements** lightweight baselines and unified metrics for fair comparison
3. **Releases** data schemas, code, and documentation for the community

---

## 📊 Key Results

| Model | Mortality AUROC | Description |
|-------|-----------------|-------------|
| **Enhanced GRU** | **0.831** | Time-series + LLM + Annotations |
| Temporal GRU | 0.824 | Time-series + LLM |
| XGBoost (Tabular) | 0.804 | Tabular features + Annotations |
| Early Fusion | 0.779 | Feature concatenation |
| Text-only | 0.759 | Annotation features only |

---

## 📁 Project Structure

```
TIMELY-Bench_Final/
├── code/
│   ├── baselines/              # Model training scripts
│   │   ├── train_tabular_baselines.py
│   │   ├── train_text_only.py
│   │   ├── train_enhanced_gru.py
│   │   ├── train_fusion.py
│   │   ├── train_aligner_comparison.py
│   │   ├── eval_calibration.py
│   │   └── eval_note_ablation.py
│   ├── data_processing/        # Data processing pipeline
│   │   ├── episode_builder.py
│   │   ├── pattern_detector.py
│   │   └── smart_rule_matcher_full.py
│   └── config.py               # Configuration
├── data/
│   └── processed/              # Processed data files
├── episodes/
│   └── episodes_all/           # 74,829 Episode JSONs
├── results/                    # Training results
│   ├── tabular_baselines/
│   ├── text_only_baselines/
│   ├── enhanced_gru/
│   ├── fusion_baselines/
│   ├── aligner_comparison/
│   ├── calibration/
│   └── note_ablation/
└── docs/                       # Documentation
    ├── DATA_CARD.md
    ├── ALIGNMENT_PROTOCOL_CARD.md
    └── MODEL_CARD.md
```

---

## 🚀 Quick Start

### Prerequisites

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install pandas numpy scikit-learn xgboost torch tqdm
```

### Run Baselines

```bash
cd code/baselines

# Train tabular baselines (XGBoost, LR)
python train_tabular_baselines.py

# Train text-only model
python train_text_only.py

# Train enhanced GRU
python train_enhanced_gru.py

# Train fusion models
python train_fusion.py

# Run aligner comparison (±6h/±12h/±24h)
python train_aligner_comparison.py
```

### Evaluate

```bash
# Calibration metrics (ECE, Hosmer-Lemeshow)
python eval_calibration.py

# Note category ablation
python eval_note_ablation.py
```

---

## 📈 Benchmark Tasks

| Task | Definition | Positive Rate |
|------|------------|---------------|
| **In-Hospital Mortality** | Death during hospital stay | ~12.4% |
| **Prolonged LOS** | ICU stay > 7 days | ~15.2% |

---

## 🔬 Alignment Windows

| Window | AUROC | Recommendation |
|--------|-------|----------------|
| ±6h | 0.777 | High precision, low coverage |
| ±12h | 0.800 | Balanced |
| **±24h** | **0.833** | Best performance |

---

## 📄 Documentation

- [Data Card](docs/DATA_CARD.md) - Dataset description and statistics
- [Alignment Protocol Card](docs/ALIGNMENT_PROTOCOL_CARD.md) - Time alignment details
- [Model Card](docs/MODEL_CARD.md) - Baseline model specifications

---

## 📊 Results Files

| File | Description |
|------|-------------|
| `results/tabular_baselines/tabular_results.csv` | XGBoost/LR results |
| `results/text_only_baselines/text_only_results.csv` | Text-only results |
| `results/enhanced_gru/enhanced_gru_results.csv` | Enhanced GRU results |
| `results/fusion_baselines/fusion_results.csv` | Early/Late fusion results |
| `results/aligner_comparison/aligner_results.csv` | Window comparison |
| `results/calibration/calibration_results.csv` | ECE/HL metrics |
| `results/note_ablation/note_ablation_results.csv` | Note category ablation |

---

## 📜 Citation

```bibtex
@misc{timely-bench-2025,
  title={TIMELY-Bench: A Benchmark for Time-Aligned Fusion of 
         Clinical Time-Series and Notes in MIMIC},
  author={[Author Name]},
  year={2025},
  institution={King's College London}
}
```

---

## 📝 License

This project uses MIMIC-IV data, which requires PhysioNet Credentialed Access.

---

## 🙏 Acknowledgments

- MIMIC-IV Database (PhysioNet)
- King's College London, Department of Informatics
