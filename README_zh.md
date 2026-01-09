# TIMELY-Bench

多模态临床时序数据融合基准

[![Python](https://img.shields.io/badge/Python-3.12+-green.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org)

[English](README.md) | 中文

## 项目进展

总Episodes数: **74,829** | 增强版: **74,711** | LLM时间线: **74,711**

最近更新: 2026年1月

## 这个项目做什么

简单来说，TIMELY-Bench是一个做临床多模态数据融合的benchmark。我们把MIMIC-IV的时序指标和临床笔记对齐起来，然后用大模型生成疾病发展时间线，最后在死亡率预测、住院时长预测这些任务上跑baseline。

主要贡献：
- 从MIMIC-IV整理了可直接用的队列数据，对齐方式都是透明的
- 用LLM生成了疾病演变时间线和推理链
- 提供了完整的评估框架和baseline实现
- 代码、数据schema、文档全开源

## 实验结果

**预测任务**

| 任务 | AUROC |
|------|-------|
| 死亡率预测 | 0.844 |
| 住院>7天 | 0.844 |
| 30天再入院 | 0.632 |

**按疾病分层**

| 疾病 | AUROC |
|------|-------|
| AKI | 0.820 |
| Sepsis | 0.807 |
| ARDS | 0.676 |

时间窗口±24h效果最好（AUROC 0.833）。

## 新功能

**LLM疾病时间线**

用DeepSeek API处理了74,711条episode，给每个病人生成了疾病发展的概率时间线，包括发病时间估计和预后评估。

**推理链**

做了syndrome detection（Sepsis F1: 85.3%，AKI F1: 68.4%），基于规则的诊断推理，还有48小时的病人状态空间重建。

**增强的Episode结构**

每个episode里现在都有：
- `patient_state_space`: 逐小时的状态向量
- `reasoning.syndrome_detection`: 临床诊断标准检测
- `reasoning.reasoning_chain`: 诊断证据链
- `reasoning.disease_timeline`: LLM生成的疾病进展

## 目录结构

```
TIMELY-Bench_Final/
├── code/
│   ├── baselines/              # 模型训练脚本
│   ├── data_processing/        # 数据处理流程
│   └── config.py
├── data/
│   └── processed/
│       ├── disease_timelines/   # LLM生成的时间线
│       ├── hidden_features/     # 隐特征
│       └── medcat_umls/         # UMLS概念抽取
├── episodes/
│   └── episodes_enhanced/       # 74,829个增强Episodes
├── results/                     # 训练结果
└── docs/                        # 文档
```

## 快速开始

### 环境配置

```bash
python -m venv venv
source venv/bin/activate
pip install pandas numpy scikit-learn xgboost torch tqdm openai aiohttp
```

### 跑Baseline

```bash
cd code/baselines

# 死亡率预测
python train_los_baselines.py

# 再入院预测
python train_readmission_baselines.py

# 差分诊断
python train_differential_diagnosis.py
```

### 生成疾病时间线（需要API key）

```bash
export DEEPSEEK_API_KEY='your-api-key'
python code/data_processing/generate_timeline_concurrent.py
```

## Benchmark任务定义

| 任务 | 定义 | 阳性率 |
|------|------|--------|
| 院内死亡 | 住院期间死亡 | ~12.4% |
| 延长住院 | ICU住院>7天 | ~15.2% |
| 30天再入院 | 30天内再次入院 | ~8.5% |

## 文档

- [数据卡](docs/DATA_CARD.md) - 数据集描述和统计
- [对齐协议](docs/ALIGNMENT_PROTOCOL_CARD.md) - 时间对齐细节
- [模型卡](docs/MODEL_CARD.md) - Baseline模型说明
- [结果汇总](docs/RESULTS_SUMMARY.md) - 完整结果

## 引用

```bibtex
@misc{timely-bench-2026,
  title={TIMELY-Bench: A Unified Framework for Multimodal 
         Clinical Reasoning at Scale},
  author={[Author Name]},
  year={2026},
  institution={King's College London}
}
```

## 许可

本项目使用MIMIC-IV数据，需要PhysioNet认证访问权限。

## 致谢

- MIMIC-IV数据库 (PhysioNet)
- King's College London LOPPN组
