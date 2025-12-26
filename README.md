# TIMELY-Bench: Temporal-Textual Alignment Benchmark for Clinical AI

**Version**: 2.0 Final
**Date**: 2025-12

---

## 项目概述

TIMELY-Bench 是一个用于临床AI中时序-文本对齐融合的基准测试框架。本项目基于 MIMIC-IV v3.1 数据集，提供：

1. **多窗口对齐协议** (±6h / ±12h / ±24h)
2. **生理学模式检测** (Sepsis-3, KDIGO AKI, Berlin ARDS)
3. **早期/晚期融合基线模型**
4. **校准度评估** (ECE, Brier Score)

---

## 目录结构

```
TIMELY-Bench_Final/
├── code/                          # 核心代码
│   ├── data_processing/           # 数据处理脚本
│   │   ├── generate_data_splits.py
│   │   ├── aggregate_features.py
│   │   ├── physiology_patterns.py
│   │   ├── alignment_protocols.py
│   │   └── ...
│   └── baselines/                 # 基线模型
│       ├── run_baselines.py
│       ├── run_fusion_baselines.py
│       ├── run_temporal_gru.py
│       └── ...
│
├── data/                          # 数据文件
│   ├── raw/                       # 原始数据
│   │   ├── timeseries.csv
│   │   ├── note_time.csv
│   │   └── ...
│   ├── processed/                 # 处理后数据
│   │   ├── data_windows/          # 时序窗口特征
│   │   │   ├── window_6h/
│   │   │   ├── window_12h/
│   │   │   └── window_24h/
│   │   └── merge_output/          # cohort合并结果
│   ├── splits/                    # 预定义数据分割
│   │   ├── train.csv              # 70% (2102 episodes)
│   │   ├── val.csv                # 15% (451 episodes)
│   │   ├── test.csv               # 15% (447 episodes)
│   │   └── split_summary.json
│   └── llm_features/              # LLM提取特征
│       └── llm_features_deepseek.csv
│
├── episodes/                      # Episode JSON文件
│   └── episodes_core/             # 3000核心episodes
│
├── documentation/                 # 文档
│   ├── SURVEY_TAXONOMY.md         # D1: 文献综述与分类
│   ├── MODEL_CARD.md              # 模型卡片
│   ├── ALIGNMENT_PROTOCOL.md      # 对齐协议说明
│   └── ...
│
├── results/                       # 实验结果
│   └── benchmark_results/
│
└── sql/                           # MIMIC-IV SQL查询
```

---

## 快速开始

### 环境要求

```bash
python >= 3.8
pandas >= 1.5
numpy >= 1.21
scikit-learn >= 1.0
xgboost >= 1.7
pytorch >= 1.12  # 仅GRU模型需要
```

### 安装依赖

```bash
pip install -r requirements.txt
```

### 运行方式 (三选一)

#### 方式1: 一键运行脚本 (推荐)
```bash
# 验证数据完整性
python run_all.py --verify

# 运行完整pipeline
python run_all.py

# 只运行特定实验
python run_all.py --baselines   # XGBoost基线
python run_all.py --fusion      # 融合实验
python run_all.py --gru         # GRU模型
```

#### 方式2: 使用Makefile
```bash
make install     # 安装依赖
make baselines   # 运行XGBoost基线
make fusion      # 运行融合实验
make run-all     # 运行所有实验
```

#### 方式3: 使用Snakemake
```bash
pip install snakemake
snakemake --cores 4 all
```

---

## 🔄 可复现性 (Reproducibility)

本项目提供完整的可复现性支持：

### 固定随机种子
```python
RANDOM_SEED = 42  # 所有实验使用相同随机种子
```

### 预定义数据分割
- `data/splits/train.csv` - 2102 episodes (70%)
- `data/splits/val.csv` - 451 episodes (15%)
- `data/splits/test.csv` - 447 episodes (15%)
- 按 `subject_id` 分组，防止数据泄露

### 自动化工具
| 工具 | 文件 | 说明 |
|------|------|------|
| Python | `run_all.py` | 一键运行脚本 |
| Make | `Makefile` | 标准构建工具 |
| Snakemake | `Snakefile` | 工作流管理 |
| Config | `config.yaml` | 统一配置文件 |

### 在新机器上复现

```bash
# 1. 克隆/复制项目
cd TIMELY-Bench_Final

# 2. 创建虚拟环境 (推荐)
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# 3. 安装依赖
pip install -r requirements.txt

# 4. 验证数据
python run_all.py --verify

# 5. 运行实验
python run_all.py
```

---

## 核心任务

| 任务 | 标签 | 阳性率 | 临床意义 |
|------|------|--------|----------|
| **Mortality** | 院内死亡 | ~10% | 主要预测任务 |
| **Prolonged LOS** | ICU住院≥7天 | ~35% | 资源规划 |
| **Readmission** | 30天ICU再入院 | ~15% | 护理质量指标 |

---

## 评估指标

### 区分度 (Discrimination)
- **AUROC**: ROC曲线下面积
- **AUPRC**: PR曲线下面积 (针对不平衡数据)

### 校准度 (Calibration) - 临床AI核心指标
- **ECE**: Expected Calibration Error (越低越好, 目标<0.10)
- **Brier Score**: 概率预测均方误差

---

## 对齐协议

| 协议 | 窗口 | 描述 | 适用场景 |
|------|------|------|----------|
| D0_daily | 当天 | 同一日历日对齐 | 每日汇总 |
| ±6h | -6h~+6h | 紧密时间窗口 | 急性事件预测 |
| ±12h | -12h~+12h | 中等窗口 | 班次对齐 |
| ±24h | -24h~+24h | 宽松窗口 | 综合上下文 |
| asymmetric | -6h~+2h | 非对称窗口 | 因果建模 |

---

## 数据分割

使用预定义的 subject-level stratified 分割:

```python
# 加载数据分割
import json
with open('data/splits/split_ids.json') as f:
    splits = json.load(f)

train_ids = splits['train']  # 2102 episodes
val_ids = splits['val']      # 451 episodes
test_ids = splits['test']    # 447 episodes
```

分层键: `mortality × has_sepsis × has_aki`

---

## 引用

```bibtex
@misc{timely-bench-2025,
  title={TIMELY-Bench: A Temporal-Textual Alignment Benchmark for Clinical AI},
  author={TIMELY-Bench Team},
  year={2025},
  note={Version 2.0}
}
```

---

## 版本历史

| 版本 | 日期 | 变更 |
|------|------|------|
| 1.0 | 2024-11 | 初始版本 |
| 2.0 | 2025-12 | 添加校准度评估、预定义数据分割、多窗口对齐协议 |

---

*Last Updated: 2025-12-24*
