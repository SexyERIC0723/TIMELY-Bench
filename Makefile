# TIMELY-Bench v2.0 Makefile
# ==========================
# 简单的自动化构建工具

.PHONY: all install baselines fusion gru clean help

# Python解释器
PYTHON = python3

# 默认目标
all: baselines fusion

# 安装依赖
install:
	pip install -r requirements.txt

# 运行XGBoost基线
baselines:
	@echo "🚀 Running XGBoost baselines..."
	$(PYTHON) code/baselines/run_baselines.py

# 运行融合实验
fusion:
	@echo "🚀 Running Fusion experiments..."
	$(PYTHON) code/baselines/run_fusion_baselines.py

# 运行GRU模型
gru:
	@echo "🚀 Running GRU models..."
	$(PYTHON) code/baselines/run_temporal_gru.py

# 运行所有实验
run-all: baselines fusion gru

# 验证数据
verify:
	@echo "🔍 Verifying data integrity..."
	$(PYTHON) -c "import pandas as pd; \
		print('Train:', len(pd.read_csv('data/splits/train.csv'))); \
		print('Val:', len(pd.read_csv('data/splits/val.csv'))); \
		print('Test:', len(pd.read_csv('data/splits/test.csv')))"

# 清理结果
clean:
	rm -rf results/benchmark_results/*.csv
	@echo "✅ Cleaned results"

# 帮助信息
help:
	@echo "TIMELY-Bench v2.0 Makefile"
	@echo "=========================="
	@echo ""
	@echo "Commands:"
	@echo "  make install    - Install Python dependencies"
	@echo "  make baselines  - Run XGBoost baselines"
	@echo "  make fusion     - Run Fusion experiments"
	@echo "  make gru        - Run GRU models"
	@echo "  make run-all    - Run all experiments"
	@echo "  make verify     - Verify data integrity"
	@echo "  make clean      - Clean results"
	@echo "  make help       - Show this help"
