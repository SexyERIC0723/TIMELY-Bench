"""
训练带有 Delta 特征的表格基线模型

Gap-5: 添加时间变化特征 (delta_6h, delta_12h)
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
import json
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

from config import (
    TIMESERIES_FILE, COHORT_FILE,
    RESULTS_DIR, N_FOLDS, RANDOM_STATE, TEST_SIZE
)

EPISODES_DIR = Path(__file__).parent.parent.parent / 'episodes' / 'episodes_enhanced'
OUTPUT_DIR = RESULTS_DIR / 'delta_features'


def extract_features_with_deltas(stay_id: int) -> dict:
    """提取包含 delta 的特征"""
    ep_file = EPISODES_DIR / f'TIMELY_v2_{stay_id}.json'
    if not ep_file.exists():
        return None
    
    try:
        with open(ep_file) as f:
            ep = json.load(f)
    except:
        return None
    
    features = {'stay_id': stay_id}
    
    # 时序数据
    ts = ep.get('timeseries', {})
    vitals_list = ts.get('vitals', [])  # 列表格式
    labs_list = ts.get('labs', [])
    
    # 定义要提取的特征
    vitals_names = ['heart_rate', 'sbp', 'dbp', 'mbp', 'resp_rate', 
                    'temperature', 'spo2', 'gcs', 'urineoutput']
    labs_names = ['creatinine', 'bun', 'sodium', 'potassium', 'bicarbonate',
                  'chloride', 'ph', 'lactate', 'wbc', 'hemoglobin', 
                  'hematocrit', 'platelet', 'glucose']
    
    # 从列表中提取每个 vital 的时间序列
    for vital in vitals_names:
        values = []
        if isinstance(vitals_list, list):
            for record in vitals_list:
                if isinstance(record, dict):
                    v = record.get(vital)
                    if v is not None:
                        values.append(v)
        
        if values:
            features[f'{vital}_mean'] = np.mean(values)
            features[f'{vital}_std'] = np.std(values) if len(values) > 1 else 0
            features[f'{vital}_min'] = np.min(values)
            features[f'{vital}_max'] = np.max(values)
            features[f'{vital}_last'] = values[-1]
            
            # Delta 特征
            features[f'{vital}_delta_total'] = values[-1] - values[0] if len(values) >= 2 else 0
            features[f'{vital}_delta_6h'] = values[-1] - values[-6] if len(values) >= 6 else 0
            features[f'{vital}_delta_12h'] = values[-1] - values[-12] if len(values) >= 12 else 0
        else:
            for suffix in ['_mean', '_std', '_min', '_max', '_last',
                           '_delta_total', '_delta_6h', '_delta_12h']:
                features[f'{vital}{suffix}'] = 0
    
    # 从列表中提取每个 lab 的时间序列
    for lab in labs_names:
        values = []
        if isinstance(labs_list, list):
            for record in labs_list:
                if isinstance(record, dict):
                    v = record.get(lab)
                    if v is not None:
                        values.append(v)
        
        if values:
            features[f'{lab}_mean'] = np.mean(values)
            features[f'{lab}_max'] = np.max(values)
            features[f'{lab}_last'] = values[-1]
            features[f'{lab}_delta_total'] = values[-1] - values[0] if len(values) >= 2 else 0
        else:
            features[f'{lab}_mean'] = 0
            features[f'{lab}_max'] = 0
            features[f'{lab}_last'] = 0
            features[f'{lab}_delta_total'] = 0
    
    # 患者信息
    patient = ep.get('patient', {})
    features['age'] = patient.get('age', 0) or 0
    features['gender_M'] = 1 if patient.get('gender') == 'M' else 0
    
    # 标签
    labels = ep.get('labels', {})
    outcome = labels.get('outcome', {})
    features['mortality'] = outcome.get('mortality', 0) or 0
    features['prolonged_los'] = outcome.get('prolonged_los', 0) or 0
    
    # subject_id for grouping
    features['subject_id'] = patient.get('subject_id')
    
    return features


def main():
    print("=" * 60)
    print("训练带有 Delta 特征的表格基线模型")
    print("=" * 60)
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # 加载 cohort
    cohort = pd.read_csv(COHORT_FILE)
    stay_ids = cohort['stay_id'].tolist()
    
    print(f"提取特征 (含 delta)...")
    features_list = []
    for i, stay_id in enumerate(stay_ids):
        if (i + 1) % 10000 == 0:
            print(f"  {i+1}/{len(stay_ids)}")
        feat = extract_features_with_deltas(stay_id)
        if feat:
            features_list.append(feat)
    
    df = pd.DataFrame(features_list)
    print(f"提取完成: {len(df)} 样本")
    
    # 特征列（排除 id、标签等）
    feature_cols = [c for c in df.columns if c not in 
                    ['stay_id', 'subject_id', 'mortality', 'prolonged_los']]
    
    # 计算 delta 特征数量
    delta_cols = [c for c in feature_cols if 'delta' in c]
    print(f"总特征数: {len(feature_cols)}")
    print(f"Delta 特征数: {len(delta_cols)}")
    
    # 填充 NaN
    df[feature_cols] = df[feature_cols].fillna(0)
    
    X = df[feature_cols].values
    y_mort = df['mortality'].values
    y_los = df['prolonged_los'].values
    groups = df['subject_id'].values
    
    # 训练-测试划分
    gss = GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    train_idx, test_idx = next(gss.split(X, y_mort, groups=groups))
    
    X_train, X_test = X[train_idx], X[test_idx]
    y_mort_train, y_mort_test = y_mort[train_idx], y_mort[test_idx]
    y_los_train, y_los_test = y_los[train_idx], y_los[test_idx]
    
    results = []
    
    # 1. XGBoost Mortality
    print("\n训练 XGBoost (Mortality)...")
    xgb_mort = xgb.XGBClassifier(
        n_estimators=100, max_depth=5, learning_rate=0.1,
        random_state=RANDOM_STATE, use_label_encoder=False, eval_metric='logloss'
    )
    xgb_mort.fit(X_train, y_mort_train)
    pred_mort = xgb_mort.predict_proba(X_test)[:, 1]
    auroc_mort = roc_auc_score(y_mort_test, pred_mort)
    print(f"  XGBoost Mortality AUROC: {auroc_mort:.4f}")
    results.append({'model': 'XGBoost', 'task': 'Mortality', 'auroc': auroc_mort})
    
    # 2. XGBoost LOS
    print("训练 XGBoost (LOS)...")
    xgb_los = xgb.XGBClassifier(
        n_estimators=100, max_depth=5, learning_rate=0.1,
        random_state=RANDOM_STATE, use_label_encoder=False, eval_metric='logloss'
    )
    xgb_los.fit(X_train, y_los_train)
    pred_los = xgb_los.predict_proba(X_test)[:, 1]
    auroc_los = roc_auc_score(y_los_test, pred_los)
    print(f"  XGBoost LOS AUROC: {auroc_los:.4f}")
    results.append({'model': 'XGBoost', 'task': 'LOS', 'auroc': auroc_los})
    
    # 3. 特征重要性 (重点看 delta 特征)
    print("\nDelta 特征重要性 (Top 10):")
    importances = xgb_mort.feature_importances_
    feat_importance = list(zip(feature_cols, importances))
    feat_importance.sort(key=lambda x: x[1], reverse=True)
    
    delta_importance = [(f, i) for f, i in feat_importance if 'delta' in f]
    for f, imp in delta_importance[:10]:
        print(f"  {f}: {imp:.4f}")
    
    # 保存结果
    results_df = pd.DataFrame(results)
    results_df.to_csv(OUTPUT_DIR / 'delta_features_results.csv', index=False)
    
    # 保存特征重要性
    importance_df = pd.DataFrame(feat_importance, columns=['feature', 'importance'])
    importance_df.to_csv(OUTPUT_DIR / 'feature_importance.csv', index=False)
    
    print("\n" + "=" * 60)
    print("结果")
    print("=" * 60)
    print(results_df.to_string(index=False))
    print(f"\n保存到: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
