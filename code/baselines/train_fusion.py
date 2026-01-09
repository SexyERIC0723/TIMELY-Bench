"""
Early Fusion 和 Late Fusion 模型
结合时序特征和标注特征的两种融合策略

Early Fusion: 特征级别拼接，然后送入单一模型
Late Fusion: 分别训练时序模型和标注模型，然后集成预测
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

from config import (
    TIMESERIES_FILE, NOTE_TIME_FILE, LLM_FEATURES_FILE, COHORT_FILE,
    RESULTS_DIR, N_FOLDS, RANDOM_STATE, TEST_SIZE, LLM_COLS
)

# 配置
EPISODES_DIR = Path(__file__).parent.parent.parent / 'episodes' / 'episodes_enhanced'
OUTPUT_DIR = RESULTS_DIR / 'fusion_baselines'


def load_annotation_features():
    """从 Episode 加载标注特征"""
    print("加载标注特征...")
    
    episode_files = list(EPISODES_DIR.glob('TIMELY_v2_*.json'))
    annotations = []
    
    for ep_file in tqdm(episode_files, desc="Loading annotations"):
        try:
            with open(ep_file) as f:
                ep = json.load(f)
            
            stay_id = ep.get('stay_id')
            reasoning = ep.get('reasoning', {})
            
            n_supportive = reasoning.get('n_supportive', 0)
            n_contradictory = reasoning.get('n_contradictory', 0)
            n_alignments = reasoning.get('n_alignments', 0)
            
            total_annot = n_supportive + n_contradictory
            supportive_ratio = n_supportive / total_annot if total_annot > 0 else 0.5
            annotation_density = total_annot / n_alignments if n_alignments > 0 else 0
            
            annotations.append({
                'stay_id': stay_id,
                'n_supportive': n_supportive,
                'n_contradictory': n_contradictory,
                'supportive_ratio': supportive_ratio,
                'annotation_density': annotation_density
            })
        except:
            pass
    
    return pd.DataFrame(annotations)


def load_tabular_features():
    """加载表格化的时序特征"""
    print("加载时序特征...")
    
    episode_files = list(EPISODES_DIR.glob('TIMELY_v2_*.json'))
    vitals_cols = ['heart_rate', 'sbp', 'dbp', 'mbp', 'resp_rate', 'temperature', 'spo2']
    
    features = []
    for ep_file in tqdm(episode_files, desc="Loading timeseries"):
        try:
            with open(ep_file) as f:
                ep = json.load(f)
            
            feat = {'stay_id': ep.get('stay_id')}
            ts = ep.get('timeseries', {})
            vitals = ts.get('vitals', [])
            
            if vitals:
                df = pd.DataFrame(vitals)
                for col in vitals_cols:
                    if col in df.columns:
                        values = pd.to_numeric(df[col], errors='coerce').dropna()
                        if len(values) > 0:
                            feat[f'{col}_mean'] = values.mean()
                            feat[f'{col}_std'] = values.std() if len(values) > 1 else 0
                            feat[f'{col}_min'] = values.min()
                            feat[f'{col}_max'] = values.max()
            
            features.append(feat)
        except:
            pass
    
    return pd.DataFrame(features)


def train_early_fusion(X_ts, X_annot, y, groups):
    """Early Fusion: 特征拼接"""
    print("\n=== Early Fusion (XGBoost) ===")
    
    # 拼接特征
    X = np.concatenate([X_ts, X_annot], axis=1)
    
    gss = GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    train_val_idx, test_idx = next(gss.split(X, y, groups=groups))
    
    X_tv, X_test = X[train_val_idx], X[test_idx]
    y_tv, y_test = y[train_val_idx], y[test_idx]
    groups_tv = groups[train_val_idx]
    
    gkf = GroupKFold(n_splits=N_FOLDS)
    fold_results = []
    
    for fold, (tr_idx, val_idx) in enumerate(gkf.split(X_tv, y_tv, groups=groups_tv)):
        X_tr, X_val = X_tv[tr_idx], X_tv[val_idx]
        y_tr, y_val = y_tv[tr_idx], y_tv[val_idx]
        
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X_tr)
        X_val = scaler.transform(X_val)
        
        model = xgb.XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.1,
                                  random_state=RANDOM_STATE, use_label_encoder=False,
                                  eval_metric='logloss', n_jobs=-1)
        model.fit(X_tr, y_tr)
        pred = model.predict_proba(X_val)[:, 1]
        
        auroc = roc_auc_score(y_val, pred)
        fold_results.append(auroc)
        print(f"   Fold {fold+1}: AUROC={auroc:.4f}")
    
    # 测试
    scaler = StandardScaler()
    X_tv_s = scaler.fit_transform(X_tv)
    X_test_s = scaler.transform(X_test)
    
    model = xgb.XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.1,
                              random_state=RANDOM_STATE, use_label_encoder=False,
                              eval_metric='logloss', n_jobs=-1)
    model.fit(X_tv_s, y_tv)
    test_pred = model.predict_proba(X_test_s)[:, 1]
    test_auroc = roc_auc_score(y_test, test_pred)
    
    print(f"\n   CV AUROC: {np.mean(fold_results):.4f} ± {np.std(fold_results):.4f}")
    print(f"   Test AUROC: {test_auroc:.4f}")
    
    return np.mean(fold_results), np.std(fold_results), test_auroc


def train_late_fusion(X_ts, X_annot, y, groups):
    """Late Fusion: 分别训练，然后集成"""
    print("\n=== Late Fusion (Stacking) ===")
    
    gss = GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    train_val_idx, test_idx = next(gss.split(X_ts, y, groups=groups))
    
    X_ts_tv, X_ts_test = X_ts[train_val_idx], X_ts[test_idx]
    X_annot_tv, X_annot_test = X_annot[train_val_idx], X_annot[test_idx]
    y_tv, y_test = y[train_val_idx], y[test_idx]
    groups_tv = groups[train_val_idx]
    
    gkf = GroupKFold(n_splits=N_FOLDS)
    fold_results = []
    
    for fold, (tr_idx, val_idx) in enumerate(gkf.split(X_ts_tv, y_tv, groups=groups_tv)):
        # 时序模型
        scaler_ts = StandardScaler()
        X_ts_tr = scaler_ts.fit_transform(X_ts_tv[tr_idx])
        X_ts_val = scaler_ts.transform(X_ts_tv[val_idx])
        
        model_ts = xgb.XGBClassifier(n_estimators=100, max_depth=6, random_state=RANDOM_STATE,
                                     use_label_encoder=False, eval_metric='logloss', n_jobs=-1)
        model_ts.fit(X_ts_tr, y_tv[tr_idx])
        pred_ts = model_ts.predict_proba(X_ts_val)[:, 1]
        
        # 标注模型
        scaler_annot = StandardScaler()
        X_annot_tr = scaler_annot.fit_transform(X_annot_tv[tr_idx])
        X_annot_val = scaler_annot.transform(X_annot_tv[val_idx])
        
        model_annot = xgb.XGBClassifier(n_estimators=100, max_depth=4, random_state=RANDOM_STATE,
                                        use_label_encoder=False, eval_metric='logloss', n_jobs=-1)
        model_annot.fit(X_annot_tr, y_tv[tr_idx])
        pred_annot = model_annot.predict_proba(X_annot_val)[:, 1]
        
        # 集成
        pred_fusion = 0.7 * pred_ts + 0.3 * pred_annot  # 加权平均
        auroc = roc_auc_score(y_tv[val_idx], pred_fusion)
        fold_results.append(auroc)
        print(f"   Fold {fold+1}: AUROC={auroc:.4f}")
    
    # 测试
    scaler_ts = StandardScaler()
    X_ts_tv_s = scaler_ts.fit_transform(X_ts_tv)
    X_ts_test_s = scaler_ts.transform(X_ts_test)
    
    scaler_annot = StandardScaler()
    X_annot_tv_s = scaler_annot.fit_transform(X_annot_tv)
    X_annot_test_s = scaler_annot.transform(X_annot_test)
    
    model_ts = xgb.XGBClassifier(n_estimators=100, max_depth=6, random_state=RANDOM_STATE,
                                 use_label_encoder=False, eval_metric='logloss', n_jobs=-1)
    model_ts.fit(X_ts_tv_s, y_tv)
    
    model_annot = xgb.XGBClassifier(n_estimators=100, max_depth=4, random_state=RANDOM_STATE,
                                    use_label_encoder=False, eval_metric='logloss', n_jobs=-1)
    model_annot.fit(X_annot_tv_s, y_tv)
    
    test_pred_ts = model_ts.predict_proba(X_ts_test_s)[:, 1]
    test_pred_annot = model_annot.predict_proba(X_annot_test_s)[:, 1]
    test_pred = 0.7 * test_pred_ts + 0.3 * test_pred_annot
    test_auroc = roc_auc_score(y_test, test_pred)
    
    print(f"\n   CV AUROC: {np.mean(fold_results):.4f} ± {np.std(fold_results):.4f}")
    print(f"   Test AUROC: {test_auroc:.4f}")
    
    return np.mean(fold_results), np.std(fold_results), test_auroc


def main():
    print("=" * 60)
    print("Fusion Baselines (Early & Late)")
    print("=" * 60)
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # 加载数据
    ts_df = load_tabular_features()
    annot_df = load_annotation_features()
    
    # 加载 cohort
    cohort = pd.read_csv(COHORT_FILE)
    cohort['stay_id'] = cohort['stay_id'].astype(int)
    
    # 合并
    df = cohort[['stay_id', 'subject_id', 'label_mortality']].merge(
        ts_df, on='stay_id', how='inner'
    ).merge(
        annot_df, on='stay_id', how='inner'
    ).dropna()
    
    print(f"\n合并后样本数: {len(df):,}")
    
    # 准备特征
    ts_cols = [c for c in ts_df.columns if c != 'stay_id']
    annot_cols = ['n_supportive', 'n_contradictory', 'supportive_ratio', 'annotation_density']
    
    X_ts = df[ts_cols].values
    X_ts = np.nan_to_num(X_ts, nan=0.0)
    
    X_annot = df[annot_cols].values
    y = df['label_mortality'].values
    groups = df['subject_id'].values
    
    results = []
    
    # Early Fusion
    cv_auroc, cv_std, test_auroc = train_early_fusion(X_ts, X_annot, y, groups)
    results.append({
        'model': 'Early Fusion',
        'cv_auroc': cv_auroc,
        'cv_std': cv_std,
        'test_auroc': test_auroc
    })
    
    # Late Fusion
    cv_auroc, cv_std, test_auroc = train_late_fusion(X_ts, X_annot, y, groups)
    results.append({
        'model': 'Late Fusion',
        'cv_auroc': cv_auroc,
        'cv_std': cv_std,
        'test_auroc': test_auroc
    })
    
    # 保存结果
    results_df = pd.DataFrame(results)
    results_df.to_csv(OUTPUT_DIR / 'fusion_results.csv', index=False)
    
    print("\n" + "=" * 60)
    print("最终结果")
    print("=" * 60)
    print(results_df.to_string(index=False))


if __name__ == "__main__":
    main()
