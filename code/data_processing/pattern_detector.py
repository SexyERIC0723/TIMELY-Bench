"""
Pattern Detection Engine
在时序数据中检测临床模式，输出带时间戳的模式事件

核心功能：
1. 加载时序数据
2. 对每个患者检测所有适用的模式
3. 输出模式事件列表 (patient_id, hour, pattern_name, value, severity)
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import json
import os

from config import (
    TIMESERIES_FILE, COHORT_FILE, PATTERN_DETECTION_DIR
)

# 导入模式模板
from pattern_templates import (
    PatternTemplate, PatternType, Direction, DiseasePatternSet,
    PATTERN_REGISTRY, get_all_patterns, get_feature_to_patterns_mapping
)

# ==========================================
# 配置
# ==========================================
OUTPUT_DIR = PATTERN_DETECTION_DIR

# ==========================================
# 1. 检测到的模式事件
# ==========================================

@dataclass
class DetectedPattern:
    """检测到的模式实例"""
    stay_id: int
    hour: int                    # 检测到的时间点
    pattern_name: str
    pattern_type: str
    disease: str
    feature: str
    value: float                 # 实际值
    threshold: Optional[float]   # 阈值
    severity: str
    description: str
    
    def to_dict(self):
        return {
            'stay_id': self.stay_id,
            'hour': self.hour,
            'pattern_name': self.pattern_name,
            'pattern_type': self.pattern_type,
            'disease': self.disease,
            'feature': self.feature,
            'value': self.value,
            'threshold': self.threshold,
            'severity': self.severity,
            'description': self.description,
        }

# ==========================================
# 2. 模式检测器
# ==========================================

class PatternDetector:
    """模式检测引擎"""
    
    def __init__(self, patterns: Dict[str, DiseasePatternSet]):
        self.patterns = patterns
        self.feature_mapping = get_feature_to_patterns_mapping()
        
        # 特征名映射（时序数据列名 -> 模板特征名）
        self.feature_name_map = {
            'heart_rate_mean': 'heart_rate',
            'heart_rate_min': 'heart_rate',
            'heart_rate_max': 'heart_rate',
            'sbp_mean': 'sbp',
            'sbp_min': 'sbp',
            'sbp_max': 'sbp',
            'dbp_mean': 'dbp',
            'mbp_mean': 'mbp',
            'mbp_min': 'mbp',
            'resp_rate_mean': 'resp_rate',
            'resp_rate_max': 'resp_rate',
            'temperature_mean': 'temperature',
            'temperature_max': 'temperature',
            'temperature_min': 'temperature',
            'spo2_mean': 'spo2',
            'spo2_min': 'spo2',
            'glucose_mean': 'glucose',
            'lactate_max': 'lactate',
            'lactate_mean': 'lactate',
            'creatinine_max': 'creatinine',
            'creatinine_mean': 'creatinine',
            'bun_max': 'bun',
            'bun_mean': 'bun',
            'potassium_max': 'potassium',
            'potassium_mean': 'potassium',
            'sodium_mean': 'sodium',
            'hemoglobin_min': 'hemoglobin',
            'hemoglobin_mean': 'hemoglobin',
            'platelet_min': 'platelet',
            'platelet_mean': 'platelet',
            'wbc_max': 'wbc',
            'wbc_min': 'wbc',
            'wbc_mean': 'wbc',
            'bilirubin_total_max': 'bilirubin_total',
            'bicarbonate_min': 'bicarbonate',
            'gcs_min': 'gcs',
            'urineoutput': 'urineoutput',
            'pao2_fio2_mean': 'pao2_fio2',
            'pao2_fio2_min': 'pao2_fio2',
        }
    
    def _get_best_column_for_pattern(self, pattern: PatternTemplate, columns: List[str]) -> Optional[str]:
        """为模式找到最合适的数据列"""
        feature = pattern.feature
        direction = pattern.direction
        
        # 根据方向选择聚合类型
        if direction == Direction.ABOVE:
            # 检测高值：优先使用max
            preferred = [f"{feature}_max", f"{feature}_mean", feature]
        elif direction == Direction.BELOW:
            # 检测低值：优先使用min
            preferred = [f"{feature}_min", f"{feature}_mean", feature]
        else:
            preferred = [f"{feature}_mean", feature]
        
        for col in preferred:
            if col in columns:
                return col
        
        # 尝试反向映射
        for ts_col, template_feat in self.feature_name_map.items():
            if template_feat == feature and ts_col in columns:
                return ts_col
        
        return None
    
    def detect_threshold_pattern(
        self, 
        pattern: PatternTemplate, 
        values: np.ndarray,
        hours: np.ndarray
    ) -> List[Tuple[int, float]]:
        """检测阈值类型模式，返回 [(hour, value), ...]"""
        detections = []
        
        for i, (h, v) in enumerate(zip(hours, values)):
            if np.isnan(v):
                continue
            
            triggered = False
            if pattern.direction == Direction.ABOVE and v > pattern.threshold:
                triggered = True
            elif pattern.direction == Direction.BELOW and v < pattern.threshold:
                triggered = True
            
            if triggered:
                detections.append((int(h), float(v)))
        
        return detections
    
    def detect_delta_pattern(
        self,
        pattern: PatternTemplate,
        values: np.ndarray,
        hours: np.ndarray
    ) -> List[Tuple[int, float]]:
        """检测变化量模式（如肌酐48小时上升≥0.3）"""
        detections = []
        window = pattern.delta_window_hours or 24
        
        for i in range(len(values)):
            if np.isnan(values[i]):
                continue
            
            # 找到时间窗口内的最小值
            start_hour = hours[i] - window
            mask = (hours >= start_hour) & (hours <= hours[i])
            window_values = values[mask]
            window_values = window_values[~np.isnan(window_values)]
            
            if len(window_values) > 1:
                delta = values[i] - np.min(window_values)
                if delta >= pattern.delta_threshold:
                    detections.append((int(hours[i]), float(delta)))
        
        return detections
    
    def detect_patterns_for_patient(
        self,
        stay_id: int,
        patient_data: pd.DataFrame,
        disease_filter: Optional[str] = None
    ) -> List[DetectedPattern]:
        """检测单个患者的所有模式"""
        
        detected = []
        columns = patient_data.columns.tolist()
        
        # 遍历所有疾病的模式
        for disease_key, disease_set in self.patterns.items():
            if disease_filter and disease_key != disease_filter:
                continue
            
            for pattern in disease_set.patterns:
                # 找到对应的数据列
                col = self._get_best_column_for_pattern(pattern, columns)
                if col is None:
                    continue
                
                # 获取数据
                values = patient_data[col].values
                hours = patient_data['hour'].values if 'hour' in columns else np.arange(len(values))
                
                # 检测模式
                if pattern.pattern_type == PatternType.THRESHOLD:
                    detections = self.detect_threshold_pattern(pattern, values, hours)
                elif pattern.pattern_type == PatternType.DELTA:
                    detections = self.detect_delta_pattern(pattern, values, hours)
                else:
                    continue
                
                # 记录检测结果
                for hour, value in detections:
                    detected.append(DetectedPattern(
                        stay_id=stay_id,
                        hour=hour,
                        pattern_name=pattern.name,
                        pattern_type=pattern.pattern_type.value,
                        disease=disease_set.disease,
                        feature=pattern.feature,
                        value=value,
                        threshold=pattern.threshold,
                        severity=pattern.severity,
                        description=pattern.description,
                    ))
        
        return detected

# ==========================================
# 3. 批量检测
# ==========================================

def run_pattern_detection(
    timeseries_path: str,
    output_dir: str,
    max_patients: Optional[int] = None,
    window_hours: int = 24
) -> pd.DataFrame:
    """
    对所有患者运行模式检测
    
    Args:
        timeseries_path: 时序数据路径
        output_dir: 输出目录
        max_patients: 最大处理患者数（用于测试）
        window_hours: 观察窗口（小时）
    
    Returns:
        检测结果DataFrame
    """
    
    os.makedirs(output_dir, exist_ok=True)
    
    print("Loading timeseries data...")
    ts_df = pd.read_csv(timeseries_path)
    ts_df['stay_id'] = ts_df['stay_id'].astype(int)
    
    # 筛选观察窗口内的数据
    ts_df = ts_df[ts_df['hour'] < window_hours]
    
    print(f"   Loaded {len(ts_df)} records for {ts_df['stay_id'].nunique()} patients")
    print(f"   Window: 0-{window_hours}h")
    
    # 初始化检测器
    detector = PatternDetector(PATTERN_REGISTRY)
    
    # 获取患者列表
    stay_ids = ts_df['stay_id'].unique()
    if max_patients:
        stay_ids = stay_ids[:max_patients]
    
    print(f"\nRunning pattern detection on {len(stay_ids)} patients...")
    
    all_detections = []
    
    for i, stay_id in enumerate(stay_ids):
        if (i + 1) % 5000 == 0:
            print(f"   Processed {i+1}/{len(stay_ids)} patients...")
        
        patient_data = ts_df[ts_df['stay_id'] == stay_id]
        detections = detector.detect_patterns_for_patient(stay_id, patient_data)
        all_detections.extend([d.to_dict() for d in detections])
    
    # 转换为DataFrame
    results_df = pd.DataFrame(all_detections)
    
    if len(results_df) == 0:
        print("No patterns detected!")
        return results_df
    
    # 保存结果
    output_path = os.path.join(output_dir, f'detected_patterns_{window_hours}h.csv')
    results_df.to_csv(output_path, index=False)
    print(f"\nSaved: {output_path}")
    
    # ==========================================
    # 统计摘要
    # ==========================================
    print("\n" + "=" * 70)
    print("PATTERN DETECTION SUMMARY")
    print("=" * 70)
    
    print(f"\nTotal detections: {len(results_df)}")
    print(f"Patients with patterns: {results_df['stay_id'].nunique()}")
    
    print("\n[By Disease]")
    disease_counts = results_df.groupby('disease')['stay_id'].nunique()
    for disease, count in disease_counts.items():
        pct = count / len(stay_ids) * 100
        print(f"   {disease}: {count} patients ({pct:.1f}%)")
    
    print("\n[By Pattern (Top 15)]")
    pattern_counts = results_df['pattern_name'].value_counts().head(15)
    for pattern, count in pattern_counts.items():
        print(f"   {pattern}: {count}")
    
    print("\n[By Severity]")
    severity_counts = results_df.groupby('severity')['stay_id'].nunique()
    for severity, count in severity_counts.items():
        print(f"   {severity}: {count} patients")
    
    # 保存统计
    stats = {
        'total_detections': len(results_df),
        'patients_with_patterns': int(results_df['stay_id'].nunique()),
        'total_patients': len(stay_ids),
        'window_hours': window_hours,
        'by_disease': disease_counts.to_dict(),
        'by_pattern': results_df['pattern_name'].value_counts().to_dict(),
        'by_severity': severity_counts.to_dict(),
    }
    
    stats_path = os.path.join(output_dir, f'detection_stats_{window_hours}h.json')
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)
    print(f"\nStats saved: {stats_path}")
    
    return results_df

# ==========================================
# 4. 生成患者级别的模式摘要
# ==========================================

def create_patient_pattern_summary(
    detection_results: pd.DataFrame,
    output_dir: str
) -> pd.DataFrame:
    """
    为每个患者创建模式摘要（用于后续分析）
    
    输出列：
    - stay_id
    - n_patterns: 检测到的模式数量
    - n_severe: 严重模式数量
    - pattern_list: 检测到的模式列表
    - disease_list: 涉及的疾病
    - first_severe_hour: 首次严重模式的时间
    """
    
    if len(detection_results) == 0:
        return pd.DataFrame()
    
    summaries = []
    
    for stay_id, group in detection_results.groupby('stay_id'):
        severe_patterns = group[group['severity'] == 'severe']
        
        summary = {
            'stay_id': stay_id,
            'n_patterns_total': len(group),
            'n_patterns_unique': group['pattern_name'].nunique(),
            'n_severe': len(severe_patterns),
            'n_moderate': len(group[group['severity'] == 'moderate']),
            'n_mild': len(group[group['severity'] == 'mild']),
            'pattern_list': ','.join(group['pattern_name'].unique()),
            'disease_list': ','.join(group['disease'].unique()),
            'first_pattern_hour': group['hour'].min(),
            'first_severe_hour': severe_patterns['hour'].min() if len(severe_patterns) > 0 else np.nan,
            'has_sepsis_pattern': 'Sepsis' in group['disease'].values,
            'has_aki_pattern': 'AKI' in group['disease'].values,
            'has_ards_pattern': 'ARDS' in group['disease'].values,
        }
        summaries.append(summary)
    
    summary_df = pd.DataFrame(summaries)
    
    # 保存
    output_path = os.path.join(output_dir, 'patient_pattern_summary.csv')
    summary_df.to_csv(output_path, index=False)
    print(f"Patient summary saved: {output_path}")
    
    return summary_df

# ==========================================
# Pattern检测验证机制
# ==========================================

def validate_pattern_detection(
    pattern_summary: pd.DataFrame,
    cohort_path: str,
    output_dir: str
) -> Dict:
    """
    验证模式检测结果与cohort诊断标签的一致性

    添加Ground Truth验证
    计算各疾病模式检测的Precision, Recall, F1

    Args:
        pattern_summary: 患者模式摘要DataFrame
        cohort_path: cohort文件路径
        output_dir: 输出目录

    Returns:
        验证结果字典
    """

    print("\n" + "=" * 70)
    print("[VALIDATION] Pattern Detection vs Diagnosis Labels")
    print("=" * 70)

    # 加载cohort诊断标签
    cohort = pd.read_csv(cohort_path)
    cohort['stay_id'] = cohort['stay_id'].astype(int)

    # 合并模式检测结果和诊断标签
    merged = pattern_summary.merge(cohort[['stay_id', 'has_sepsis_final', 'has_aki_final', 'has_ards']],
                                    on='stay_id', how='outer')

    # 填充缺失值
    merged['has_sepsis_pattern'] = merged['has_sepsis_pattern'].fillna(False)
    merged['has_aki_pattern'] = merged['has_aki_pattern'].fillna(False)
    merged['has_ards_pattern'] = merged['has_ards_pattern'].fillna(False)
    merged['has_sepsis_final'] = merged['has_sepsis_final'].fillna(0).astype(int)
    merged['has_aki_final'] = merged['has_aki_final'].fillna(0).astype(int)
    merged['has_ards'] = merged['has_ards'].fillna(0).astype(int)

    validation_results = {}

    # 验证各疾病
    disease_pairs = [
        ('Sepsis', 'has_sepsis_pattern', 'has_sepsis_final'),
        ('AKI', 'has_aki_pattern', 'has_aki_final'),
        ('ARDS', 'has_ards_pattern', 'has_ards'),
    ]

    print("\n[Validation Results]")
    print("-" * 60)
    print(f"{'Disease':<10} {'TP':>6} {'FP':>6} {'FN':>6} {'TN':>6} {'Prec':>8} {'Recall':>8} {'F1':>8}")
    print("-" * 60)

    for disease_name, pattern_col, label_col in disease_pairs:
        # 计算混淆矩阵
        y_pred = merged[pattern_col].astype(int).values
        y_true = merged[label_col].astype(int).values

        TP = ((y_pred == 1) & (y_true == 1)).sum()
        FP = ((y_pred == 1) & (y_true == 0)).sum()
        FN = ((y_pred == 0) & (y_true == 1)).sum()
        TN = ((y_pred == 0) & (y_true == 0)).sum()

        precision = TP / (TP + FP) if (TP + FP) > 0 else 0
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        print(f"{disease_name:<10} {TP:>6} {FP:>6} {FN:>6} {TN:>6} {precision:>8.3f} {recall:>8.3f} {f1:>8.3f}")

        validation_results[disease_name] = {
            'TP': int(TP), 'FP': int(FP), 'FN': int(FN), 'TN': int(TN),
            'precision': float(precision),
            'recall': float(recall),
            'f1': float(f1),
            'support': int(y_true.sum()),
            'detected': int(y_pred.sum())
        }

    print("-" * 60)

    # 分析假阳性和假阴性的原因
    print("\n[Analysis Notes]")
    print(" 模式检测基于生理阈值，与临床诊断标签存在差异是预期的：")
    print("    - 临床诊断需要多条件组合（如Sepsis需要感染+器官功能障碍）")
    print("    - 模式检测仅检测单个指标异常")
    print("    - 高Recall + 低Precision表明模式检测敏感但不特异")
    print("    - 这是合理的设计：宁可多检测，由后续分析筛选")

    # 保存验证结果
    validation_path = os.path.join(output_dir, 'pattern_validation_report.json')
    with open(validation_path, 'w') as f:
        json.dump(validation_results, f, indent=2)
    print(f"\nValidation report saved: {validation_path}")

    return validation_results


def generate_validation_summary(validation_results: Dict, output_dir: str):
    """生成验证报告摘要"""

    summary = """
# Pattern Detection Validation Report

## Overview
This report compares pattern detection results with clinical diagnosis labels.

## Key Findings
"""

    for disease, metrics in validation_results.items():
        summary += f"""
### {disease}
- **Precision**: {metrics['precision']:.3f} (of detected patterns, {metrics['precision']*100:.1f}% match diagnosis)
- **Recall**: {metrics['recall']:.3f} (of diagnosed patients, {metrics['recall']*100:.1f}% detected)
- **F1 Score**: {metrics['f1']:.3f}
- **Support**: {metrics['support']} diagnosed patients
- **Detected**: {metrics['detected']} patients with patterns
"""

    summary += """
## Interpretation

Pattern detection is designed to be **sensitive** (high recall) rather than **specific** (high precision).

- **High Recall**: Most patients with the diagnosis have detectable patterns
- **Lower Precision**: Not all pattern detections indicate clinical diagnosis
  - This is expected because clinical diagnosis requires multiple criteria
  - Single physiological abnormalities may occur in other conditions

## Recommendation

Use pattern detection as a **screening tool**, not a diagnostic tool.
Patterns provide temporal context for downstream analysis.
"""

    report_path = os.path.join(output_dir, 'validation_summary.md')
    with open(report_path, 'w') as f:
        f.write(summary)
    print(f"Validation summary saved: {report_path}")


# ==========================================
# Main
# ==========================================

def main():
    print("Starting Pattern Detection Pipeline")
    print("=" * 70)

    # 运行检测
    results = run_pattern_detection(
        timeseries_path=TIMESERIES_FILE,
        output_dir=OUTPUT_DIR,
        max_patients=None,  # 处理所有患者
        window_hours=24
    )

    if len(results) > 0:
        # 创建患者摘要
        print("\n" + "=" * 70)
        print("Creating patient-level summary...")
        summary = create_patient_pattern_summary(results, OUTPUT_DIR)

        print(f"\nPatients with patterns: {len(summary)}")
        print(f"Patients with severe patterns: {(summary['n_severe'] > 0).sum()}")

        # 运行验证
        print("\n" + "=" * 70)
        print("Running pattern validation against diagnosis labels...")
        validation_results = validate_pattern_detection(
            pattern_summary=summary,
            cohort_path=COHORT_FILE,
            output_dir=OUTPUT_DIR
        )

        # 生成验证报告
        generate_validation_summary(validation_results, OUTPUT_DIR)

    print("\nPattern Detection Complete!")
    print(f"\nOutput files in: {OUTPUT_DIR}/")
    print("   - detected_patterns_24h.csv")
    print("   - detection_stats_24h.json")
    print("   - patient_pattern_summary.csv")
    print("   - pattern_validation_report.json  (新增)")
    print("   - validation_summary.md  (新增)")

if __name__ == "__main__":
    main()