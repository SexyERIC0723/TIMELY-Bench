"""
Annotation-based Reasoning Features
从LLM标注结果中提取患者级别的推理得分特征

输出特征:
- supportive_score: 文本支持模式的频率
- contradictory_score: 文本与模式矛盾的频率
- uncertainty_score: LLM无法判断的频率
- alignment_coverage: 有标注的对齐比例
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
from typing import Dict, Optional
import os

from config import PROCESSED_DIR, TEMPORAL_ALIGNMENT_DIR


# 输入文件
ANNOTATIONS_DIR = PROCESSED_DIR / 'pattern_annotations'
ALIGNMENT_FILE = TEMPORAL_ALIGNMENT_DIR / 'temporal_textual_alignment.csv'

# 输出文件
OUTPUT_FILE = PROCESSED_DIR / 'annotation_features.csv'


def compute_annotation_features(
    annotations_path: Optional[str] = None,
    alignment_path: Optional[str] = None,
    output_path: Optional[str] = None
) -> pd.DataFrame:
    """
    从标注结果计算患者级别的推理得分特征

    Features:
        supportive_score: SUPPORTIVE标注占比
        contradictory_score: CONTRADICTORY标注占比
        uncertainty_score: AMBIGUOUS + UNRELATED占比
        alignment_coverage: 有标注的对齐数 / 总对齐数
        n_annotations: 标注总数
        avg_confidence: 平均置信度
    """

    # 加载标注数据
    if annotations_path is None:
        # 尝试加载所有标注文件
        annotation_files = list(ANNOTATIONS_DIR.glob('annotated_samples_*.csv'))
        if not annotation_files:
            print("No annotation files found!")
            return pd.DataFrame()

        dfs = [pd.read_csv(f) for f in annotation_files]
        annotations_df = pd.concat(dfs, ignore_index=True)
        # 去重
        annotations_df = annotations_df.drop_duplicates(
            subset=['stay_id', 'pattern_name', 'pattern_hour', 'note_type'],
            keep='first'
        )
        print(f"Loaded {len(annotations_df)} annotations from {len(annotation_files)} files")
    else:
        annotations_df = pd.read_csv(annotations_path)
        print(f"Loaded {len(annotations_df)} annotations")

    # 加载对齐数据（用于计算coverage）
    if alignment_path is None:
        alignment_path = ALIGNMENT_FILE

    if os.path.exists(alignment_path):
        alignment_df = pd.read_csv(alignment_path)
        alignment_counts = alignment_df.groupby('stay_id').size().to_dict()
        print(f"Loaded {len(alignment_df)} alignments for coverage calculation")
    else:
        alignment_counts = {}
        print("Alignment file not found, coverage will be 0")

    # 按患者计算特征
    features = []

    for stay_id, group in annotations_df.groupby('stay_id'):
        n_total = len(group)

        # 各类别计数
        category_counts = group['annotation_category'].value_counts().to_dict()
        n_supportive = category_counts.get('SUPPORTIVE', 0)
        n_contradictory = category_counts.get('CONTRADICTORY', 0)
        n_ambiguous = category_counts.get('AMBIGUOUS', 0)
        n_unrelated = category_counts.get('UNRELATED', 0)

        # 计算得分
        supportive_score = n_supportive / n_total if n_total > 0 else 0
        contradictory_score = n_contradictory / n_total if n_total > 0 else 0
        uncertainty_score = (n_ambiguous + n_unrelated) / n_total if n_total > 0 else 0

        # 计算coverage
        total_alignments = alignment_counts.get(stay_id, 0)
        alignment_coverage = n_total / total_alignments if total_alignments > 0 else 0

        # 平均置信度
        avg_confidence = group['annotation_confidence'].mean() if 'annotation_confidence' in group.columns else 0.5

        # 额外特征：按严重程度加权的得分
        if 'pattern_severity' in group.columns:
            severe_group = group[group['pattern_severity'] == 'severe']
            if len(severe_group) > 0:
                severe_supportive = (severe_group['annotation_category'] == 'SUPPORTIVE').sum() / len(severe_group)
                severe_contradictory = (severe_group['annotation_category'] == 'CONTRADICTORY').sum() / len(severe_group)
            else:
                severe_supportive = 0
                severe_contradictory = 0
        else:
            severe_supportive = 0
            severe_contradictory = 0

        features.append({
            'stay_id': stay_id,
            # 核心推理得分
            'supportive_score': round(supportive_score, 4),
            'contradictory_score': round(contradictory_score, 4),
            'uncertainty_score': round(uncertainty_score, 4),
            # 覆盖率和数量
            'alignment_coverage': round(alignment_coverage, 4),
            'n_annotations': n_total,
            'avg_confidence': round(avg_confidence, 4),
            # 严重模式的得分（用于识别高风险患者）
            'severe_supportive_score': round(severe_supportive, 4),
            'severe_contradictory_score': round(severe_contradictory, 4),
            # 原始计数
            'n_supportive': n_supportive,
            'n_contradictory': n_contradictory,
            'n_ambiguous': n_ambiguous,
            'n_unrelated': n_unrelated,
        })

    features_df = pd.DataFrame(features)

    # 保存
    if output_path is None:
        output_path = OUTPUT_FILE

    features_df.to_csv(output_path, index=False)
    print(f"\nSaved annotation features: {output_path}")
    print(f"   Total patients: {len(features_df)}")

    # 统计摘要
    print("\n[Feature Statistics]")
    for col in ['supportive_score', 'contradictory_score', 'uncertainty_score', 'alignment_coverage']:
        if col in features_df.columns:
            mean_val = features_df[col].mean()
            std_val = features_df[col].std()
            print(f"   {col}: mean={mean_val:.3f}, std={std_val:.3f}")

    return features_df


def merge_annotation_features_with_data(
    data_df: pd.DataFrame,
    features_df: Optional[pd.DataFrame] = None,
    fill_missing: bool = True
) -> pd.DataFrame:
    """
    将标注特征合并到现有数据集

    Args:
        data_df: 原始数据（需包含stay_id列）
        features_df: 标注特征DataFrame，如果为None则从文件加载
        fill_missing: 是否填充缺失值（无标注的患者）
    """

    if features_df is None:
        if OUTPUT_FILE.exists():
            features_df = pd.read_csv(OUTPUT_FILE)
        else:
            print("Annotation features file not found, computing...")
            features_df = compute_annotation_features()

    if len(features_df) == 0:
        print("No annotation features available")
        return data_df

    # 合并
    merged = data_df.merge(features_df, on='stay_id', how='left')

    # 填充缺失值（无标注的患者默认uncertainty_score=1）
    if fill_missing:
        feature_cols = [
            'supportive_score', 'contradictory_score', 'uncertainty_score',
            'alignment_coverage', 'avg_confidence',
            'severe_supportive_score', 'severe_contradictory_score'
        ]
        for col in feature_cols:
            if col in merged.columns:
                if col == 'uncertainty_score':
                    merged[col] = merged[col].fillna(1.0)  # 无标注 = 完全不确定
                else:
                    merged[col] = merged[col].fillna(0.0)

        count_cols = ['n_annotations', 'n_supportive', 'n_contradictory', 'n_ambiguous', 'n_unrelated']
        for col in count_cols:
            if col in merged.columns:
                merged[col] = merged[col].fillna(0).astype(int)

    print(f"Merged annotation features: {len(merged)} samples")
    n_with_features = (merged['n_annotations'] > 0).sum() if 'n_annotations' in merged.columns else 0
    print(f"   Samples with annotations: {n_with_features}")

    return merged


# 用于基线模型的特征列表
ANNOTATION_FEATURE_COLS = [
    'supportive_score',
    'contradictory_score',
    'uncertainty_score',
    'alignment_coverage',
    'avg_confidence',
    'severe_supportive_score',
    'severe_contradictory_score',
]


def main():
    print("=" * 60)
    print("Computing Annotation-based Reasoning Features")
    print("=" * 60)

    features_df = compute_annotation_features()

    if len(features_df) > 0:
        print("\n[Sample Output]")
        print(features_df.head(10).to_string())

    print("\nDone!")


if __name__ == "__main__":
    main()
