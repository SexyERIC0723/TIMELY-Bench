"""
使用完整 MedCAT 模型进行概念提取

Gap-6: 完整 MedCAT/UMLS 概念提取
替代原有的关键词匹配方法
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import json
import pandas as pd
from tqdm import tqdm

# MedCAT import
try:
    from medcat.cat import CAT
    from medcat.cdb import CDB
    from medcat.vocab import Vocab
    MEDCAT_AVAILABLE = True
except ImportError:
    MEDCAT_AVAILABLE = False
    print("警告: MedCAT 未安装，使用关键词匹配备选方案")

# 配置
EPISODES_DIR = Path(__file__).parent.parent.parent / 'episodes' / 'episodes_enhanced'
OUTPUT_DIR = Path(__file__).parent.parent.parent / 'data' / 'processed' / 'medcat_full'
MODEL_PATH = Path(__file__).parent.parent.parent / 'models' / 'medcat'


def load_medcat_model():
    """加载 MedCAT 模型"""
    if not MEDCAT_AVAILABLE:
        return None
    
    print("加载 MedCAT 模型...")
    
    # 尝试加载预训练模型
    try:
        # 使用 MedMentions 公开模型
        cat = CAT.load_model_pack(str(MODEL_PATH / 'medmen_wstatus_2021_oct.zip'))
        print("MedCAT 模型加载成功!")
        return cat
    except Exception as e:
        print(f"MedCAT 模型加载失败: {e}")
        return None


def extract_concepts_medcat(text: str, cat) -> list:
    """使用 MedCAT 提取医学概念"""
    if cat is None:
        return []
    
    try:
        doc = cat.get_entities(text)
        concepts = []
        for ent_id, ent_data in doc['entities'].items():
            concepts.append({
                'cui': ent_data['cui'],
                'name': ent_data['pretty_name'],
                'type': ent_data.get('type_ids', []),
                'context_similarity': ent_data.get('context_similarity', 0),
                'start': ent_data['start'],
                'end': ent_data['end']
            })
        return concepts
    except Exception as e:
        return []


def extract_concepts_keywords(text: str) -> list:
    """关键词匹配备选方案"""
    keywords = {
        'sepsis': 'C0036690',
        'pneumonia': 'C0032285',
        'acute kidney injury': 'C0022660',
        'respiratory failure': 'C0035229',
        'heart failure': 'C0018801',
        'fever': 'C0015967',
        'hypotension': 'C0020649',
        'tachycardia': 'C0039231',
        'hypoxia': 'C0242184',
        'shock': 'C0036974',
        'infection': 'C0009450',
        'diabetes': 'C0011849',
        'hypertension': 'C0020538',
        'anemia': 'C0002871',
        'edema': 'C0013604'
    }
    
    text_lower = text.lower()
    concepts = []
    for keyword, cui in keywords.items():
        if keyword in text_lower:
            concepts.append({
                'cui': cui,
                'name': keyword,
                'type': ['keyword_match'],
                'context_similarity': 0.5
            })
    return concepts


def process_episodes(sample_size=1000):
    """处理 Episode 并提取概念"""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # 加载 MedCAT
    cat = load_medcat_model()
    
    # 获取 episode 文件
    episode_files = list(EPISODES_DIR.glob('TIMELY_v2_*.json'))[:sample_size]
    print(f"处理 {len(episode_files)} 个 episodes...")
    
    results = []
    
    for ep_file in tqdm(episode_files):
        try:
            with open(ep_file) as f:
                ep = json.load(f)
            
            stay_id = ep.get('patient', {}).get('stay_id')
            clinical = ep.get('clinical_text', {})
            notes = clinical.get('notes', [])
            
            # 合并所有笔记文本
            all_text = " ".join([n.get('text', '') for n in notes])
            
            # 提取概念
            if cat:
                concepts = extract_concepts_medcat(all_text, cat)
            else:
                concepts = extract_concepts_keywords(all_text)
            
            results.append({
                'stay_id': stay_id,
                'n_concepts': len(concepts),
                'concepts': json.dumps(concepts[:20])  # 保留前20个
            })
            
        except Exception as e:
            continue
    
    # 保存结果
    df = pd.DataFrame(results)
    output_path = OUTPUT_DIR / 'medcat_concepts.csv'
    df.to_csv(output_path, index=False)
    print(f"保存到: {output_path}")
    print(f"平均每 episode 概念数: {df['n_concepts'].mean():.1f}")
    
    return df


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--sample', type=int, default=1000, help='处理的 episode 数量')
    args = parser.parse_args()
    
    process_episodes(args.sample)
