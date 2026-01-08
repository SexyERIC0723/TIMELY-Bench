"""
批量构建完整Episode数据集（全部 74,807 个 stay_id）
使用完整 47GB 对齐数据和 30 workers

基于 batch_build_core_episodes.py 修改
"""

import pandas as pd
import json
import shutil
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from tqdm import tqdm
from multiprocessing import Pool, Manager, cpu_count
from datetime import datetime
import logging
import sys
import time

# 导入现有的 Builder 和 Enhancer
from episode_builder import EpisodeBuilder, NumpyEncoder
from episode_enhancer import EpisodeEnhancer

# ==========================================
# 配置
# ==========================================

_SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = _SCRIPT_DIR.parent.parent

EPISODES_DIR = PROJECT_ROOT / 'episodes'
EPISODES_ALL_DIR = EPISODES_DIR / 'episodes_all'  # 新的输出目录
LOG_FILE = EPISODES_ALL_DIR / 'batch_build_all.log'
FAILED_IDS_FILE = EPISODES_ALL_DIR / 'failed_stay_ids.txt'

# 从 cohort 获取所有 stay_ids
sys.path.insert(0, str(_SCRIPT_DIR.parent))
from config import COHORT_FILE, TEMPORAL_ALIGNMENT_DIR

# 强制使用完整对齐文件（47GB）
FULL_ALIGNMENT_FILE = TEMPORAL_ALIGNMENT_DIR / 'temporal_textual_alignment.csv'

# ==========================================
# 日志配置
# ==========================================

def setup_logging():
    """配置日志系统"""
    EPISODES_ALL_DIR.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.FileHandler(LOG_FILE, encoding='utf-8'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger(__name__)


# ==========================================
# 全局变量
# ==========================================

_global_builder = None
_global_enhancer = None


def init_worker_full():
    """初始化工作进程：加载完整数据到全局变量"""
    global _global_builder, _global_enhancer
    
    # 创建 builder，禁用索引模式以直接加载完整文件
    _global_builder = EpisodeBuilder(use_alignment_index=False)
    _global_enhancer = EpisodeEnhancer()
    
    # 加载数据时强制使用完整对齐文件
    print(f"Worker {os.getpid()}: Loading data...")
    _global_builder.load_all_data_full()  # 使用新方法
    _global_enhancer.aligner.load_data()
    print(f"Worker {os.getpid()}: Data loaded")


def process_single_stay_id_full(args: Tuple) -> Dict:
    """处理单个 stay_id（使用全局 builder）"""
    global _global_builder, _global_enhancer
    
    stay_id, target_dir, force_rebuild = args
    
    result = {
        'stay_id': stay_id,
        'status': 'unknown',
        'message': '',
        'method': ''
    }
    
    try:
        target_file = target_dir / f"TIMELY_v2_{stay_id}.json"
        
        if not force_rebuild and target_file.exists():
            result['status'] = 'skipped'
            result['method'] = 'already_exists'
            return result
        
        # 构建 Episode
        episode = _global_builder.build_episode(stay_id)
        
        if episode is None:
            result['status'] = 'failed'
            result['method'] = 'build'
            result['message'] = 'No data found'
            return result
        
        # 转换为字典
        episode_dict = episode.to_dict()
        
        # 增强 Episode
        enhanced_dict = _global_enhancer.enhance_episode(episode_dict)
        
        # 保存
        with open(target_file, 'w', encoding='utf-8') as f:
            json.dump(enhanced_dict, f, indent=2, ensure_ascii=False, cls=NumpyEncoder)
        
        result['status'] = 'success'
        result['method'] = 'build_enhance'
        
    except Exception as e:
        result['status'] = 'failed'
        result['message'] = str(e)
    
    return result


def main(n_workers: int = 30, max_episodes: Optional[int] = None, force_rebuild: bool = False):
    """主处理流程"""
    import os
    global os  # 让 init_worker 可以使用
    
    logger = setup_logging()
    
    print("=" * 80)
    print("批量构建完整Episode数据集 (全部 74,807 stay_ids)")
    print("=" * 80)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Workers: {n_workers}")
    print(f"Alignment file: {FULL_ALIGNMENT_FILE}")
    print()
    
    # 创建目标目录
    EPISODES_ALL_DIR.mkdir(parents=True, exist_ok=True)
    
    # 读取所有 stay_ids
    cohort_df = pd.read_csv(COHORT_FILE)
    stay_ids = cohort_df['stay_id'].tolist()
    
    if max_episodes:
        stay_ids = stay_ids[:max_episodes]
        logger.info(f"Limited to first {max_episodes} episodes for testing")
    
    logger.info(f"Total stay_ids: {len(stay_ids)}")
    
    # 检查已存在的文件
    if not force_rebuild:
        existing_files = set()
        for f in EPISODES_ALL_DIR.glob('TIMELY_v2_*.json'):
            try:
                sid = int(f.stem.replace('TIMELY_v2_', ''))
                existing_files.add(sid)
            except:
                pass
        logger.info(f"Already processed: {len(existing_files)}")
        stay_ids = [sid for sid in stay_ids if sid not in existing_files]
    
    if not stay_ids:
        logger.info("All episodes already processed!")
        return
    
    logger.info(f"Need to process: {len(stay_ids)}")
    
    # 准备参数
    args_list = [(sid, EPISODES_ALL_DIR, force_rebuild) for sid in stay_ids]
    
    # 并行处理
    start_time = datetime.now()
    results = []
    
    # 注意：这里我们不能直接用 init_worker_full，因为每个进程需要加载 47GB
    # 更好的方式是使用共享内存或者顺序处理
    # 暂时使用顺序处理方式，每个进程独立加载数据
    
    print(f"\n🔄 Processing {len(stay_ids)} episodes...")
    print("   Note: Loading 47GB alignment data (~6 min per worker initialization)...")
    
    with Pool(processes=n_workers, initializer=init_worker_full) as pool:
        for result in tqdm(
            pool.imap(process_single_stay_id_full, args_list),
            total=len(stay_ids),
            desc="Processing"
        ):
            results.append(result)
    
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    
    # 统计
    stats = {
        'total': len(args_list),
        'success': sum(1 for r in results if r['status'] == 'success'),
        'failed': sum(1 for r in results if r['status'] == 'failed'),
        'skipped': sum(1 for r in results if r['status'] == 'skipped'),
        'duration': duration
    }
    
    # 记录失败
    failed = [r for r in results if r['status'] == 'failed']
    if failed:
        with open(FAILED_IDS_FILE, 'w') as f:
            for r in failed:
                f.write(f"{r['stay_id']}\t{r['message']}\n")
    
    # 打印摘要
    print("\n" + "=" * 80)
    print("处理摘要")
    print("=" * 80)
    print(f"总数: {stats['total']}")
    print(f"成功: {stats['success']} ({stats['success']/max(stats['total'],1)*100:.1f}%)")
    print(f"失败: {stats['failed']}")
    print(f"跳过: {stats['skipped']}")
    print(f"耗时: {duration:.1f}秒 ({duration/60:.1f}分钟)")
    if stats['total'] > 0:
        print(f"速度: {stats['duration']/stats['total']:.2f}秒/episode")
    print(f"输出: {EPISODES_ALL_DIR}")
    print("=" * 80)
    
    logger.info(f"Complete: {stats['success']}/{stats['total']} success, {duration:.1f}s")


if __name__ == "__main__":
    import argparse
    import os
    
    parser = argparse.ArgumentParser(description='构建完整 Episode 数据集')
    parser.add_argument('--workers', type=int, default=30, help='进程数')
    parser.add_argument('--max', type=int, default=None, help='最大处理数量')
    parser.add_argument('--force', action='store_true', help='强制重建')
    
    args = parser.parse_args()
    
    try:
        main(n_workers=args.workers, max_episodes=args.max, force_rebuild=args.force)
    except KeyboardInterrupt:
        print("\n用户中断")
    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()
