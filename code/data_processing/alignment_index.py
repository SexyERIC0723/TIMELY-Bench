"""
临床对齐数据索引模块
用于处理 47GB 的 temporal_textual_alignment.csv 文件

策略：
1. 首次运行时构建 stay_id -> 行号区间的索引
2. 后续运行时加载索引，按需读取特定 stay_id 的数据
3. 使用分块读取避免内存溢出
"""

import os
import json
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import pickle


class AlignmentIndexer:
    """
    对齐数据索引器
    
    思路：
    - 索引文件存储每个 stay_id 的行号范围 (start_line, end_line)
    - 查询时使用 skiprows + nrows 精确读取
    """
    
    def __init__(self, csv_path: Path, index_path: Optional[Path] = None):
        self.csv_path = Path(csv_path)
        self.index_path = index_path or self.csv_path.with_suffix('.index.pkl')
        self.header_line = None  # CSV 表头行
        self.column_dtypes = {
            'stay_id': int,
            'pattern_hour': float,
            'note_hour': float,
            'time_delta_hours': float,
        }
        self._index: Dict[int, Tuple[int, int]] = {}  # stay_id -> (start_line, end_line)
        self._loaded = False
        
    def build_index(self, force_rebuild: bool = False) -> bool:
        """
        构建索引文件
        
        Returns:
            True 如果成功构建或已存在索引
        """
        if self.index_path.exists() and not force_rebuild:
            print(f"索引文件已存在: {self.index_path}")
            return self.load_index()
        
        print(f"开始构建索引 (CSV: {self.csv_path.stat().st_size / 1e9:.1f} GB)...")
        
        # 分块读取，只读取 stay_id 列来构建索引
        chunk_size = 500_000  # 每次读取 50万行
        line_offset = 0
        current_stay_id = None
        stay_start_line = 0
        
        index_data = {}
        
        try:
            reader = pd.read_csv(
                self.csv_path,
                usecols=['stay_id'],  # 只读取 stay_id 列节省内存
                dtype={'stay_id': int},
                chunksize=chunk_size
            )
            
            for chunk_idx, chunk in enumerate(reader):
                if chunk_idx == 0:
                    # 保存表头信息
                    self.header_line = 1
                
                for i, row in chunk.iterrows():
                    actual_line = line_offset + i - (chunk_idx * chunk_size) + 1  # 1-indexed, 跳过表头
                    stay_id = int(row['stay_id'])
                    
                    if current_stay_id is None:
                        current_stay_id = stay_id
                        stay_start_line = actual_line
                    elif stay_id != current_stay_id:
                        # 记录前一个 stay_id 的范围
                        index_data[current_stay_id] = (stay_start_line, actual_line - 1)
                        current_stay_id = stay_id
                        stay_start_line = actual_line
                
                line_offset = actual_line
                
                if (chunk_idx + 1) % 10 == 0:
                    print(f"  已处理 {(chunk_idx + 1) * chunk_size / 1e6:.1f}M 行, "
                          f"已索引 {len(index_data)} 个 stay_id...")
            
            # 最后一个 stay_id
            if current_stay_id is not None:
                index_data[current_stay_id] = (stay_start_line, line_offset)
            
            print(f"索引构建完成: {len(index_data)} 个 stay_id")
            
            # 保存索引
            self._index = index_data
            with open(self.index_path, 'wb') as f:
                pickle.dump({
                    'index': index_data,
                    'csv_size': self.csv_path.stat().st_size,
                    'csv_mtime': self.csv_path.stat().st_mtime
                }, f)
            
            print(f"索引已保存: {self.index_path}")
            self._loaded = True
            return True
            
        except Exception as e:
            print(f"构建索引失败: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def load_index(self) -> bool:
        """加载已有索引"""
        if not self.index_path.exists():
            print(f"索引文件不存在: {self.index_path}")
            return False
        
        try:
            with open(self.index_path, 'rb') as f:
                data = pickle.load(f)
            
            # 验证 CSV 是否改变
            if self.csv_path.exists():
                current_size = self.csv_path.stat().st_size
                if data.get('csv_size') != current_size:
                    print("警告: CSV 文件大小已改变，索引可能过期")
                    return False
            
            self._index = data['index']
            self._loaded = True
            print(f"索引加载成功: {len(self._index)} 个 stay_id")
            return True
            
        except Exception as e:
            print(f"加载索引失败: {e}")
            return False
    
    def get_alignment_data(self, stay_id: int) -> Optional[pd.DataFrame]:
        """
        获取指定 stay_id 的对齐数据
        
        Args:
            stay_id: 患者 stay_id
            
        Returns:
            包含该患者所有对齐记录的 DataFrame，如果没有则返回 None
        """
        if not self._loaded:
            if not self.load_index():
                print("无法加载索引，尝试构建...")
                if not self.build_index():
                    return None
        
        if stay_id not in self._index:
            return None
        
        start_line, end_line = self._index[stay_id]
        n_rows = end_line - start_line + 1
        
        try:
            # 使用 skiprows 和 nrows 精确读取
            df = pd.read_csv(
                self.csv_path,
                skiprows=range(1, start_line),  # 跳过表头和前面的行
                nrows=n_rows,
                dtype=self.column_dtypes
            )
            df['stay_id'] = df['stay_id'].astype(int)
            return df
            
        except Exception as e:
            print(f"读取 stay_id={stay_id} 数据失败: {e}")
            return None
    
    def get_stay_ids(self) -> List[int]:
        """获取所有已索引的 stay_id"""
        if not self._loaded:
            self.load_index()
        return list(self._index.keys())
    
    def has_stay_id(self, stay_id: int) -> bool:
        """检查是否有指定 stay_id 的数据"""
        if not self._loaded:
            self.load_index()
        return stay_id in self._index


class ChunkedAlignmentReader:
    """
    分块对齐数据读取器（备选方案）
    
    如果索引构建太慢，可以使用此类进行简单分块读取
    """
    
    def __init__(self, csv_path: Path, chunk_size: int = 100_000):
        self.csv_path = Path(csv_path)
        self.chunk_size = chunk_size
        self._cache: Dict[int, pd.DataFrame] = {}
        self._cache_max_size = 500  # 最多缓存 500 个 stay_id
    
    def get_alignment_data(self, stay_id: int) -> Optional[pd.DataFrame]:
        """
        分块读取获取指定 stay_id 的数据
        注意：这种方法效率较低，每次需要扫描整个文件
        """
        if stay_id in self._cache:
            return self._cache[stay_id]
        
        result_chunks = []
        
        for chunk in pd.read_csv(self.csv_path, chunksize=self.chunk_size):
            chunk['stay_id'] = chunk['stay_id'].astype(int)
            matching = chunk[chunk['stay_id'] == stay_id]
            if len(matching) > 0:
                result_chunks.append(matching)
        
        if result_chunks:
            result = pd.concat(result_chunks, ignore_index=True)
            
            # 简单 LRU 缓存
            if len(self._cache) >= self._cache_max_size:
                # 删除最早的缓存项
                oldest_key = next(iter(self._cache))
                del self._cache[oldest_key]
            
            self._cache[stay_id] = result
            return result
        
        return None


# 便捷函数
def build_alignment_index(csv_path: Path, force: bool = False) -> AlignmentIndexer:
    """构建并返回索引器"""
    indexer = AlignmentIndexer(csv_path)
    indexer.build_index(force_rebuild=force)
    return indexer


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    
    from config import TEMPORAL_ALIGNMENT_DIR
    
    csv_path = TEMPORAL_ALIGNMENT_DIR / 'temporal_textual_alignment.csv'
    
    print("=" * 60)
    print("构建对齐数据索引")
    print("=" * 60)
    
    indexer = build_alignment_index(csv_path, force='--force' in sys.argv)
    
    # 测试
    print("\n测试随机 stay_id...")
    stay_ids = indexer.get_stay_ids()[:5]
    for sid in stay_ids:
        df = indexer.get_alignment_data(sid)
        if df is not None:
            print(f"  stay_id={sid}: {len(df)} 条对齐记录")
