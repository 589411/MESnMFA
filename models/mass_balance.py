"""
MES + MFA 物質流分析模型
Mass Balance Analysis for Outsourced Manufacturing

現實限制：
- 無法取得乾重（所有物料都含水分）
- 餘料是估算的
- 廢料是估算的（反應器殘留）
- 可靠數據：投入重、產出重、加水量

核心方法：轉換率統計分析
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional


# ============================================================
# 站點配置
# ============================================================

@dataclass
class StationConfig:
    """站點標準參數"""
    name: str
    expected_conversion_rate: float  # 預期轉換率
    std_tolerance: float             # 標準差容許倍數
    adds_water: bool = False         # 是否為加水站


# 各站點歷史平均轉換率（從數據中學習得到）
STATION_CONFIGS = {
    'A': StationConfig('收料站', 0.97, 2.0),
    'B': StationConfig('前處理站', 0.965, 2.0),
    'C': StationConfig('分流加水站', 0.97, 2.0, adds_water=True),
    'D_CG': StationConfig('CG脫水站', 0.70, 2.0),
    'D_P': StationConfig('P脫水站', 0.70, 2.0),
    'E': StationConfig('混合成品站', 0.53, 2.0),
}


# ============================================================
# 轉換率計算
# ============================================================

def calculate_conversion_rate(
    input_kg: float, 
    output_kg: float, 
    water_added_kg: float = 0,
    remnant_in_kg: float = 0
) -> float:
    """
    計算站點轉換率
    
    轉換率 = 產出 / (投入 + 加水 + 餘料投入)
    """
    total_input = input_kg + water_added_kg + remnant_in_kg
    if total_input == 0:
        return 0
    return output_kg / total_input


# ============================================================
# 異常偵測
# ============================================================

class AnomalyDetector:
    """異常偵測器"""
    
    def __init__(self, data: pd.DataFrame):
        self.data = data.copy()
        self.station_stats = {}
        self._calculate_station_statistics()
    
    def _calculate_station_statistics(self):
        """計算各站點的歷史統計值"""
        for station in self.data['Station'].unique():
            station_data = self.data[
                (self.data['Station'] == station) & 
                (self.data['Status_Label'] == 'Normal')
            ]
            
            if len(station_data) > 0:
                rates = station_data['Conversion_Rate']
                self.station_stats[station] = {
                    'mean': rates.mean(),
                    'std': rates.std(),
                    'min': rates.min(),
                    'max': rates.max(),
                    'count': len(rates)
                }
    
    def detect_conversion_anomaly(self, threshold_std: float = 2.0) -> pd.DataFrame:
        """
        偵測轉換率異常
        
        Args:
            threshold_std: 標準差倍數閾值
        
        Returns:
            異常記錄 DataFrame
        """
        anomalies = []
        
        for _, row in self.data.iterrows():
            station = row['Station']
            if station not in self.station_stats:
                continue
            
            stats = self.station_stats[station]
            actual_rate = row['Conversion_Rate']
            expected_rate = stats['mean']
            std = stats['std'] if stats['std'] > 0 else 0.01
            
            z_score = abs(actual_rate - expected_rate) / std
            
            if z_score > threshold_std:
                anomaly = {
                    'Date': row['Date'],
                    'Batch_ID': row['Batch_ID'],
                    'Station': station,
                    'Owner': row['Owner'],
                    'Actual_Rate': actual_rate,
                    'Expected_Rate': expected_rate,
                    'Z_Score': z_score,
                    'Direction': 'HIGH' if actual_rate > expected_rate else 'LOW',
                    'Severity': 'Critical' if z_score > 3 else 'Warning'
                }
                anomalies.append(anomaly)
        
        return pd.DataFrame(anomalies)
    
    def get_station_summary(self) -> pd.DataFrame:
        """取得各站點統計摘要"""
        summary = []
        for station, stats in self.station_stats.items():
            config = STATION_CONFIGS.get(station)
            summary.append({
                'Station': station,
                'Name': config.name if config else station,
                'Mean_Rate': stats['mean'],
                'Std': stats['std'],
                'Min_Rate': stats['min'],
                'Max_Rate': stats['max'],
                'Sample_Count': stats['count']
            })
        return pd.DataFrame(summary)


# ============================================================
# 批次譜系追蹤
# ============================================================

class BatchGenealogy:
    """批次譜系追蹤器"""
    
    def __init__(self, data: pd.DataFrame):
        self.data = data.copy()
        self.genealogy = {}  # {batch_id: {station: composition}}
        self._build_genealogy()
    
    def _build_genealogy(self):
        """建立批次譜系樹"""
        batches = self.data['Batch_ID'].unique()
        
        for batch in batches:
            batch_data = self.data[self.data['Batch_ID'] == batch]
            self.genealogy[batch] = {}
            
            for _, row in batch_data.iterrows():
                station = row['Station']
                input_kg = row['Input_Kg']
                remnant_in = row['Remnant_In_Kg_Est']
                total_input = input_kg + remnant_in + row['Water_Added_Kg']
                
                if total_input > 0:
                    composition = {
                        'new_material_ratio': input_kg / total_input,
                        'remnant_ratio': remnant_in / total_input,
                        'water_ratio': row['Water_Added_Kg'] / total_input,
                        'remnant_source': self._get_previous_batch(batch)
                    }
                else:
                    composition = {
                        'new_material_ratio': 1.0,
                        'remnant_ratio': 0,
                        'water_ratio': 0,
                        'remnant_source': None
                    }
                
                self.genealogy[batch][station] = composition
    
    def _get_previous_batch(self, current_batch: str) -> Optional[str]:
        """取得前一批次 ID"""
        batches = sorted(self.data['Batch_ID'].unique().tolist())
        idx = batches.index(current_batch)
        if idx > 0:
            return batches[idx - 1]
        return None
    
    def trace_forward(self, batch_id: str) -> Dict:
        """
        正向追溯：這批原料影響了哪些後續批次？
        """
        affected = []
        batches = sorted(self.data['Batch_ID'].unique().tolist())
        start_idx = batches.index(batch_id)
        
        for i in range(start_idx + 1, len(batches)):
            next_batch = batches[i]
            if next_batch in self.genealogy:
                for station, comp in self.genealogy[next_batch].items():
                    if comp['remnant_ratio'] > 0:
                        affected.append({
                            'Batch_ID': next_batch,
                            'Station': station,
                            'Remnant_Ratio': comp['remnant_ratio'],
                            'Source_Batch': comp['remnant_source']
                        })
        
        return {
            'source_batch': batch_id,
            'affected_batches': affected
        }
    
    def trace_backward(self, batch_id: str) -> Dict:
        """
        反向追溯：這批成品的原料來源組成？
        """
        if batch_id not in self.genealogy:
            return {}
        
        composition = self.genealogy[batch_id]
        sources = []
        
        for station, comp in composition.items():
            sources.append({
                'Station': station,
                'New_Material_Ratio': comp['new_material_ratio'],
                'Remnant_Ratio': comp['remnant_ratio'],
                'Remnant_From': comp['remnant_source']
            })
        
        return {
            'batch_id': batch_id,
            'composition': sources
        }


# ============================================================
# 餘料影響分析
# ============================================================

class RemnantAnalyzer:
    """餘料影響分析器"""
    
    def __init__(self, data: pd.DataFrame):
        self.data = data.copy()
    
    def calculate_remnant_dilution(self, anomaly_batch: str, target_batch: str) -> Dict:
        """
        計算異常批次的餘料在後續批次中的稀釋程度
        
        Args:
            anomaly_batch: 異常批次 ID
            target_batch: 目標批次 ID
        
        Returns:
            各站點的稀釋比例
        """
        batches = sorted(self.data['Batch_ID'].unique().tolist())
        start_idx = batches.index(anomaly_batch)
        end_idx = batches.index(target_batch)
        
        dilution = {}
        
        for station in self.data['Station'].unique():
            # 計算從異常批次到目標批次的累積稀釋
            cumulative_ratio = 1.0
            
            for i in range(start_idx + 1, end_idx + 1):
                batch = batches[i]
                batch_station_data = self.data[
                    (self.data['Batch_ID'] == batch) & 
                    (self.data['Station'] == station)
                ]
                
                if len(batch_station_data) > 0:
                    row = batch_station_data.iloc[0]
                    total_input = row['Input_Kg'] + row['Remnant_In_Kg_Est'] + row['Water_Added_Kg']
                    if total_input > 0:
                        remnant_ratio = row['Remnant_In_Kg_Est'] / total_input
                        cumulative_ratio *= remnant_ratio
            
            dilution[station] = cumulative_ratio
        
        return {
            'anomaly_batch': anomaly_batch,
            'target_batch': target_batch,
            'dilution_by_station': dilution
        }
    
    def estimate_steady_state_remnant(self) -> pd.DataFrame:
        """
        估算各站點的穩態餘料量
        
        使用長期平均來推估「看不見的餘料」
        """
        results = []
        
        for station in self.data['Station'].unique():
            station_data = self.data[self.data['Station'] == station]
            
            avg_remnant_in = station_data['Remnant_In_Kg_Est'].mean()
            avg_remnant_out = station_data['Remnant_Out_Kg_Est'].mean()
            
            # 穩態時，餘料進 ≈ 餘料出
            steady_state = (avg_remnant_in + avg_remnant_out) / 2
            
            results.append({
                'Station': station,
                'Avg_Remnant_In': avg_remnant_in,
                'Avg_Remnant_Out': avg_remnant_out,
                'Estimated_Steady_State': steady_state
            })
        
        return pd.DataFrame(results)


# ============================================================
# 主程式：Demo 用
# ============================================================

def run_demo_analysis(csv_path: str):
    """執行 Demo 分析"""
    
    print("=" * 60)
    print("MES + MFA 物質流分析 Demo")
    print("=" * 60)
    
    # 載入數據
    df = pd.read_csv(csv_path)
    print(f"\n📊 載入 {len(df)} 筆生產記錄")
    
    # 1. 異常偵測
    print("\n" + "-" * 40)
    print("🔍 異常偵測分析")
    print("-" * 40)
    
    detector = AnomalyDetector(df)
    
    print("\n各站點轉換率統計：")
    summary = detector.get_station_summary()
    print(summary.to_string(index=False))
    
    print("\n偵測到的異常：")
    anomalies = detector.detect_conversion_anomaly(threshold_std=2.0)
    if len(anomalies) > 0:
        print(anomalies.to_string(index=False))
    else:
        print("無異常")
    
    # 2. 批次譜系
    print("\n" + "-" * 40)
    print("🧬 批次譜系追蹤")
    print("-" * 40)
    
    genealogy = BatchGenealogy(df)
    
    # 正向追溯：B003 異常批次影響了誰？
    print("\n正向追溯 B003（異常批次）的影響：")
    forward = genealogy.trace_forward('B003')
    for affected in forward['affected_batches'][:5]:
        print(f"  → {affected['Batch_ID']} @ {affected['Station']}: "
              f"含 {affected['Remnant_Ratio']*100:.1f}% 餘料")
    
    # 反向追溯：B005 成品的組成
    print("\n反向追溯 B005 成品的原料組成：")
    backward = genealogy.trace_backward('B005')
    for comp in backward['composition']:
        print(f"  {comp['Station']}: "
              f"新料 {comp['New_Material_Ratio']*100:.1f}% + "
              f"餘料 {comp['Remnant_Ratio']*100:.1f}%")
    
    # 3. 餘料影響
    print("\n" + "-" * 40)
    print("📦 餘料影響分析")
    print("-" * 40)
    
    remnant = RemnantAnalyzer(df)
    
    print("\n各站點穩態餘料估算：")
    steady = remnant.estimate_steady_state_remnant()
    print(steady.to_string(index=False))
    
    print("\nB003 異常在 B005 中的稀釋程度：")
    dilution = remnant.calculate_remnant_dilution('B003', 'B005')
    for station, ratio in dilution['dilution_by_station'].items():
        if ratio > 0:
            print(f"  {station}: {ratio*100:.4f}%")
    
    print("\n" + "=" * 60)
    print("分析完成")
    print("=" * 60)


if __name__ == "__main__":
    import os
    
    # 取得 CSV 路徑
    script_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(script_dir, "..", "data", "simulated_production.csv")
    
    run_demo_analysis(csv_path)
