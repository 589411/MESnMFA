"""
CCL 樹脂配方模擬數據生成器
=====================================
基於改性環氧 + Active Ester 系統的物理化學規則生成模擬數據

樹脂系統假設：
- 主樹脂：低極性改性環氧 (Modified Epoxy)
- 硬化劑：Active Ester (活性酯) - 降低 Dk/Df 的關鍵
- 填料：球形二氧化矽 (Spherical Silica) - 降低 CTE
- 阻燃劑：磷系阻燃劑 (Phosphorus FR) - 無鹵環保
- 增韌劑：彈性體 (Elastomer) - 提升 Peel Strength

參考文獻：
- Active Ester 固化環氧：Dk≈3.0, Df≈0.005 @10GHz
- 球形 SiO2 填料：Dk≈3.8，可降低 CTE 至 20-30 ppm
- 業界規格：Peel Strength ≥0.6 N/mm (低粗糙度銅箔)
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional

# 設定隨機種子
np.random.seed(42)


def generate_ccl_formulation_data(
    n_samples: int = 300,
    include_process_params: bool = True,
    random_seed: Optional[int] = 42
) -> pd.DataFrame:
    """
    生成 CCL 樹脂配方的模擬數據
    
    Parameters
    ----------
    n_samples : int
        樣本數量
    include_process_params : bool
        是否包含製程參數（水洗次數、殘留離子）
    random_seed : int, optional
        隨機種子
    
    Returns
    -------
    pd.DataFrame
        模擬數據
    """
    if random_seed is not None:
        np.random.seed(random_seed)
    
    data = []
    
    for i in range(n_samples):
        # ========================================
        # 1. 生成配方參數
        # ========================================
        
        # 硬化劑當量比 (Active Ester / Epoxy 官能基比)
        # 理想範圍：0.8-1.1，過低固化不完全，過高會有殘留
        hardener_eq_ratio = np.random.uniform(0.75, 1.15)
        
        # 填料體積百分比 (vol%)
        # 範圍：15-55%，過高會導致樹脂包覆不全
        filler_vol_pct = np.random.uniform(15, 55)
        
        # 阻燃劑重量百分比 (相對於樹脂)
        # 範圍：3-8%，磷系阻燃劑
        fr_wt_pct = np.random.uniform(3, 8)
        
        # 增韌劑重量百分比 (相對於樹脂)
        # 範圍：0-8%，過多會嚴重影響 Tg
        toughener_wt_pct = np.random.uniform(0, 8)
        
        # 製程參數
        if include_process_params:
            # 水洗次數：影響殘留離子
            wash_cycles = np.random.randint(2, 6)
            # 殘留氯離子濃度 (ppm)：水洗越多越低
            residual_cl_ppm = max(5, 100 / (wash_cycles ** 1.5) + np.random.normal(0, 5))
        
        # ========================================
        # 2. 計算物理性質 (基於物理化學規則)
        # ========================================
        
        # --- Dk (介電常數) @10GHz ---
        # 基礎值：改性環氧 Dk≈3.6, Active Ester 貢獻降低
        # Active Ester 當量比越高，極性基團越少，Dk 越低
        base_dk = 3.6 - 0.5 * (hardener_eq_ratio - 0.8)  # 當量比 0.8→1.1 時 Dk 降 0.15
        
        # 填料影響：SiO2 Dk≈3.8，與樹脂相近，影響小
        # 但高填料會引入微量空氣 (Dk≈1)，略微降低整體 Dk
        dk_filler_effect = -0.002 * filler_vol_pct  # 每增加 1vol% 降 0.002
        
        # 阻燃劑影響：磷系阻燃劑略增極性
        dk_fr_effect = 0.015 * fr_wt_pct
        
        # 殘留離子影響：離子是強極性物質
        if include_process_params:
            dk_ion_effect = 0.002 * (residual_cl_ppm / 10)  # 每 10ppm 增加 0.002
        else:
            dk_ion_effect = 0
        
        dk_value = base_dk + dk_filler_effect + dk_fr_effect + dk_ion_effect
        dk_value += np.random.normal(0, 0.03)  # 測量誤差
        dk_value = np.clip(dk_value, 2.8, 4.2)
        
        # --- Df (介電損耗) @10GHz ---
        # Active Ester 是降 Df 的關鍵
        # 當量比 0.8 時 Df≈0.010，當量比 1.1 時 Df≈0.004
        base_df = 0.012 - 0.025 * (hardener_eq_ratio - 0.8)
        
        # 填料界面效應：填料/樹脂界面會增加損耗
        df_filler_effect = 0.00005 * filler_vol_pct
        
        # 阻燃劑影響
        df_fr_effect = 0.0003 * fr_wt_pct
        
        # 增韌劑影響：彈性體會增加損耗
        df_toughener_effect = 0.0003 * toughener_wt_pct
        
        # 殘留離子影響：離子遷移造成損耗
        if include_process_params:
            df_ion_effect = 0.0002 * (residual_cl_ppm / 10)
        else:
            df_ion_effect = 0
        
        df_value = base_df + df_filler_effect + df_fr_effect + df_toughener_effect + df_ion_effect
        df_value += np.random.normal(0, 0.0005)
        df_value = np.clip(df_value, 0.003, 0.015)
        
        # --- Peel Strength (銅箔剝離強度) N/mm ---
        # 基礎值：1.0 N/mm
        base_peel = 1.0
        
        # Active Ester 影響：當量比越高，極性越低，黏著力越差
        peel_hardener_effect = -0.3 * (hardener_eq_ratio - 0.8)
        
        # 填料影響：填料多會變脆，且減少樹脂與銅箔接觸面積
        # 非線性：超過 40vol% 後急劇下降
        if filler_vol_pct <= 40:
            peel_filler_effect = -0.008 * filler_vol_pct
        else:
            peel_filler_effect = -0.008 * 40 - 0.02 * (filler_vol_pct - 40)
        
        # 增韌劑影響：顯著提升黏著力
        peel_toughener_effect = 0.06 * toughener_wt_pct
        
        peel_value = base_peel + peel_hardener_effect + peel_filler_effect + peel_toughener_effect
        peel_value += np.random.normal(0, 0.04)
        peel_value = np.clip(peel_value, 0.3, 1.3)
        
        # --- Tg (玻璃轉移溫度) °C ---
        # 基礎值：改性環氧 + Active Ester 系統 Tg≈170°C
        base_tg = 170
        
        # 當量比影響：適當當量比有最高 Tg
        # 偏離 1.0 時 Tg 下降
        tg_hardener_effect = -30 * abs(hardener_eq_ratio - 1.0)
        
        # 填料影響：無機填料略微提升 Tg
        tg_filler_effect = 0.1 * filler_vol_pct
        
        # 增韌劑影響：彈性體大幅降低 Tg
        tg_toughener_effect = -3.0 * toughener_wt_pct
        
        tg_value = base_tg + tg_hardener_effect + tg_filler_effect + tg_toughener_effect
        tg_value += np.random.normal(0, 3)
        tg_value = np.clip(tg_value, 120, 200)
        
        # --- CTE (熱膨脹係數) ppm/°C (Z軸) ---
        # 基礎值：純樹脂 CTE≈60 ppm
        base_cte = 60
        
        # 填料影響：SiO2 CTE≈0.5 ppm，是降低 CTE 的主要手段
        # 混合定律 (簡化)
        cte_filler_effect = -0.7 * filler_vol_pct
        
        # 增韌劑影響：彈性體 CTE 較高
        cte_toughener_effect = 0.5 * toughener_wt_pct
        
        cte_value = base_cte + cte_filler_effect + cte_toughener_effect
        cte_value += np.random.normal(0, 2)
        cte_value = np.clip(cte_value, 15, 70)
        
        # ========================================
        # 3. 組裝數據
        # ========================================
        row = {
            'Experiment_ID': f'EXP{i+1:04d}',
            'Hardener_Eq_Ratio': round(hardener_eq_ratio, 3),
            'Filler_Vol_Pct': round(filler_vol_pct, 1),
            'FR_Wt_Pct': round(fr_wt_pct, 2),
            'Toughener_Wt_Pct': round(toughener_wt_pct, 2),
            'Dk_10GHz': round(dk_value, 3),
            'Df_10GHz': round(df_value, 5),
            'Peel_Strength_N_mm': round(peel_value, 3),
            'Tg_C': round(tg_value, 1),
            'CTE_ppm': round(cte_value, 1)
        }
        
        if include_process_params:
            row['Wash_Cycles'] = wash_cycles
            row['Residual_Cl_ppm'] = round(residual_cl_ppm, 1)
        
        data.append(row)
    
    df = pd.DataFrame(data)
    
    # 重新排列欄位順序
    if include_process_params:
        cols = ['Experiment_ID', 'Hardener_Eq_Ratio', 'Filler_Vol_Pct', 'FR_Wt_Pct', 
                'Toughener_Wt_Pct', 'Wash_Cycles', 'Residual_Cl_ppm',
                'Dk_10GHz', 'Df_10GHz', 'Peel_Strength_N_mm', 'Tg_C', 'CTE_ppm']
    else:
        cols = ['Experiment_ID', 'Hardener_Eq_Ratio', 'Filler_Vol_Pct', 'FR_Wt_Pct', 
                'Toughener_Wt_Pct', 'Dk_10GHz', 'Df_10GHz', 'Peel_Strength_N_mm', 'Tg_C', 'CTE_ppm']
    
    return df[cols]


def print_data_summary(df: pd.DataFrame) -> None:
    """印出數據摘要"""
    print("=" * 60)
    print("CCL 樹脂配方模擬數據摘要")
    print("=" * 60)
    print(f"\n樣本數: {len(df)}")
    print("\n【輸入變數 (配方參數)】")
    print("-" * 40)
    
    input_cols = ['Hardener_Eq_Ratio', 'Filler_Vol_Pct', 'FR_Wt_Pct', 'Toughener_Wt_Pct']
    if 'Wash_Cycles' in df.columns:
        input_cols += ['Wash_Cycles', 'Residual_Cl_ppm']
    
    for col in input_cols:
        print(f"{col:25s}: {df[col].min():.3f} ~ {df[col].max():.3f} (mean: {df[col].mean():.3f})")
    
    print("\n【輸出變數 (物理性質)】")
    print("-" * 40)
    output_cols = ['Dk_10GHz', 'Df_10GHz', 'Peel_Strength_N_mm', 'Tg_C', 'CTE_ppm']
    for col in output_cols:
        print(f"{col:25s}: {df[col].min():.4f} ~ {df[col].max():.4f} (mean: {df[col].mean():.4f})")
    
    print("\n【數據預覽 (前5筆)】")
    print("-" * 40)
    print(df.head().to_string(index=False))


if __name__ == '__main__':
    # 生成數據
    df = generate_ccl_formulation_data(n_samples=300, include_process_params=True)
    
    # 印出摘要
    print_data_summary(df)
    
    # 儲存 CSV
    output_path = 'ccl_resin_simulation.csv'
    df.to_csv(output_path, index=False)
    print(f"\n✅ 數據已儲存至: {output_path}")
