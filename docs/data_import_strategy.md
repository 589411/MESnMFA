# 真實數據導入策略與步驟

## 概述

本文件說明如何將 Demo 架構擴展到真實生產環境，處理更複雜的數據。

---

## 第一階段：數據盤點與評估（1-2 週）

### 1.1 盤點現有數據來源

| 數據來源 | 負責單位 | 格式 | 頻率 | 可靠度 |
|---------|---------|------|------|--------|
| 母公司出貨單 | M公司 | Excel/ERP | 每批 | 高 |
| P公司收料單 | P公司 | Excel | 每日 | 中 |
| CG公司生產日報 | CG公司 | Excel/紙本 | 每日 | 中 |
| P公司生產日報 | P公司 | Excel/紙本 | 每日 | 中 |
| 成品入庫單 | 母公司 | ERP | 每批 | 高 |

### 1.2 確認必要欄位

**最低要求（必須有）：**
```
□ 日期 (Date)
□ 批號 (Batch_ID)
□ 站點 (Station)
□ 負責單位 (Owner)
□ 投入重量 (Input_Kg)
□ 產出重量 (Output_Kg)
□ 加水量 (Water_Added_Kg) - 僅加水站
```

**建議有（提高準確度）：**
```
□ 餘料投入估算 (Remnant_In_Kg_Est)
□ 餘料結存估算 (Remnant_Out_Kg_Est)
□ 廢料估算 (Waste_Kg_Est)
□ 備註 (Notes)
```

**進階欄位（未來擴充）：**
```
□ 操作人員 (Operator)
□ 設備編號 (Machine_ID)
□ 開始時間 (Start_Time)
□ 結束時間 (End_Time)
□ 製程參數 (溫度、壓力、pH等)
```

### 1.3 數據品質評估

針對每個數據來源，評估：

| 評估項目 | 檢查方法 | 合格標準 |
|---------|---------|---------|
| 完整性 | 缺失值比例 | < 5% |
| 一致性 | 跨表對帳 | 差異 < 2% |
| 時效性 | 數據延遲 | < 24 小時 |
| 格式統一 | 欄位名稱/單位 | 100% 一致 |

---

## 第二階段：數據標準化（2-3 週）

### 2.1 建立統一的數據格式

**標準 CSV 格式：**
```csv
Date,Batch_ID,Station,Owner,Input_Kg,Remnant_In_Kg_Est,Water_Added_Kg,Output_Kg,Remnant_Out_Kg_Est,Waste_Kg_Est,Notes
```

**站點代碼標準化：**
```python
STATION_MAPPING = {
    # 原始名稱 → 標準代碼
    '收料': 'A',
    '進料': 'A',
    '前處理': 'B',
    '預處理': 'B',
    'CG加水': 'C_CG',
    'P加水': 'C_P',
    'CG脫水': 'D_CG',
    'P脫水': 'D_P',
    '混合': 'E',
    '成品': 'E',
}

OWNER_MAPPING = {
    'CG': 'CG_Corp',
    'CG公司': 'CG_Corp',
    'P': 'P_Corp',
    'P公司': 'P_Corp',
    '母公司': 'Mixed',
    '混合': 'Mixed',
}
```

### 2.2 建立數據轉換腳本

```python
# data_import/transform.py

import pandas as pd
from typing import Dict, List

class DataTransformer:
    """數據轉換器：將各來源數據轉換為標準格式"""
    
    def __init__(self, station_mapping: Dict, owner_mapping: Dict):
        self.station_mapping = station_mapping
        self.owner_mapping = owner_mapping
    
    def transform_cg_report(self, df: pd.DataFrame) -> pd.DataFrame:
        """轉換 CG 公司日報格式"""
        # 欄位對應
        column_mapping = {
            '日期': 'Date',
            '批次': 'Batch_ID',
            '工序': 'Station',
            '進料(kg)': 'Input_Kg',
            '出料(kg)': 'Output_Kg',
            '加水(kg)': 'Water_Added_Kg',
            '殘留(kg)': 'Remnant_Out_Kg_Est',
        }
        df = df.rename(columns=column_mapping)
        
        # 標準化站點名稱
        df['Station'] = df['Station'].map(self.station_mapping)
        df['Owner'] = 'CG_Corp'
        
        return df
    
    def transform_p_report(self, df: pd.DataFrame) -> pd.DataFrame:
        """轉換 P 公司日報格式"""
        # 類似邏輯...
        pass
    
    def merge_and_validate(self, dfs: List[pd.DataFrame]) -> pd.DataFrame:
        """合併多個來源並驗證"""
        merged = pd.concat(dfs, ignore_index=True)
        
        # 排序
        merged = merged.sort_values(['Date', 'Batch_ID', 'Station'])
        
        # 驗證
        self._validate_data(merged)
        
        return merged
    
    def _validate_data(self, df: pd.DataFrame):
        """數據驗證"""
        # 檢查必要欄位
        required = ['Date', 'Batch_ID', 'Station', 'Input_Kg', 'Output_Kg']
        missing = [col for col in required if col not in df.columns]
        if missing:
            raise ValueError(f"缺少必要欄位: {missing}")
        
        # 檢查數值合理性
        if (df['Output_Kg'] > df['Input_Kg'] * 2).any():
            print("警告: 有產出大於投入2倍的異常記錄")
```

### 2.3 處理缺失值

```python
# data_import/fill_missing.py

class MissingValueHandler:
    """缺失值處理器"""
    
    def __init__(self, historical_stats: Dict):
        self.stats = historical_stats
    
    def fill_remnant(self, df: pd.DataFrame) -> pd.DataFrame:
        """填補餘料缺失值"""
        for station in df['Station'].unique():
            mask = (df['Station'] == station) & df['Remnant_In_Kg_Est'].isna()
            
            if station in self.stats:
                # 用該站點的歷史平均值填補
                df.loc[mask, 'Remnant_In_Kg_Est'] = self.stats[station]['avg_remnant']
            else:
                # 用 0 填補（第一批）
                df.loc[mask, 'Remnant_In_Kg_Est'] = 0
        
        return df
    
    def fill_waste(self, df: pd.DataFrame) -> pd.DataFrame:
        """填補廢料缺失值：用物質守恆反推"""
        mask = df['Waste_Kg_Est'].isna()
        
        df.loc[mask, 'Waste_Kg_Est'] = (
            df.loc[mask, 'Input_Kg'] 
            + df.loc[mask, 'Remnant_In_Kg_Est'].fillna(0)
            + df.loc[mask, 'Water_Added_Kg'].fillna(0)
            - df.loc[mask, 'Output_Kg']
            - df.loc[mask, 'Remnant_Out_Kg_Est'].fillna(0)
        ).clip(lower=0)  # 不能為負
        
        return df
```

---

## 第三階段：歷史數據導入（2-4 週）

### 3.1 導入順序

```
1. 先導入最近 3 個月的數據（建立基準線）
2. 驗證模型準確度
3. 逐步擴展到 6 個月、1 年
```

### 3.2 批次導入流程

```python
# data_import/batch_import.py

import os
from datetime import datetime

class BatchImporter:
    """批次導入器"""
    
    def __init__(self, data_dir: str, output_dir: str):
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.transformer = DataTransformer(STATION_MAPPING, OWNER_MAPPING)
        self.missing_handler = MissingValueHandler({})
    
    def import_month(self, year: int, month: int):
        """導入指定月份的數據"""
        print(f"導入 {year}-{month:02d} 數據...")
        
        # 1. 讀取各來源
        cg_file = f"{self.data_dir}/cg_report_{year}{month:02d}.xlsx"
        p_file = f"{self.data_dir}/p_report_{year}{month:02d}.xlsx"
        
        dfs = []
        if os.path.exists(cg_file):
            cg_df = pd.read_excel(cg_file)
            cg_df = self.transformer.transform_cg_report(cg_df)
            dfs.append(cg_df)
        
        if os.path.exists(p_file):
            p_df = pd.read_excel(p_file)
            p_df = self.transformer.transform_p_report(p_df)
            dfs.append(p_df)
        
        # 2. 合併
        merged = self.transformer.merge_and_validate(dfs)
        
        # 3. 填補缺失值
        merged = self.missing_handler.fill_remnant(merged)
        merged = self.missing_handler.fill_waste(merged)
        
        # 4. 計算轉換率
        merged['Conversion_Rate'] = merged['Output_Kg'] / (
            merged['Input_Kg'] + merged['Remnant_In_Kg_Est'] + merged['Water_Added_Kg'].fillna(0)
        )
        
        # 5. 儲存
        output_file = f"{self.output_dir}/production_{year}{month:02d}.csv"
        merged.to_csv(output_file, index=False)
        print(f"已儲存: {output_file}, 共 {len(merged)} 筆記錄")
        
        return merged
    
    def import_range(self, start_year: int, start_month: int, 
                     end_year: int, end_month: int):
        """導入日期範圍內的所有數據"""
        all_data = []
        
        current = datetime(start_year, start_month, 1)
        end = datetime(end_year, end_month, 1)
        
        while current <= end:
            df = self.import_month(current.year, current.month)
            all_data.append(df)
            
            # 更新歷史統計（用於填補下個月的缺失值）
            self._update_historical_stats(df)
            
            # 下個月
            if current.month == 12:
                current = datetime(current.year + 1, 1, 1)
            else:
                current = datetime(current.year, current.month + 1, 1)
        
        # 合併所有數據
        final = pd.concat(all_data, ignore_index=True)
        final.to_csv(f"{self.output_dir}/production_all.csv", index=False)
        
        return final
    
    def _update_historical_stats(self, df: pd.DataFrame):
        """更新歷史統計值"""
        for station in df['Station'].unique():
            station_data = df[df['Station'] == station]
            self.missing_handler.stats[station] = {
                'avg_remnant': station_data['Remnant_Out_Kg_Est'].mean(),
                'avg_conversion': station_data['Conversion_Rate'].mean(),
                'std_conversion': station_data['Conversion_Rate'].std(),
            }
```

### 3.3 數據驗證報告

```python
# data_import/validation_report.py

def generate_validation_report(df: pd.DataFrame) -> str:
    """生成數據驗證報告"""
    report = []
    report.append("=" * 60)
    report.append("數據導入驗證報告")
    report.append("=" * 60)
    
    # 基本統計
    report.append(f"\n總記錄數: {len(df)}")
    report.append(f"日期範圍: {df['Date'].min()} ~ {df['Date'].max()}")
    report.append(f"批次數量: {df['Batch_ID'].nunique()}")
    report.append(f"站點數量: {df['Station'].nunique()}")
    
    # 缺失值統計
    report.append("\n缺失值統計:")
    for col in df.columns:
        missing = df[col].isna().sum()
        if missing > 0:
            report.append(f"  {col}: {missing} ({missing/len(df)*100:.1f}%)")
    
    # 各站點轉換率統計
    report.append("\n各站點轉換率統計:")
    for station in sorted(df['Station'].unique()):
        station_data = df[df['Station'] == station]
        rate = station_data['Conversion_Rate']
        report.append(f"  {station}: 平均 {rate.mean():.3f}, "
                     f"標準差 {rate.std():.3f}, "
                     f"範圍 [{rate.min():.3f}, {rate.max():.3f}]")
    
    # 潛在異常
    report.append("\n潛在異常記錄:")
    for station in df['Station'].unique():
        station_data = df[df['Station'] == station]
        mean = station_data['Conversion_Rate'].mean()
        std = station_data['Conversion_Rate'].std()
        
        anomalies = station_data[
            abs(station_data['Conversion_Rate'] - mean) > 2 * std
        ]
        if len(anomalies) > 0:
            report.append(f"  {station}: {len(anomalies)} 筆 (Z > 2)")
    
    report.append("\n" + "=" * 60)
    
    return "\n".join(report)
```

---

## 第四階段：系統整合（3-4 週）

### 4.1 資料庫設計

```sql
-- 生產記錄表
CREATE TABLE production_records (
    id SERIAL PRIMARY KEY,
    date DATE NOT NULL,
    batch_id VARCHAR(50) NOT NULL,
    station VARCHAR(20) NOT NULL,
    owner VARCHAR(50) NOT NULL,
    input_kg DECIMAL(10,2) NOT NULL,
    remnant_in_kg_est DECIMAL(10,2),
    water_added_kg DECIMAL(10,2) DEFAULT 0,
    output_kg DECIMAL(10,2) NOT NULL,
    remnant_out_kg_est DECIMAL(10,2),
    waste_kg_est DECIMAL(10,2),
    conversion_rate DECIMAL(6,4),
    status_label VARCHAR(50) DEFAULT 'Normal',
    notes TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    UNIQUE(date, batch_id, station)
);

-- 站點基準線表（自動學習更新）
CREATE TABLE station_baselines (
    station VARCHAR(20) PRIMARY KEY,
    avg_conversion_rate DECIMAL(6,4),
    std_conversion_rate DECIMAL(6,4),
    avg_remnant_kg DECIMAL(10,2),
    sample_count INT,
    last_updated TIMESTAMP
);

-- 異常記錄表
CREATE TABLE anomaly_records (
    id SERIAL PRIMARY KEY,
    date DATE NOT NULL,
    batch_id VARCHAR(50) NOT NULL,
    station VARCHAR(20) NOT NULL,
    actual_rate DECIMAL(6,4),
    expected_rate DECIMAL(6,4),
    z_score DECIMAL(6,2),
    severity VARCHAR(20),
    resolved BOOLEAN DEFAULT FALSE,
    resolution_notes TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 批次譜系表
CREATE TABLE batch_genealogy (
    id SERIAL PRIMARY KEY,
    batch_id VARCHAR(50) NOT NULL,
    station VARCHAR(20) NOT NULL,
    parent_batch_id VARCHAR(50),
    remnant_ratio DECIMAL(6,4),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

### 4.2 API 設計

```python
# api/endpoints.py

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional

app = FastAPI(title="MES+MFA API")

class ProductionRecord(BaseModel):
    date: str
    batch_id: str
    station: str
    owner: str
    input_kg: float
    output_kg: float
    water_added_kg: Optional[float] = 0
    remnant_in_kg_est: Optional[float] = None
    remnant_out_kg_est: Optional[float] = None
    waste_kg_est: Optional[float] = None
    notes: Optional[str] = None

@app.post("/api/records/batch")
async def import_batch_records(records: List[ProductionRecord]):
    """批次導入生產記錄"""
    # 1. 驗證數據
    # 2. 填補缺失值
    # 3. 計算轉換率
    # 4. 異常偵測
    # 5. 儲存到資料庫
    pass

@app.get("/api/anomalies")
async def get_anomalies(
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    station: Optional[str] = None
):
    """查詢異常記錄"""
    pass

@app.get("/api/genealogy/{batch_id}")
async def get_batch_genealogy(batch_id: str, direction: str = "both"):
    """查詢批次譜系（正向/反向追溯）"""
    pass

@app.get("/api/stations/{station}/baseline")
async def get_station_baseline(station: str):
    """查詢站點基準線"""
    pass
```

### 4.3 自動化排程

```python
# scheduler/daily_import.py

from apscheduler.schedulers.background import BackgroundScheduler
import logging

scheduler = BackgroundScheduler()
logger = logging.getLogger(__name__)

@scheduler.scheduled_job('cron', hour=6, minute=0)
def daily_import_job():
    """每日 6:00 自動導入前一天的數據"""
    logger.info("開始每日數據導入...")
    
    try:
        # 1. 從 FTP/共享資料夾抓取檔案
        fetch_daily_reports()
        
        # 2. 轉換並導入
        importer = BatchImporter(DATA_DIR, OUTPUT_DIR)
        df = importer.import_yesterday()
        
        # 3. 執行異常偵測
        detector = AnomalyDetector(df)
        anomalies = detector.detect_conversion_anomaly()
        
        # 4. 發送異常通知
        if len(anomalies) > 0:
            send_anomaly_alert(anomalies)
        
        logger.info(f"導入完成: {len(df)} 筆記錄, {len(anomalies)} 筆異常")
        
    except Exception as e:
        logger.error(f"導入失敗: {e}")
        send_error_alert(e)
```

---

## 第五階段：上線與優化（持續）

### 5.1 上線檢查清單

```
□ 歷史數據導入完成（至少 3 個月）
□ 基準線已建立（各站點平均值/標準差）
□ 異常閾值已設定（預設 Z > 2）
□ 數據來源自動化（FTP/API/手動上傳）
□ 異常通知機制已設定（Email/LINE/簡訊）
□ Dashboard 已部署
□ 使用者培訓完成
```

### 5.2 持續優化項目

| 項目 | 說明 | 優先級 |
|------|------|--------|
| 基準線自動更新 | 每週/每月重新計算 | 高 |
| 異常閾值調整 | 根據實際情況微調 Z 閾值 | 高 |
| 新站點/新產品 | 擴展站點配置 | 中 |
| 製程參數整合 | 加入溫度、壓力等 | 中 |
| 預測模型 | 預測下一批的轉換率 | 低 |
| 根因分析 | 自動關聯異常與製程參數 | 低 |

### 5.3 KPI 追蹤

```python
# 系統效益 KPI
kpis = {
    '異常攔截率': '出貨前發現的異常 / 總異常數',
    '追溯時間': '從發現問題到定位根因的時間',
    '數據完整率': '有完整記錄的批次 / 總批次數',
    '基準線準確度': '預測轉換率 vs 實際轉換率的誤差',
}
```

---

## 附錄：常見問題

### Q1: 加工廠不願意提供詳細數據怎麼辦？

**策略：分階段要求**
1. 第一階段：只要求投入/產出（最低要求）
2. 第二階段：加入加水量
3. 第三階段：加入餘料估算

**談判籌碼：**
- 系統可以幫他們發現自己的問題
- 減少對帳糾紛
- 提供生產效率報表

### Q2: 數據格式不統一怎麼辦？

**解法：建立轉換層**
- 每個來源一個轉換腳本
- 統一輸出格式
- 定期檢查格式變更

### Q3: 歷史數據品質很差怎麼辦？

**策略：**
1. 先用現有數據建立「粗略基準線」
2. 從今天開始收集高品質數據
3. 3 個月後用新數據更新基準線
4. 逐步淘汰舊數據的影響

### Q4: 如何處理「補單」（事後補填的數據）？

**設計：**
- 加入 `created_at` 和 `updated_at` 欄位
- 補單標記 `is_backfill = True`
- 補單不納入基準線學習（避免污染）

---

## 專案結構（擴展後）

```
MESnMFA/
├── README.md
├── data/
│   ├── raw/                    # 原始數據（各來源）
│   │   ├── cg_reports/
│   │   └── p_reports/
│   ├── processed/              # 處理後數據
│   └── simulated_production.csv
├── data_import/                # 數據導入模組
│   ├── __init__.py
│   ├── transform.py            # 數據轉換
│   ├── fill_missing.py         # 缺失值處理
│   ├── batch_import.py         # 批次導入
│   └── validation_report.py    # 驗證報告
├── models/
│   └── mass_balance.py
├── api/                        # API 服務
│   ├── __init__.py
│   └── endpoints.py
├── scheduler/                  # 排程任務
│   └── daily_import.py
├── dashboard/
│   └── index.html
├── demo/
│   └── demo_script.md
├── docs/
│   └── data_import_strategy.md # 本文件
└── prompts/
    └── flowchart_prompt.md
```
