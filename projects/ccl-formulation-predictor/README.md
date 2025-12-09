# CCL 樹脂配方預測模型

> 基於機器學習的 CCL (銅箔基板) 樹脂配方物性預測系統

## 專案背景

本專案針對 5G 高頻應用的 CCL 樹脂配方開發，建立預測模型以：
1. **正向預測**：給定配方參數 → 預測物理性質 (Dk, Df, Peel, Tg, CTE)
2. **反向設計**：給定目標規格 → 搜尋最佳配方

## 樹脂系統

假設採用 **改性環氧 + Active Ester** 系統：

```
改性環氧 (Modified Epoxy)
    ↓ + Active Ester 硬化劑
    ↓ + 球形 SiO₂ 填料
    ↓ + 磷系阻燃劑
    ↓ + 彈性體增韌劑
    ↓
低 Dk/Df 的 CCL 樹脂
```

### 為什麼選擇這個系統？

| 樹脂類型 | Dk | Df | 優點 | 缺點 |
|----------|----|----|------|------|
| 傳統環氧 FR-4 | 4.2-4.8 | 0.017-0.025 | 便宜、成熟 | Dk/Df 太高 |
| **改性環氧+Active Ester** | **3.0-3.5** | **0.004-0.010** | 成本適中、加工性好 | 需優化配方 |
| PPE | 2.4-2.6 | 0.002-0.004 | Dk/Df 最低 | 黏著力差、成本高 |
| PTFE | 2.1-2.6 | 0.001-0.006 | 超低損耗 | 加工困難、成本極高 |

## 數據格式

### 輸入變數 (配方參數)

| 變數 | 說明 | 範圍 | 單位 |
|------|------|------|------|
| `Hardener_Eq_Ratio` | 硬化劑當量比 (Active Ester/Epoxy) | 0.75-1.15 | - |
| `Filler_Vol_Pct` | 填料體積百分比 (球形 SiO₂) | 15-55 | vol% |
| `FR_Wt_Pct` | 阻燃劑重量百分比 | 3-8 | wt% |
| `Toughener_Wt_Pct` | 增韌劑重量百分比 | 0-8 | wt% |
| `Wash_Cycles` | 水洗次數 (純化製程) | 2-5 | 次 |
| `Residual_Cl_ppm` | 殘留氯離子濃度 | 5-45 | ppm |

### 輸出變數 (物理性質)

| 變數 | 說明 | 目標方向 | 典型範圍 | 單位 |
|------|------|----------|----------|------|
| `Dk_10GHz` | 介電常數 @10GHz | ↓ 越低越好 | 3.0-3.8 | - |
| `Df_10GHz` | 介電損耗 @10GHz | ↓ 越低越好 | 0.003-0.015 | - |
| `Peel_Strength_N_mm` | 銅箔剝離強度 | ↑ 越高越好 | 0.4-1.3 | N/mm |
| `Tg_C` | 玻璃轉移溫度 | ↑ 越高越好 | 140-180 | °C |
| `CTE_ppm` | 熱膨脹係數 (Z軸) | ↓ 越低越好 | 20-60 | ppm/°C |

## 核心矛盾 (Trade-offs)

模型需要學習的關鍵矛盾關係：

```
┌─────────────────────────────────────────────────────┐
│                                                     │
│   低 Dk/Df  ←──── 衝突 ────→  高 Peel Strength     │
│      ↑                              ↑               │
│  Active Ester ↑                Toughener ↑          │
│  (降低極性)                    (提升黏著)           │
│                                     ↓               │
│                                  Tg ↓               │
│                              (犧牲耐熱)             │
│                                                     │
├─────────────────────────────────────────────────────┤
│                                                     │
│   低 CTE  ←──── 衝突 ────→  高 Peel Strength       │
│      ↑                              ↑               │
│  Filler ↑                       變脆               │
│  (無機填料)                   → Peel ↓             │
│                                                     │
└─────────────────────────────────────────────────────┘
```

## 專案結構

```
ccl-formulation-predictor/
├── data/
│   ├── generate_simulation.py    # 模擬數據生成器
│   └── ccl_resin_simulation.csv  # 模擬數據 (300筆)
├── notebooks/
│   ├── 01_data_exploration.ipynb # EDA 探索分析
│   ├── 02_model_training.ipynb   # 模型訓練
│   └── 03_optimization.ipynb     # 反向設計/配方優化
├── models/
│   ├── predictor.py              # 預測模型
│   └── optimizer.py              # 配方優化器
├── dashboard/
│   └── index.html                # 互動式預測介面
└── README.md
```

## 快速開始

### 1. 生成模擬數據

```bash
cd data
python3 generate_simulation.py
```

### 2. 訓練預測模型 (待實作)

```python
from models.predictor import CCLPredictor

# 載入數據
predictor = CCLPredictor()
predictor.load_data('data/ccl_resin_simulation.csv')

# 訓練模型
predictor.train()

# 預測新配方
result = predictor.predict({
    'Hardener_Eq_Ratio': 0.95,
    'Filler_Vol_Pct': 35,
    'FR_Wt_Pct': 5,
    'Toughener_Wt_Pct': 3
})
print(result)
# {'Dk_10GHz': 3.45, 'Df_10GHz': 0.008, 'Peel_Strength_N_mm': 0.85, ...}
```

### 3. 反向設計 (待實作)

```python
from models.optimizer import CCLOptimizer

optimizer = CCLOptimizer(predictor)

# 設定目標規格
target = {
    'Dk_10GHz': {'max': 3.3},
    'Df_10GHz': {'max': 0.006},
    'Peel_Strength_N_mm': {'min': 0.7},
    'Tg_C': {'min': 160}
}

# 搜尋最佳配方
best_formulations = optimizer.search(target, n_results=5)
```

## 參考文獻

1. Active Ester 固化環氧系統：Dk≈3.0, Df≈0.005 @10GHz
   - [ScienceDirect: Reaction mechanism investigation of low Df epoxy-active ester film](https://www.sciencedirect.com/science/article/abs/pii/S1381514823002262)

2. CCL 材料規格比較
   - [SF Circuits: PCB Material Reference Guide](https://www.sfcircuits.com/pcb-production-capabilities/pcb-material-reference-guide)

3. 5G 低介電材料趨勢
   - [公隆化學: 低介電樹脂應用](https://www.es-kelly.com/news_detail.php?newsId=JCUxNzIjIQ%3D%3D)

## 開發計劃

- [x] 階段 1：模擬數據生成
- [ ] 階段 2：EDA 探索分析
- [ ] 階段 3：模型訓練 (Random Forest / XGBoost)
- [ ] 階段 4：模型評估與特徵重要性分析
- [ ] 階段 5：反向設計/配方優化
- [ ] 階段 6：互動式 Dashboard
