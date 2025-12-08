# Google AI Studio Prompt - 生成製程流程圖

## 使用方式

將以下 Prompt 完整複製到 Google AI Studio，AI 會生成 Mermaid.js 流程圖語法。

生成後可貼到 [Mermaid Live Editor](https://mermaid.live/) 查看視覺化結果。

---

## Prompt（複製以下內容）

```
You are an expert Manufacturing Systems Architect specializing in Process Flow Modeling.

Based on the following complex outsourced manufacturing process, generate a detailed Mermaid.js flowchart (using graph TD).

## Process Description

### Station A - Receiving (收料站)
- **Inputs**: Raw materials from TWO sources:
  - Mother Company (M): Supplies raw material with ~60% moisture
  - P Corporation (P): Supplies raw material with ~55% moisture
- **Process**: Materials are MIXED together at this station
- **Outputs**: Mixed raw material → Station B
- **Has Remnants**: Yes (leftover from previous batch)
- **Has Waste**: Yes

### Station B - Pre-processing (前處理站)
- **Owner**: CG Corporation (100%)
- **Inputs**: Mixed material from Station A + Remnants from previous batch
- **Process**: Pre-treatment (moisture may change)
- **Outputs**: Pre-processed material → Station C
- **Has Remnants**: Yes
- **Has Waste**: Yes

### Station C - Divergence Point with Hydration (分流/加水站)
- **Process**: 
  1. ADD WATER to the material (moisture increases significantly)
  2. SPLIT material into two parallel paths
- **Split Ratio**: Variable per batch (not fixed)
- **Outputs**:
  - Path 1 → CG Corporation's D-Line
  - Path 2 → P Corporation's D-Line
- **Has Remnants**: Yes
- **Has Waste**: Yes

### Station D - Parallel Processing (平行加工站)
Two separate lines running in parallel:

**D-CG Line (CG Corporation)**
- **Inputs**: Material from C-Path 1 + Own remnants
- **Process**: Dehydration (moisture decreases)
- **Outputs**: Semi-finished product → Station E
- **Has Remnants**: Yes
- **Has Waste**: Yes

**D-P Line (P Corporation)**
- **Inputs**: Material from C-Path 2 + Own remnants
- **Process**: Dehydration (moisture decreases)
- **Outputs**: Semi-finished product → Station E
- **Has Remnants**: Yes
- **Has Waste**: Yes

### Station E - Convergence & Final Processing (混合/成品站)
- **Critical Mixing Point**
- **Inputs**: 
  1. Output from D-CG Line
  2. Output from D-P Line
  3. Remnants from PREVIOUS BATCH of Station E (cross-batch contamination)
- **Process**: 
  1. Mixing all inputs
  2. Final dehydration → Powder (moisture ~5%)
- **Outputs**: Final powder product
- **Has Remnants**: Yes (feeds back to next batch)
- **Has Waste**: Yes

## Diagram Requirements

1. **Color Coding by Owner**:
   - Mother Company (M): Blue nodes
   - P Corporation: Green nodes
   - CG Corporation: Orange nodes
   - Mixed/Shared: Purple nodes

2. **Line Styles**:
   - Solid arrows: Normal material flow
   - Dashed arrows: Remnant loops (previous batch → current batch)
   - Thick arrows: Water addition at Station C

3. **Labels**:
   - Show moisture change direction on each arrow (↑ or ↓)
   - Mark "SPLIT" at Station C divergence
   - Mark "MERGE" at Station E convergence
   - Show "Remnant Loop" labels on dashed lines

4. **Subgraphs**:
   - Group Station D-CG and D-P in a "Parallel Processing" subgraph

5. **Special Annotations**:
   - Add a note at Station E: "Critical: Cross-batch remnant mixing"
   - Add a note at Station C: "Variable split ratio per batch"

Please generate the complete Mermaid.js code that can be directly rendered.
```

---

## 預期輸出範例

AI 應該會生成類似以下的 Mermaid 語法：

```mermaid
graph TD
    subgraph Input["原料來源"]
        M[("母公司 M<br/>含水率 60%")]:::mother
        P_in[("P公司供料<br/>含水率 55%")]:::pcorp
    end
    
    subgraph StationA["站點 A - 收料"]
        A[混合原料]:::mixed
        A_rem[(餘料)]:::remnant
    end
    
    M -->|原料| A
    P_in -->|原料| A
    A_rem -.->|上批餘料| A
    
    subgraph StationB["站點 B - 前處理 (CG)"]
        B[前處理]:::cgcorp
        B_rem[(餘料)]:::remnant
    end
    
    A -->|"含水率變化"| B
    B_rem -.->|上批餘料| B
    
    subgraph StationC["站點 C - 分流/加水"]
        C[加水混合]:::mixed
        C_rem[(餘料)]:::remnant
    end
    
    B -->|物料| C
    C_rem -.->|上批餘料| C
    
    subgraph ParallelD["站點 D - 平行加工"]
        D_CG[D-CG 脫水<br/>CG公司]:::cgcorp
        D_P[D-P 脫水<br/>P公司]:::pcorp
        D_CG_rem[(CG餘料)]:::remnant
        D_P_rem[(P餘料)]:::remnant
    end
    
    C ==>|"加水 ↑<br/>分流至CG"| D_CG
    C ==>|"加水 ↑<br/>分流至P"| D_P
    D_CG_rem -.->|上批餘料| D_CG
    D_P_rem -.->|上批餘料| D_P
    
    subgraph StationE["站點 E - 混合成品"]
        E[混合脫水<br/>→ 粉末]:::mixed
        E_rem[(餘料)]:::remnant
    end
    
    D_CG -->|"脫水 ↓"| E
    D_P -->|"脫水 ↓"| E
    E_rem -.->|"⚠️ 跨批餘料混入"| E
    
    E -->|"成品粉末<br/>含水率 5%"| Final[最終產品]:::product
    
    classDef mother fill:#3498db,color:#fff
    classDef pcorp fill:#27ae60,color:#fff
    classDef cgcorp fill:#e67e22,color:#fff
    classDef mixed fill:#9b59b6,color:#fff
    classDef remnant fill:#95a5a6,color:#fff,stroke-dasharray: 5 5
    classDef product fill:#f39c12,color:#fff
```
