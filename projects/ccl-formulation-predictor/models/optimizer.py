"""
CCL 樹脂配方優化器 (反向設計)
=============================
給定目標規格，搜尋最佳配方
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Callable
from dataclasses import dataclass, field
import random

from predictor import CCLPredictor, ModelConfig


@dataclass
class TargetSpec:
    """目標規格"""
    name: str
    target_type: str  # 'min', 'max', 'range'
    value: float = None
    min_value: float = None
    max_value: float = None
    weight: float = 1.0  # 權重，用於多目標優化
    
    def is_satisfied(self, actual: float) -> bool:
        """檢查是否滿足規格"""
        if self.target_type == 'min':
            return actual >= self.value
        elif self.target_type == 'max':
            return actual <= self.value
        elif self.target_type == 'range':
            return self.min_value <= actual <= self.max_value
        return False
    
    def get_penalty(self, actual: float) -> float:
        """計算違反規格的懲罰值"""
        if self.target_type == 'min':
            if actual >= self.value:
                return 0
            return (self.value - actual) * self.weight
        elif self.target_type == 'max':
            if actual <= self.value:
                return 0
            return (actual - self.value) * self.weight
        elif self.target_type == 'range':
            if self.min_value <= actual <= self.max_value:
                return 0
            if actual < self.min_value:
                return (self.min_value - actual) * self.weight
            return (actual - self.max_value) * self.weight
        return 0


@dataclass
class FormulationBounds:
    """配方參數範圍"""
    Hardener_Eq_Ratio: Tuple[float, float] = (0.75, 1.15)
    Filler_Vol_Pct: Tuple[float, float] = (15, 55)
    FR_Wt_Pct: Tuple[float, float] = (3, 8)
    Toughener_Wt_Pct: Tuple[float, float] = (0, 8)
    Wash_Cycles: Tuple[int, int] = (2, 5)
    Residual_Cl_ppm: Tuple[float, float] = (5, 45)
    
    def get_bounds(self) -> Dict[str, Tuple]:
        return {
            'Hardener_Eq_Ratio': self.Hardener_Eq_Ratio,
            'Filler_Vol_Pct': self.Filler_Vol_Pct,
            'FR_Wt_Pct': self.FR_Wt_Pct,
            'Toughener_Wt_Pct': self.Toughener_Wt_Pct,
            'Wash_Cycles': self.Wash_Cycles,
            'Residual_Cl_ppm': self.Residual_Cl_ppm
        }


class CCLOptimizer:
    """CCL 配方優化器"""
    
    def __init__(self, predictor: CCLPredictor, bounds: Optional[FormulationBounds] = None):
        self.predictor = predictor
        self.bounds = bounds or FormulationBounds()
        self.target_specs: List[TargetSpec] = []
        
    def set_targets(self, specs: Dict[str, Dict]):
        """
        設定目標規格
        
        Parameters
        ----------
        specs : Dict[str, Dict]
            例如:
            {
                'Dk_10GHz': {'type': 'max', 'value': 3.5, 'weight': 2.0},
                'Df_10GHz': {'type': 'max', 'value': 0.010},
                'Peel_Strength_N_mm': {'type': 'min', 'value': 0.7},
                'Tg_C': {'type': 'min', 'value': 160},
                'CTE_ppm': {'type': 'max', 'value': 35}
            }
        """
        self.target_specs = []
        for name, spec in specs.items():
            target_type = spec.get('type', 'max')
            value = spec.get('value')
            min_val = spec.get('min')
            max_val = spec.get('max')
            weight = spec.get('weight', 1.0)
            
            self.target_specs.append(TargetSpec(
                name=name,
                target_type=target_type,
                value=value,
                min_value=min_val,
                max_value=max_val,
                weight=weight
            ))
        
        print(f"✅ 已設定 {len(self.target_specs)} 個目標規格")
    
    def _random_formulation(self) -> Dict[str, float]:
        """生成隨機配方"""
        bounds = self.bounds.get_bounds()
        formulation = {}
        for param, (low, high) in bounds.items():
            if param == 'Wash_Cycles':
                formulation[param] = random.randint(int(low), int(high))
            else:
                formulation[param] = random.uniform(low, high)
        return formulation
    
    def _evaluate_formulation(self, formulation: Dict[str, float]) -> Tuple[Dict, float, bool]:
        """
        評估配方
        
        Returns
        -------
        predictions : Dict
            預測的物理性質
        total_penalty : float
            總懲罰值 (越低越好)
        all_satisfied : bool
            是否滿足所有規格
        """
        predictions = self.predictor.predict(formulation)
        
        total_penalty = 0
        all_satisfied = True
        
        for spec in self.target_specs:
            actual = predictions.get(spec.name, 0)
            penalty = spec.get_penalty(actual)
            total_penalty += penalty
            
            if not spec.is_satisfied(actual):
                all_satisfied = False
        
        return predictions, total_penalty, all_satisfied
    
    def grid_search(self, n_samples: int = 10000, n_results: int = 10) -> pd.DataFrame:
        """
        網格搜尋 (隨機採樣)
        
        Parameters
        ----------
        n_samples : int
            採樣數量
        n_results : int
            返回前 N 個最佳結果
        
        Returns
        -------
        pd.DataFrame
            最佳配方列表
        """
        print(f"\n🔍 開始網格搜尋 ({n_samples} 個配方)...")
        
        results = []
        satisfied_count = 0
        
        for i in range(n_samples):
            formulation = self._random_formulation()
            predictions, penalty, satisfied = self._evaluate_formulation(formulation)
            
            result = {**formulation, **predictions, 'penalty': penalty, 'satisfied': satisfied}
            results.append(result)
            
            if satisfied:
                satisfied_count += 1
            
            if (i + 1) % 2000 == 0:
                print(f"  進度: {i+1}/{n_samples}, 符合規格: {satisfied_count}")
        
        # 轉換為 DataFrame 並排序
        df_results = pd.DataFrame(results)
        df_results = df_results.sort_values('penalty').reset_index(drop=True)
        
        print(f"\n✅ 搜尋完成!")
        print(f"   符合所有規格: {satisfied_count}/{n_samples} ({satisfied_count/n_samples*100:.1f}%)")
        
        return df_results.head(n_results)
    
    def genetic_algorithm(
        self, 
        population_size: int = 100,
        generations: int = 50,
        mutation_rate: float = 0.1,
        n_results: int = 10
    ) -> pd.DataFrame:
        """
        遺傳演算法優化
        
        Parameters
        ----------
        population_size : int
            族群大小
        generations : int
            演化代數
        mutation_rate : float
            突變率
        n_results : int
            返回前 N 個最佳結果
        
        Returns
        -------
        pd.DataFrame
            最佳配方列表
        """
        print(f"\n🧬 開始遺傳演算法優化...")
        print(f"   族群大小: {population_size}, 代數: {generations}")
        
        bounds = self.bounds.get_bounds()
        param_names = list(bounds.keys())
        
        # 初始化族群
        population = [self._random_formulation() for _ in range(population_size)]
        
        best_penalty = float('inf')
        best_formulation = None
        
        for gen in range(generations):
            # 評估適應度
            fitness_scores = []
            for individual in population:
                _, penalty, _ = self._evaluate_formulation(individual)
                fitness_scores.append(penalty)
            
            # 記錄最佳
            min_penalty_idx = np.argmin(fitness_scores)
            if fitness_scores[min_penalty_idx] < best_penalty:
                best_penalty = fitness_scores[min_penalty_idx]
                best_formulation = population[min_penalty_idx].copy()
            
            # 選擇 (輪盤賭)
            # 將懲罰轉換為適應度 (越低越好 -> 越高越好)
            max_penalty = max(fitness_scores) + 1
            fitness = [max_penalty - p for p in fitness_scores]
            total_fitness = sum(fitness)
            probabilities = [f / total_fitness for f in fitness]
            
            # 選擇父代
            selected_indices = np.random.choice(
                len(population), 
                size=population_size, 
                p=probabilities, 
                replace=True
            )
            selected = [population[i] for i in selected_indices]
            
            # 交叉
            new_population = []
            for i in range(0, population_size, 2):
                parent1 = selected[i]
                parent2 = selected[min(i + 1, population_size - 1)]
                
                # 單點交叉
                crossover_point = random.randint(1, len(param_names) - 1)
                child1 = {}
                child2 = {}
                for j, param in enumerate(param_names):
                    if j < crossover_point:
                        child1[param] = parent1[param]
                        child2[param] = parent2[param]
                    else:
                        child1[param] = parent2[param]
                        child2[param] = parent1[param]
                
                new_population.extend([child1, child2])
            
            # 突變
            for individual in new_population:
                if random.random() < mutation_rate:
                    # 隨機選擇一個參數進行突變
                    param = random.choice(param_names)
                    low, high = bounds[param]
                    if param == 'Wash_Cycles':
                        individual[param] = random.randint(int(low), int(high))
                    else:
                        individual[param] = random.uniform(low, high)
            
            population = new_population[:population_size]
            
            # 精英保留
            population[0] = best_formulation.copy()
            
            if (gen + 1) % 10 == 0:
                print(f"  代數 {gen+1}/{generations}, 最佳懲罰值: {best_penalty:.4f}")
        
        # 最終評估
        results = []
        for individual in population:
            predictions, penalty, satisfied = self._evaluate_formulation(individual)
            result = {**individual, **predictions, 'penalty': penalty, 'satisfied': satisfied}
            results.append(result)
        
        df_results = pd.DataFrame(results)
        df_results = df_results.sort_values('penalty').reset_index(drop=True)
        df_results = df_results.drop_duplicates(subset=list(bounds.keys())).head(n_results)
        
        print(f"\n✅ 遺傳演算法完成!")
        
        return df_results
    
    def search(
        self, 
        method: str = 'grid',
        n_results: int = 10,
        **kwargs
    ) -> pd.DataFrame:
        """
        搜尋最佳配方
        
        Parameters
        ----------
        method : str
            'grid' (網格搜尋) 或 'genetic' (遺傳演算法)
        n_results : int
            返回結果數量
        **kwargs
            傳遞給具體方法的參數
        
        Returns
        -------
        pd.DataFrame
            最佳配方列表
        """
        if method == 'grid':
            return self.grid_search(n_results=n_results, **kwargs)
        elif method == 'genetic':
            return self.genetic_algorithm(n_results=n_results, **kwargs)
        else:
            raise ValueError(f"Unknown method: {method}")


def format_results(df: pd.DataFrame, target_specs: List[TargetSpec]) -> str:
    """格式化輸出結果"""
    output = []
    output.append("\n" + "=" * 80)
    output.append("🏆 最佳配方搜尋結果")
    output.append("=" * 80)
    
    for i, row in df.iterrows():
        output.append(f"\n📋 配方 #{i+1} {'✅ 符合規格' if row['satisfied'] else '❌ 未完全符合'}")
        output.append("-" * 40)
        
        # 配方參數
        output.append("【配方參數】")
        output.append(f"  硬化劑當量比:    {row['Hardener_Eq_Ratio']:.3f}")
        output.append(f"  填料體積%:       {row['Filler_Vol_Pct']:.1f} vol%")
        output.append(f"  阻燃劑重量%:     {row['FR_Wt_Pct']:.2f} wt%")
        output.append(f"  增韌劑重量%:     {row['Toughener_Wt_Pct']:.2f} wt%")
        output.append(f"  水洗次數:        {int(row['Wash_Cycles'])} 次")
        output.append(f"  殘留氯離子:      {row['Residual_Cl_ppm']:.1f} ppm")
        
        # 預測性質
        output.append("\n【預測物理性質】")
        for spec in target_specs:
            actual = row[spec.name]
            satisfied = spec.is_satisfied(actual)
            status = "✅" if satisfied else "❌"
            
            if spec.target_type == 'max':
                target_str = f"≤ {spec.value}"
            elif spec.target_type == 'min':
                target_str = f"≥ {spec.value}"
            else:
                target_str = f"{spec.min_value} ~ {spec.max_value}"
            
            output.append(f"  {spec.name:25s}: {actual:8.4f} (目標: {target_str}) {status}")
        
        output.append(f"\n  總懲罰值: {row['penalty']:.4f}")
    
    return "\n".join(output)


if __name__ == '__main__':
    print("CCL 樹脂配方優化器")
    print("=" * 50)
    
    # 載入訓練好的模型
    predictor = CCLPredictor()
    predictor.load_data('../data/ccl_resin_simulation.csv')
    predictor.train(model_type='random_forest')
    
    # 建立優化器
    optimizer = CCLOptimizer(predictor)
    
    # 設定目標規格 (5G 高頻 CCL 規格)
    target_specs = {
        'Dk_10GHz': {'type': 'max', 'value': 3.5, 'weight': 2.0},
        'Df_10GHz': {'type': 'max', 'value': 0.010, 'weight': 2.0},
        'Peel_Strength_N_mm': {'type': 'min', 'value': 0.7, 'weight': 1.5},
        'Tg_C': {'type': 'min', 'value': 160, 'weight': 1.0},
        'CTE_ppm': {'type': 'max', 'value': 35, 'weight': 1.0}
    }
    optimizer.set_targets(target_specs)
    
    # 方法 1: 網格搜尋
    print("\n" + "=" * 50)
    print("方法 1: 網格搜尋")
    print("=" * 50)
    results_grid = optimizer.search(method='grid', n_samples=10000, n_results=5)
    print(format_results(results_grid, optimizer.target_specs))
    
    # 方法 2: 遺傳演算法
    print("\n" + "=" * 50)
    print("方法 2: 遺傳演算法")
    print("=" * 50)
    results_ga = optimizer.search(method='genetic', population_size=100, generations=30, n_results=5)
    print(format_results(results_ga, optimizer.target_specs))
