"""
CCL 樹脂配方預測模型
====================
使用 Random Forest 和 XGBoost 進行多目標預測
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
import json
import pickle
from pathlib import Path

# 機器學習
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# 嘗試導入 XGBoost (可選)
try:
    from xgboost import XGBRegressor
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    print("Warning: XGBoost not installed. Using GradientBoosting as fallback.")


@dataclass
class ModelConfig:
    """模型配置"""
    input_cols: List[str] = None
    output_cols: List[str] = None
    test_size: float = 0.2
    random_state: int = 42
    n_estimators: int = 100
    
    def __post_init__(self):
        if self.input_cols is None:
            self.input_cols = [
                'Hardener_Eq_Ratio', 'Filler_Vol_Pct', 'FR_Wt_Pct',
                'Toughener_Wt_Pct', 'Wash_Cycles', 'Residual_Cl_ppm'
            ]
        if self.output_cols is None:
            self.output_cols = [
                'Dk_10GHz', 'Df_10GHz', 'Peel_Strength_N_mm', 'Tg_C', 'CTE_ppm'
            ]


class CCLPredictor:
    """CCL 樹脂配方預測器"""
    
    def __init__(self, config: Optional[ModelConfig] = None):
        self.config = config or ModelConfig()
        self.models: Dict[str, any] = {}
        self.scaler_X: Optional[StandardScaler] = None
        self.scaler_y: Dict[str, StandardScaler] = {}
        self.feature_importance: Dict[str, pd.DataFrame] = {}
        self.metrics: Dict[str, Dict[str, float]] = {}
        self._is_trained = False
        
    def load_data(self, filepath: str) -> pd.DataFrame:
        """載入數據"""
        self.df = pd.read_csv(filepath)
        print(f"✅ 載入 {len(self.df)} 筆數據")
        return self.df
    
    def prepare_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """準備訓練/測試數據"""
        X = self.df[self.config.input_cols].values
        y = self.df[self.config.output_cols].values
        
        # 分割數據
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=self.config.test_size,
            random_state=self.config.random_state
        )
        
        # 標準化輸入
        self.scaler_X = StandardScaler()
        X_train_scaled = self.scaler_X.fit_transform(X_train)
        X_test_scaled = self.scaler_X.transform(X_test)
        
        print(f"✅ 訓練集: {len(X_train)} 筆, 測試集: {len(X_test)} 筆")
        
        return X_train_scaled, X_test_scaled, y_train, y_test
    
    def train(self, model_type: str = 'random_forest') -> Dict[str, float]:
        """
        訓練模型
        
        Parameters
        ----------
        model_type : str
            'random_forest' 或 'xgboost'
        
        Returns
        -------
        Dict[str, float]
            各目標的 R² 分數
        """
        X_train, X_test, y_train, y_test = self.prepare_data()
        
        print(f"\n🚀 開始訓練 {model_type.upper()} 模型...")
        print("=" * 50)
        
        # 為每個輸出變數訓練獨立模型
        for i, target in enumerate(self.config.output_cols):
            print(f"\n訓練 {target}...")
            
            # 選擇模型
            if model_type == 'random_forest':
                model = RandomForestRegressor(
                    n_estimators=self.config.n_estimators,
                    random_state=self.config.random_state,
                    n_jobs=-1
                )
            elif model_type == 'xgboost' and HAS_XGBOOST:
                model = XGBRegressor(
                    n_estimators=self.config.n_estimators,
                    random_state=self.config.random_state,
                    n_jobs=-1,
                    verbosity=0
                )
            else:
                model = GradientBoostingRegressor(
                    n_estimators=self.config.n_estimators,
                    random_state=self.config.random_state
                )
            
            # 訓練
            y_target_train = y_train[:, i]
            y_target_test = y_test[:, i]
            
            model.fit(X_train, y_target_train)
            
            # 預測
            y_pred_train = model.predict(X_train)
            y_pred_test = model.predict(X_test)
            
            # 評估
            r2_train = r2_score(y_target_train, y_pred_train)
            r2_test = r2_score(y_target_test, y_pred_test)
            mae_test = mean_absolute_error(y_target_test, y_pred_test)
            rmse_test = np.sqrt(mean_squared_error(y_target_test, y_pred_test))
            
            self.metrics[target] = {
                'r2_train': r2_train,
                'r2_test': r2_test,
                'mae': mae_test,
                'rmse': rmse_test
            }
            
            print(f"  R² (train): {r2_train:.4f}")
            print(f"  R² (test):  {r2_test:.4f}")
            print(f"  MAE:        {mae_test:.4f}")
            print(f"  RMSE:       {rmse_test:.4f}")
            
            # 儲存模型
            self.models[target] = model
            
            # 特徵重要性
            if hasattr(model, 'feature_importances_'):
                importance = pd.DataFrame({
                    'feature': self.config.input_cols,
                    'importance': model.feature_importances_
                }).sort_values('importance', ascending=False)
                self.feature_importance[target] = importance
        
        self._is_trained = True
        print("\n" + "=" * 50)
        print("✅ 訓練完成!")
        
        return {k: v['r2_test'] for k, v in self.metrics.items()}
    
    def predict(self, formulation: Dict[str, float]) -> Dict[str, float]:
        """
        預測單一配方的物理性質
        
        Parameters
        ----------
        formulation : Dict[str, float]
            配方參數，例如:
            {
                'Hardener_Eq_Ratio': 0.95,
                'Filler_Vol_Pct': 35,
                'FR_Wt_Pct': 5,
                'Toughener_Wt_Pct': 3,
                'Wash_Cycles': 4,
                'Residual_Cl_ppm': 15
            }
        
        Returns
        -------
        Dict[str, float]
            預測的物理性質
        """
        if not self._is_trained:
            raise ValueError("模型尚未訓練，請先呼叫 train()")
        
        # 準備輸入
        X = np.array([[formulation.get(col, 0) for col in self.config.input_cols]])
        X_scaled = self.scaler_X.transform(X)
        
        # 預測各目標
        predictions = {}
        for target in self.config.output_cols:
            pred = self.models[target].predict(X_scaled)[0]
            predictions[target] = round(pred, 4)
        
        return predictions
    
    def predict_batch(self, formulations: pd.DataFrame) -> pd.DataFrame:
        """批次預測多個配方"""
        if not self._is_trained:
            raise ValueError("模型尚未訓練，請先呼叫 train()")
        
        X = formulations[self.config.input_cols].values
        X_scaled = self.scaler_X.transform(X)
        
        results = formulations.copy()
        for target in self.config.output_cols:
            results[f'{target}_pred'] = self.models[target].predict(X_scaled)
        
        return results
    
    def get_feature_importance(self, target: str = None) -> pd.DataFrame:
        """取得特徵重要性"""
        if target:
            return self.feature_importance.get(target)
        
        # 合併所有目標的特徵重要性
        all_importance = []
        for t, imp in self.feature_importance.items():
            imp_copy = imp.copy()
            imp_copy['target'] = t
            all_importance.append(imp_copy)
        
        return pd.concat(all_importance, ignore_index=True)
    
    def get_metrics_summary(self) -> pd.DataFrame:
        """取得模型評估摘要"""
        return pd.DataFrame(self.metrics).T
    
    def save(self, filepath: str):
        """儲存模型"""
        save_dict = {
            'models': self.models,
            'scaler_X': self.scaler_X,
            'config': self.config,
            'metrics': self.metrics,
            'feature_importance': self.feature_importance
        }
        with open(filepath, 'wb') as f:
            pickle.dump(save_dict, f)
        print(f"✅ 模型已儲存至: {filepath}")
    
    def load(self, filepath: str):
        """載入模型"""
        with open(filepath, 'rb') as f:
            save_dict = pickle.load(f)
        
        self.models = save_dict['models']
        self.scaler_X = save_dict['scaler_X']
        self.config = save_dict['config']
        self.metrics = save_dict['metrics']
        self.feature_importance = save_dict['feature_importance']
        self._is_trained = True
        print(f"✅ 模型已載入: {filepath}")


def print_feature_importance_chart(predictor: CCLPredictor, target: str):
    """印出特徵重要性圖表 (ASCII)"""
    imp = predictor.get_feature_importance(target)
    if imp is None:
        print(f"No feature importance for {target}")
        return
    
    print(f"\n📊 {target} 特徵重要性")
    print("-" * 50)
    
    max_imp = imp['importance'].max()
    for _, row in imp.iterrows():
        bar_len = int(row['importance'] / max_imp * 30)
        bar = '█' * bar_len
        print(f"{row['feature']:25s} {bar} {row['importance']:.3f}")


if __name__ == '__main__':
    # 測試
    print("CCL 樹脂配方預測模型")
    print("=" * 50)
    
    # 建立預測器
    predictor = CCLPredictor()
    
    # 載入數據
    predictor.load_data('../data/ccl_resin_simulation.csv')
    
    # 訓練模型
    scores = predictor.train(model_type='random_forest')
    
    # 顯示評估結果
    print("\n📈 模型評估摘要:")
    print(predictor.get_metrics_summary())
    
    # 顯示特徵重要性
    for target in predictor.config.output_cols:
        print_feature_importance_chart(predictor, target)
    
    # 測試預測
    print("\n🔮 測試預測:")
    test_formulation = {
        'Hardener_Eq_Ratio': 0.95,
        'Filler_Vol_Pct': 35,
        'FR_Wt_Pct': 5,
        'Toughener_Wt_Pct': 3,
        'Wash_Cycles': 4,
        'Residual_Cl_ppm': 15
    }
    print(f"輸入配方: {test_formulation}")
    predictions = predictor.predict(test_formulation)
    print(f"預測結果: {predictions}")
    
    # 儲存模型
    predictor.save('ccl_predictor.pkl')
