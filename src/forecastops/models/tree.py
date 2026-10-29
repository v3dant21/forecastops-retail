"""Tree-based models for ForecastOps."""

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    import xgboost as xgb
except ImportError:
    xgb = None

try:
    import lightgbm as lgb
except ImportError:
    lgb = None

try:
    import catboost as cb
except ImportError:
    cb = None

from forecastops.models.base import BaseModel
from forecastops.utils.logger import setup_logger

logger = setup_logger(__name__)


class XGBoostForecaster(BaseModel):
    """XGBoost forecaster."""
    
    def __init__(self, config: Optional[dict] = None):
        """Initialize XGBoost forecaster.
        
        Args:
            config: Optional configuration
        """
        super().__init__("xgboost", config)
        self.model = None
        self.feature_columns = []
        
        self.params = {
            "objective": "reg:squarederror",
            "eval_metric": "rmse",
            "max_depth": 6,
            "learning_rate": 0.1,
            "n_estimators": 100,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "random_state": 42,
            "n_jobs": -1,
        }
        
        # Update with config if provided
        if config and "xgboost" in config:
            self.params.update(config["xgboost"])
    
    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None
    ) -> "XGBoostForecaster":
        """Train the XGBoost model.
        
        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Optional validation features
            y_val: Optional validation targets
            
        Returns:
            Self for method chaining
        """
        if xgb is None:
            raise ImportError("xgboost is not installed. Install with: pip install xgboost")
        
        # Select numeric columns
        self.feature_columns = X_train.select_dtypes(include=[np.number]).columns.tolist()
        
        X_train_numeric = X_train[self.feature_columns].fillna(0)
        
        # Create DMatrix
        dtrain = xgb.DMatrix(X_train_numeric.values, label=y_train.values)
        
        # Add validation if provided
        evals = []
        if X_val is not None and y_val is not None:
            X_val_numeric = X_val[self.feature_columns].fillna(0)
            dval = xgb.DMatrix(X_val_numeric.values, label=y_val.values)
            evals = [(dval, "val")]
        
        # Train
        self.model = xgb.train(
            {k: v for k, v in self.params.items() if k != "n_estimators"},
            dtrain,
            num_boost_round=self.params.get("n_estimators", 100),
            evals=evals,
            verbose_eval=False
        )
        
        self.trained = True
        self.logger.info("XGBoost model trained")
        
        return self
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions.
        
        Args:
            X: Features dataframe
            
        Returns:
            Predictions array
        """
        X_numeric = X[self.feature_columns].fillna(0)
        dmatrix = xgb.DMatrix(X_numeric.values)
        predictions = self.model.predict(dmatrix)
        
        return np.maximum(predictions, 0)
    
    def get_params(self) -> Dict[str, Any]:
        """Get model parameters."""
        return {"params": self.params, "feature_columns": self.feature_columns}
    
    def set_params(self, params: Dict[str, Any]) -> "XGBoostForecaster":
        """Set model parameters."""
        if "params" in params:
            self.params.update(params["params"])
        return self


class LightGBMForecaster(BaseModel):
    """LightGBM forecaster."""
    
    def __init__(self, config: Optional[dict] = None):
        """Initialize LightGBM forecaster.
        
        Args:
            config: Optional configuration
        """
        super().__init__("lightgbm", config)
        self.model = None
        self.feature_columns = []
        
        self.params = {
            "objective": "regression",
            "metric": "rmse",
            "num_leaves": 31,
            "learning_rate": 0.1,
            "n_estimators": 100,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "random_state": 42,
            "n_jobs": -1,
            "verbose": -1,
        }
        
        # Update with config if provided
        if config and "lightgbm" in config:
            self.params.update(config["lightgbm"])
    
    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None
    ) -> "LightGBMForecaster":
        """Train the LightGBM model.
        
        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Optional validation features
            y_val: Optional validation targets
            
        Returns:
            Self for method chaining
        """
        if lgb is None:
            raise ImportError("lightgbm is not installed. Install with: pip install lightgbm")
        
        # Select numeric columns
        self.feature_columns = X_train.select_dtypes(include=[np.number]).columns.tolist()
        
        X_train_numeric = X_train[self.feature_columns].fillna(0)
        
        # Create dataset
        train_data = lgb.Dataset(X_train_numeric.values, label=y_train.values)
        
        # Add validation if provided
        eval_set = None
        if X_val is not None and y_val is not None:
            X_val_numeric = X_val[self.feature_columns].fillna(0)
            eval_set = [lgb.Dataset(X_val_numeric.values, label=y_val.values, reference=train_data)]
        
        # Train
        self.model = lgb.train(
            self.params,
            train_data,
            num_boost_round=self.params.get("n_estimators", 100),
            valid_sets=eval_set,
            callbacks=[lgb.log_evaluation(period=0)]
        )
        
        self.trained = True
        self.logger.info("LightGBM model trained")
        
        return self
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions.
        
        Args:
            X: Features dataframe
            
        Returns:
            Predictions array
        """
        X_numeric = X[self.feature_columns].fillna(0)
        predictions = self.model.predict(X_numeric.values)
        
        return np.maximum(predictions, 0)
    
    def get_params(self) -> Dict[str, Any]:
        """Get model parameters."""
        return {"params": self.params, "feature_columns": self.feature_columns}
    
    def set_params(self, params: Dict[str, Any]) -> "LightGBMForecaster":
        """Set model parameters."""
        if "params" in params:
            self.params.update(params["params"])
        return self


class CatBoostForecaster(BaseModel):
    """CatBoost forecaster."""
    
    def __init__(self, config: Optional[dict] = None):
        """Initialize CatBoost forecaster.
        
        Args:
            config: Optional configuration
        """
        super().__init__("catboost", config)
        self.model = None
        self.feature_columns = []
        
        self.params = {
            "loss_function": "RMSE",
            "eval_metric": "RMSE",
            "depth": 6,
            "learning_rate": 0.1,
            "iterations": 100,
            "subsample": 0.8,
            "colsample_bylevel": 0.8,
            "random_seed": 42,
            "verbose": 0,
            "allow_writing_files": False,
        }
        
        # Update with config if provided
        if config and "catboost" in config:
            self.params.update(config["catboost"])
    
    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None
    ) -> "CatBoostForecaster":
        """Train the CatBoost model.
        
        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Optional validation features
            y_val: Optional validation targets
            
        Returns:
            Self for method chaining
        """
        if cb is None:
            raise ImportError("catboost is not installed. Install with: pip install catboost")
        
        # Select numeric columns
        self.feature_columns = X_train.select_dtypes(include=[np.number]).columns.tolist()
        
        X_train_numeric = X_train[self.feature_columns].fillna(0)
        
        # Create Pool
        train_pool = cb.Pool(X_train_numeric.values, y_train.values)
        
        # Add validation if provided
        eval_pool = None
        if X_val is not None and y_val is not None:
            X_val_numeric = X_val[self.feature_columns].fillna(0)
            eval_pool = cb.Pool(X_val_numeric.values, y_val.values)
        
        # Train
        self.model = cb.CatBoostRegressor(**self.params)
        self.model.fit(train_pool, eval_set=eval_pool, verbose=0)
        
        self.trained = True
        self.logger.info("CatBoost model trained")
        
        return self
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions.
        
        Args:
            X: Features dataframe
            
        Returns:
            Predictions array
        """
        X_numeric = X[self.feature_columns].fillna(0)
        predictions = self.model.predict(X_numeric.values)
        
        return np.maximum(predictions, 0)
    
    def get_params(self) -> Dict[str, Any]:
        """Get model parameters."""
        return {"params": self.params, "feature_columns": self.feature_columns}
    
    def set_params(self, params: Dict[str, Any]) -> "CatBoostForecaster":
        """Set model parameters."""
        if "params" in params:
            self.params.update(params["params"])
        return self


__all__ = ["XGBoostForecaster", "LightGBMForecaster", "CatBoostForecaster"]
