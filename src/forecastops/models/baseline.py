"""Baseline models for ForecastOps."""

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import StandardScaler

from forecastops.models.base import BaseModel
from forecastops.utils.logger import setup_logger

logger = setup_logger(__name__)


class NaiveForecaster(BaseModel):
    """Naive baseline forecaster - predicts last known value."""
    
    def __init__(self, config: Optional[dict] = None):
        """Initialize naive forecaster.
        
        Args:
            config: Optional configuration
        """
        super().__init__("naive", config)
        self.last_values: Dict[str, float] = {}
    
    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None
    ) -> "NaiveForecaster":
        """Train the naive forecaster.
        
        Args:
            X_train: Training features (used for product/store keys)
            y_train: Training targets
            X_val: Optional validation features
            y_val: Optional validation targets
            
        Returns:
            Self for method chaining
        """
        self.feature_columns = ["product_id", "store_id"]
        if not set(self.feature_columns).issubset(X_train.columns):
            raise ValueError("Baseline models require product_id and store_id")
        # Store the last value for each product-store combination
        self.feature_columns = ["product_id", "store_id"]
        if not set(self.feature_columns).issubset(X_train.columns):
            raise ValueError("Baseline models require product_id and store_id")
        if "product_id" in X_train.columns and "store_id" in X_train.columns:
            df = X_train.copy()
            df["sales"] = y_train.values
            
            for (product_id, store_id), group in df.groupby(["product_id", "store_id"]):
                if "date" in group:
                    group = group.sort_values("date")
                self.last_values[(product_id, store_id)] = group["sales"].iloc[-1]
        
        self.trained = True
        self.logger.info(f"Naive forecaster trained with {len(self.last_values)} product-store combinations")
        
        return self
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions using naive method.
        
        Args:
            X: Features dataframe with product_id and store_id
            
        Returns:
            Predictions array
        """
        predictions = []
        
        for _, row in X.iterrows():
            key = (row.get("product_id"), row.get("store_id"))
            if key in self.last_values:
                predictions.append(self.last_values[key])
            else:
                # Fallback: use global mean if key not found
                predictions.append(0.0)
        
        return np.array(predictions)
    
    def get_params(self) -> Dict[str, Any]:
        """Get model parameters."""
        return {"last_values_count": len(self.last_values)}
    
    def set_params(self, params: Dict[str, Any]) -> "NaiveForecaster":
        """Set model parameters."""
        return self
    
    def score(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        metric: str = "mae"
    ) -> float:
        """Calculate score, handling unseen product-store combinations."""
        predictions = self.predict(X)
        
        if metric == "mae":
            return np.mean(np.abs(y.values - predictions))
        elif metric == "rmse":
            return np.sqrt(np.mean((y.values - predictions) ** 2))
        elif metric == "wape":
            numerator = np.sum(np.abs(y.values - predictions))
            denominator = np.sum(y.values)
            return numerator / denominator if denominator != 0 else 0.0
        else:
            return super().score(X, y, metric)


class SeasonalNaiveForecaster(BaseModel):
    """Seasonal naive baseline forecaster."""
    
    def __init__(self, season_length: int = 7, config: Optional[dict] = None):
        """Initialize seasonal naive forecaster.
        
        Args:
            season_length: Length of seasonal period
            config: Optional configuration
        """
        super().__init__("seasonal_naive", config)
        self.season_length = season_length
        self.seasonal_values: Dict[str, List[float]] = {}
        self.last_dates = {}
    
    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None
    ) -> "SeasonalNaiveForecaster":
        """Train the seasonal naive forecaster.
        
        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Optional validation features
            y_val: Optional validation targets
            
        Returns:
            Self for method chaining
        """
        self.feature_columns = ["product_id", "store_id"]
        if not set(self.feature_columns).issubset(X_train.columns):
            raise ValueError("Baseline models require product_id and store_id")
        if "product_id" in X_train.columns and "store_id" in X_train.columns:
            df = X_train.copy()
            df["sales"] = y_train.values
            
            for (product_id, store_id), group in df.groupby(["product_id", "store_id"]):
                if "date" in group:
                    group = group.sort_values("date")
                    self.last_dates[(product_id, store_id)] = pd.Timestamp(group["date"].iloc[-1])
                sales = group["sales"].tolist()
                if len(sales) >= self.season_length:
                    self.seasonal_values[(product_id, store_id)] = sales[-self.season_length:]
                else:
                    self.seasonal_values[(product_id, store_id)] = sales
        
        self.trained = True
        self.logger.info(f"Seasonal naive trained with {len(self.seasonal_values)} combinations")
        
        return self
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make seasonal naive predictions.
        
        Args:
            X: Features dataframe
            
        Returns:
            Predictions array
        """
        predictions = []
        
        positions = {}
        for _, row in X.iterrows():
            key = (row.get("product_id"), row.get("store_id"))
            
            if key in self.seasonal_values:
                seasonal = self.seasonal_values[key]
                # Predict the value from the same position in the season
                if "date" in row and key in self.last_dates:
                    position = (pd.Timestamp(row["date"]) - self.last_dates[key]).days - 1
                else:
                    position = positions.get(key, 0)
                    positions[key] = position + 1
                position %= len(seasonal)
                predictions.append(seasonal[position] if position < len(seasonal) else seasonal[-1])
            else:
                predictions.append(0.0)
        
        return np.array(predictions)
    
    def get_params(self) -> Dict[str, Any]:
        """Get model parameters."""
        return {
            "season_length": self.season_length,
            "seasonal_values_count": len(self.seasonal_values)
        }
    
    def set_params(self, params: Dict[str, Any]) -> "SeasonalNaiveForecaster":
        """Set model parameters."""
        if "season_length" in params:
            self.season_length = params["season_length"]
        return self


class LinearRegressor(BaseModel):
    """Linear regression baseline."""
    
    def __init__(self, config: Optional[dict] = None, use_ridge: bool = False):
        """Initialize linear regressor.
        
        Args:
            config: Optional configuration
            use_ridge: Use Ridge regularization
        """
        super().__init__("linear", config)
        self.use_ridge = use_ridge
        self.model: Optional[Any] = None
        self.scaler: Optional[StandardScaler] = None
        self.feature_columns: List[str] = []
    
    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None
    ) -> "LinearRegressor":
        """Train the linear regressor.
        
        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Optional validation features
            y_val: Optional validation targets
            
        Returns:
            Self for method chaining
        """
        # Select numeric columns
        self.feature_columns = X_train.select_dtypes(include=[np.number]).columns.tolist()
        
        X_train_numeric = X_train[self.feature_columns].copy()
        X_train_numeric = X_train_numeric.fillna(0)
        
        # Scale features
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train_numeric)
        
        # Create and train model
        if self.use_ridge:
            self.model = Ridge(alpha=1.0)
        else:
            self.model = LinearRegression()
        
        self.model.fit(X_train_scaled, y_train)
        self.trained = True
        
        self.logger.info(f"Linear regressor trained with {len(self.feature_columns)} features")
        
        return self
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions.
        
        Args:
            X: Features dataframe
            
        Returns:
            Predictions array
        """
        X_numeric = X[self.feature_columns].copy()
        X_numeric = X_numeric.fillna(0)
        
        X_scaled = self.scaler.transform(X_numeric)
        predictions = self.model.predict(X_scaled)
        
        # Ensure non-negative predictions
        return np.maximum(predictions, 0)
    
    def get_params(self) -> Dict[str, Any]:
        """Get model parameters."""
        return {
            "use_ridge": self.use_ridge,
            "feature_columns": self.feature_columns,
            "n_features": len(self.feature_columns)
        }
    
    def set_params(self, params: Dict[str, Any]) -> "LinearRegressor":
        """Set model parameters."""
        if "use_ridge" in params:
            self.use_ridge = params["use_ridge"]
        return self


# Export all baseline models
__all__ = ["NaiveForecaster", "SeasonalNaiveForecaster", "LinearRegressor"]
