"""Base model interface for ForecastOps."""

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from forecastops.utils.logger import setup_logger

logger = setup_logger(__name__)


class BaseModel(ABC):
    """Abstract base class for all ForecastOps models."""
    
    def __init__(self, name: str, config: Optional[dict] = None):
        """Initialize model.
        
        Args:
            name: Model name
            config: Optional configuration
        """
        self.name = name
        self.config = config or {}
        self.logger = setup_logger(__name__)
        self.trained = False
        self.model: Optional[Any] = None
        self.feature_columns: List[str] = []
        self.target_column: str = "sales"
        self.metadata: Dict[str, Any] = {}
    
    @abstractmethod
    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None
    ) -> "BaseModel":
        """Train the model.
        
        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Optional validation features
            y_val: Optional validation targets
            
        Returns:
            Self for method chaining
        """
        pass
    
    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions.
        
        Args:
            X: Features to predict
            
        Returns:
            Predictions array
        """
        pass
    
    @abstractmethod
    def get_params(self) -> Dict[str, Any]:
        """Get model parameters.
        
        Returns:
            Dictionary of model parameters
        """
        pass
    
    @abstractmethod
    def set_params(self, params: Dict[str, Any]) -> "BaseModel":
        """Set model parameters.
        
        Args:
            params: Dictionary of parameters
            
        Returns:
            Self for method chaining
        """
        pass
    
    def score(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        metric: str = "mae"
    ) -> float:
        """Calculate prediction score.
        
        Args:
            X: Features
            y: True values
            metric: Metric name ("mae", "rmse", "wape", "smape")
            
        Returns:
            Score value
        """
        predictions = self.predict(X)
        
        if metric == "mae":
            return np.mean(np.abs(y.values - predictions))
        elif metric == "rmse":
            return np.sqrt(np.mean((y.values - predictions) ** 2))
        elif metric == "wape":
            numerator = np.sum(np.abs(y.values - predictions))
            denominator = np.sum(y.values)
            return numerator / denominator if denominator != 0 else 0.0
        elif metric == "smape":
            numerator = np.sum(np.abs(y.values - predictions))
            denominator = np.sum(np.abs(y.values) + np.abs(predictions))
            return 2 * numerator / denominator if denominator != 0 else 0.0
        else:
            raise ValueError(f"Unknown metric: {metric}")
    
    def fit_predict(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame
    ) -> Tuple[np.ndarray, "BaseModel"]:
        """Fit and predict in one call.
        
        Args:
            X_train: Training features
            y_train: Training targets
            X_test: Test features
            
        Returns:
            Tuple of (predictions, fitted model)
        """
        self.fit(X_train, y_train)
        predictions = self.predict(X_test)
        return predictions, self
    
    def set_feature_columns(self, columns: List[str]) -> "BaseModel":
        """Set feature columns.
        
        Args:
            columns: List of feature column names
            
        Returns:
            Self for method chaining
        """
        self.feature_columns = columns
        return self
    
    def set_target_column(self, column: str) -> "BaseModel":
        """Set target column name.
        
        Args:
            column: Target column name
            
        Returns:
            Self for method chaining
        """
        self.target_column = column
        return self
    
    def save(self, path: str) -> None:
        """Save model to path.
        
        Args:
            path: Path to save model
        """
        import pickle
        
        data = {
            "name": self.name,
            "trained": self.trained,
            "feature_columns": self.feature_columns,
            "target_column": self.target_column,
            "metadata": self.metadata,
            "params": self.get_params(),
            "format_version": 2,
            "state": {k: v for k, v in self.__dict__.items() if k != "logger"}
        }
        
        if self.model is not None:
            data["model"] = self.model
        
        with open(path, "wb") as f:
            pickle.dump(data, f)
        
        self.logger.info(f"Saved model to {path}")
    
    def load(self, path: str) -> "BaseModel":
        """Load model from path.
        
        Args:
            path: Path to load model from
            
        Returns:
            Self for method chaining
        """
        import pickle
        
        with open(path, "rb") as f:
            data = pickle.load(f)
        
        self.name = data["name"]
        self.trained = data["trained"]
        self.feature_columns = data["feature_columns"]
        self.target_column = data["target_column"]
        self.metadata = data.get("metadata", {})
        
        if "model" in data:
            self.model = data["model"]
        
        if "params" in data:
            self.set_params(data["params"])
        
        if "state" in data:
            self.__dict__.update(data["state"])

        if "state" not in data and self.name not in {"xgboost", "lightgbm", "catboost"}:
            raise ValueError("Legacy artifact lacks preprocessing/history; retrain this model")

        self.logger.info(f"Loaded model from {path}")
        return self
    
    def __repr__(self) -> str:
        """String representation."""
        return f"{self.__class__.__name__}(name={self.name}, trained={self.trained})"
