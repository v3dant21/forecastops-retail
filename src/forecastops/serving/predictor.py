"""Model predictor for ForecastOps."""

import logging
import os
import pickle
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from forecastops.utils.logger import setup_logger

logger = setup_logger(__name__)


class ModelPredictor:
    """Predictor for loaded models."""
    
    def __init__(self, model_path: Optional[str] = None):
        """Initialize predictor.
        
        Args:
            model_path: Optional path to pre-trained model
        """
        self.model_path = model_path
        self.logger = setup_logger(__name__)
        self.model: Optional[Any] = None
        self.model_info: Dict[str, Any] = {}
        self.feature_columns: List[str] = []
        self.target_column: str = "sales"
    
    def load(self, path: Optional[str] = None) -> None:
        """Load model from path.
        
        Args:
            path: Path to model file
        """
        path = path or self.model_path
        
        if not path:
            raise ValueError("No model path provided")
        
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model file not found: {path}")
        
        from forecastops.models.io import load_model
        model = load_model(path)
        self.model = model
        self.feature_columns = model.feature_columns
        self.target_column = model.target_column
        self.model_info = {
            "name": model.name, "trained": model.trained,
            "params": model.get_params(), "metadata": model.metadata,
        }

        self.model_path = path
        self.logger.info(f"Loaded model: {self.model_info['name']}")
    
    def predict(
        self,
        features: pd.DataFrame,
        product_id: Optional[str] = None,
        store_id: Optional[str] = None
    ) -> np.ndarray:
        """Make prediction for given features.
        
        Args:
            features: Feature dataframe
            product_id: Optional product ID for single prediction
            store_id: Optional store ID for single prediction
            
        Returns:
            Predictions array
        """
        if self.model is None:
            raise ValueError("Model not loaded. Call load() first.")
        
        if len(features) == 0:
            raise ValueError("At least one feature row is required")
        missing = set(self.feature_columns) - set(features.columns)
        if missing:
            raise ValueError(f"Missing features: {sorted(missing)}")
        numeric = [c for c in self.feature_columns if c not in {"product_id", "store_id"}]
        X = features.copy()
        try:
            X[numeric] = X[numeric].apply(pd.to_numeric, errors="raise")
        except (ValueError, TypeError) as exc:
            raise ValueError("Model features must be numeric") from exc
        if not np.isfinite(X[numeric].to_numpy(dtype=float)).all():
            raise ValueError("Model features must be finite and non-null")
        predictions = self.model.predict(X)

        return np.maximum(predictions, 0)
    
    def predict_single(
        self,
        features: Dict[str, Any]
    ) -> float:
        """Make prediction for a single record.
        
        Args:
            features: Dictionary of feature values
            
        Returns:
            Prediction value
        """
        df = pd.DataFrame([features])
        prediction = self.predict(df)
        return float(prediction[0])
    
    def batch_predict(
        self,
        features_list: List[Dict[str, Any]]
    ) -> List[float]:
        """Make predictions for multiple records.
        
        Args:
            features_list: List of feature dictionaries
            
        Returns:
            List of predictions
        """
        df = pd.DataFrame(features_list)
        predictions = self.predict(df)
        return predictions.tolist()
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information.
        
        Returns:
            Dictionary of model info
        """
        return {
            **self.model_info,
            "feature_columns": self.feature_columns,
            "target_column": self.target_column,
        }
    
    def get_feature_columns(self) -> List[str]:
        """Get feature column names.
        
        Returns:
            List of feature column names
        """
        return self.feature_columns
    
    def validate_input(
        self,
        features: Dict[str, Any]
    ) -> bool:
        """Validate input features.
        
        Args:
            features: Feature dictionary
            
        Returns:
            True if valid
        """
        required = set(self.feature_columns)
        provided = set(features.keys())
        
        missing = required - provided
        if missing:
            logger.warning(f"Missing features: {missing}")
            return False
        
        return True
    
    def save(self, path: str) -> None:
        """Save predictor configuration.
        
        Args:
            path: Path to save
        """
        data = {
            "model_path": self.model_path,
            "model_info": self.model_info,
            "feature_columns": self.feature_columns,
            "target_column": self.target_column
        }
        
        with open(path, "wb") as f:
            pickle.dump(data, f)
        
        self.logger.info(f"Saved predictor config to {path}")
    
    @classmethod
    def load_from_config(cls, config_path: str) -> "ModelPredictor":
        """Load predictor from configuration file.
        
        Args:
            config_path: Path to config file
            
        Returns:
            Predictor instance
        """
        predictor = cls()
        predictor.load(config_path)
        return predictor


class EnsemblePredictor:
    """Ensemble predictor combining multiple models."""
    
    def __init__(self, weights: Optional[List[float]] = None):
        """Initialize ensemble predictor.
        
        Args:
            weights: Optional weights for each model
        """
        self.predictors: List[ModelPredictor] = []
        self.weights = weights or []
        self.logger = setup_logger(__name__)
    
    def add_model(self, predictor: ModelPredictor, weight: float = 1.0) -> None:
        """Add a model to the ensemble.
        
        Args:
            predictor: Model predictor
            weight: Weight for this model
        """
        self.predictors.append(predictor)
        self.weights.append(weight)
    
    def predict(self, features: pd.DataFrame) -> np.ndarray:
        """Make ensemble prediction.
        
        Args:
            features: Feature dataframe
            
        Returns:
            Ensemble predictions
        """
        if not self.predictors:
            raise ValueError("No models in ensemble")
        
        # Get predictions from all models
        predictions = np.array([
            p.predict(features) for p in self.predictors
        ])
        
        # Apply weights (normalize first)
        weights = np.array(self.weights)
        weights = weights / weights.sum()
        
        # Weighted average
        ensemble_pred = np.average(predictions, axis=0, weights=weights)
        
        return ensemble_pred
    
    def get_model_info(self) -> List[Dict[str, Any]]:
        """Get info for all models.
        
        Returns:
            List of model info dictionaries
        """
        return [p.get_model_info() for p in self.predictors]
