"""Time-series validation utilities."""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from forecastops.utils.logger import setup_logger

logger = setup_logger(__name__)


class TimeSeriesValidator:
    """Validator for time-series model performance."""
    
    def __init__(self):
        """Initialize validator."""
        self.logger = setup_logger(__name__)
        self.validation_results: Dict = {}
    
    def validate_no_leakage(
        self,
        train_dates: pd.Series,
        val_dates: pd.Series,
        test_dates: pd.Series
    ) -> bool:
        """Validate that there's no date leakage between splits.
        
        Args:
            train_dates: Training dates
            val_dates: Validation dates
            test_dates: Test dates
            
        Returns:
            True if no leakage detected
        """
        train_end = train_dates.max()
        val_start = val_dates.min()
        val_end = val_dates.max()
        test_start = test_dates.min()
        
        leakage = False
        
        if val_start <= train_end:
            self.logger.error(f"Validation start ({val_start}) is before or at training end ({train_end})")
            leakage = True
        
        if test_start <= val_end:
            self.logger.error(f"Test start ({test_start}) is before or at validation end ({val_end})")
            leakage = True
        
        if not leakage:
            self.logger.info("No date leakage detected")
        
        return not leakage
    
    def validate_stationarity(
        self,
        series: pd.Series,
        window: int = 30
    ) -> Dict[str, float]:
        """Check for stationarity in the series.
        
        Args:
            series: Time series
            window: Rolling window size
            
        Returns:
            Dictionary of stationarity checks
        """
        results = {
            "mean_change": 0.0,
            "std_change": 0.0,
            "is_stationary": False
        }
        
        # Rolling statistics
        rolling_mean = series.rolling(window).mean()
        rolling_std = series.rolling(window).std()
        
        results["mean_change"] = float(rolling_mean.diff().abs().mean())
        results["std_change"] = float(rolling_std.diff().abs().mean())
        
        # Simple stationarity check (not rigorous, just heuristic)
        if results["mean_change"] < series.mean() * 0.1:
            results["is_stationary"] = True
        
        return results
    
    def validate_distribution_shift(
        self,
        train_values: np.ndarray,
        val_values: np.ndarray,
        test_values: np.ndarray
    ) -> Dict[str, float]:
        """Check for distribution shift between splits.
        
        Args:
            train_values: Training values
            val_values: Validation values
            test_values: Test values
            
        Returns:
            Dictionary of shift checks
        """
        results = {
            "train_mean": float(np.mean(train_values)),
            "val_mean": float(np.mean(val_values)),
            "test_mean": float(np.mean(test_values)),
            "train_std": float(np.std(train_values)),
            "val_std": float(np.std(val_values)),
            "test_std": float(np.std(test_values)),
            "val_train_mean_ratio": float(np.mean(val_values) / (np.mean(train_values) + 1e-8)),
            "test_train_mean_ratio": float(np.mean(test_values) / (np.mean(train_values) + 1e-8)),
        }
        
        return results
    
    def validate_sparsity(
        self,
        series: pd.Series,
        threshold: float = 0.0
    ) -> Dict[str, float]:
        """Check for sparsity in the series.
        
        Args:
            series: Time series
            threshold: Value threshold for sparsity
        
        Returns:
            Dictionary of sparsity metrics
        """
        total = len(series)
        zeros = (series <= threshold).sum()
        sparsity = zeros / total if total > 0 else 0.0
        
        return {
            "total": int(total),
            "zeros": int(zeros),
            "sparsity": float(sparsity),
            "non_zero_mean": float(series[series > threshold].mean()) if (series > threshold).any() else 0.0
        }
    
    def validate_model_performance(
        self,
        model,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series
    ) -> Dict[str, float]:
        """Validate model performance across all splits.
        
        Args:
            model: Trained model
            X_train: Training features
            y_train: Training targets
            X_val: Validation features
            y_val: Validation targets
            X_test: Test features
            y_test: Test targets
            
        Returns:
            Dictionary of performance metrics
        """
        results = {}
        
        # Get predictions
        train_pred = model.predict(X_train)
        val_pred = model.predict(X_val)
        test_pred = model.predict(X_test)
        
        # Calculate metrics for each split
        from forecastops.evaluation.metrics import mae, rmse, wape
        
        for split_name, y_true, y_pred in [
            ("train", y_train.values, train_pred),
            ("val", y_val.values, val_pred),
            ("test", y_test.values, test_pred)
        ]:
            results[f"{split_name}_mae"] = mae(y_true, y_pred)
            results[f"{split_name}_rmse"] = rmse(y_true, y_pred)
            results[f"{split_name}_wape"] = wape(y_true, y_pred)
            
        train_mae = results["train_mae"]
        results["train_overfitting"] = bool(
            (results["val_mae"] - train_mae) / max(train_mae, 1e-8) > 0.2
        )

        self.validation_results["performance"] = results
        return results
    
    def validate_error_distribution(
        self,
        actual: np.ndarray,
        predicted: np.ndarray
    ) -> Dict[str, float]:
        """Validate error distribution assumptions.
        
        Args:
            actual: Actual values
            predicted: Predicted values
            
        Returns:
            Dictionary of error distribution checks
        """
        errors = actual - predicted
        
        results = {
            "mean_error": float(np.mean(errors)),
            "std_error": float(np.std(errors)),
            "skewness": float(pd.Series(errors).skew()),
            "kurtosis": float(pd.Series(errors).kurt()),
            "bias": float(np.mean(errors)),  # Positive = under-predicting
        }
        
        # Check for systematic bias
        if abs(results["mean_error"]) > np.std(errors) * 0.5:
            self.logger.warning("Systematic bias detected in predictions")
        
        return results
    
    def validate_low_demand_periods(
        self,
        actual: np.ndarray,
        predicted: np.ndarray,
        threshold: float = 0.0
    ) -> Dict[str, float]:
        """Validate performance during low-demand periods.
        
        Args:
            actual: Actual values
            predicted: Predicted values
            threshold: Threshold for low demand
        
        Returns:
            Dictionary of low-demand metrics
        """
        low_mask = actual <= threshold
        if not np.any(low_mask):
            return {"no_low_demand": True}
        
        low_actual = actual[low_mask]
        low_predicted = predicted[low_mask]
        
        from forecastops.evaluation.metrics import mae, rmse
        
        return {
            "low_demand_count": int(np.sum(low_mask)),
            "low_demand_mae": mae(low_actual, low_predicted),
            "low_demand_rmse": rmse(low_actual, low_predicted),
            "low_demand_wape": float(np.sum(np.abs(low_actual - low_predicted)) / (np.sum(low_actual) + 1e-8))
        }
    
    def validate_high_demand_periods(
        self,
        actual: np.ndarray,
        predicted: np.ndarray,
        threshold: float = None
    ) -> Dict[str, float]:
        """Validate performance during high-demand periods.
        
        Args:
            actual: Actual values
            predicted: Predicted values
            threshold: Threshold for high demand (default: 75th percentile)
        
        Returns:
            Dictionary of high-demand metrics
        """
        if threshold is None:
            threshold = np.percentile(actual, 75)
        
        high_mask = actual > threshold
        if not np.any(high_mask):
            return {"no_high_demand": True}
        
        high_actual = actual[high_mask]
        high_predicted = predicted[high_mask]
        
        from forecastops.evaluation.metrics import mae, rmse
        
        return {
            "high_demand_count": int(np.sum(high_mask)),
            "high_demand_mae": mae(high_actual, high_predicted),
            "high_demand_rmse": rmse(high_actual, high_predicted),
            "high_demand_wape": float(np.sum(np.abs(high_actual - high_predicted)) / (np.sum(high_actual) + 1e-8))
        }
    
    def run_full_validation(
        self,
        model,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series
    ) -> Dict:
        """Run complete validation.
        
        Args:
            model: Trained model
            X_train: Training features
            y_train: Training targets
            X_val: Validation features
            y_val: Validation targets
            X_test: Test features
            y_test: Test targets
            
        Returns:
            Complete validation results
        """
        results = {}
        
        # Date validation
        results["date_validation"] = self.validate_no_leakage(
            X_train["date"], X_val["date"], X_test["date"]
        )
        
        # Performance validation
        results["performance"] = self.validate_model_performance(
            model, X_train, y_train, X_val, y_val, X_test, y_test
        )
        
        # Error distribution
        results["error_distribution"] = self.validate_error_distribution(
            y_test.values, model.predict(X_test)
        )
        
        # Low/high demand validation
        results["low_demand"] = self.validate_low_demand_periods(
            y_test.values, model.predict(X_test)
        )
        
        results["high_demand"] = self.validate_high_demand_periods(
            y_test.values, model.predict(X_test)
        )
        
        self.validation_results["full"] = results
        return results
