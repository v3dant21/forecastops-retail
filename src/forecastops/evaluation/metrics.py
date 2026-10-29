"""Evaluation metrics for ForecastOps."""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from forecastops.utils.logger import setup_logger

logger = setup_logger(__name__)


def mae(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Calculate Mean Absolute Error.
    
    Args:
        actual: Actual values
        predicted: Predicted values
        
    Returns:
        MAE value
    """
    actual = np.asarray(actual)
    predicted = np.asarray(predicted)
    
    return np.mean(np.abs(actual - predicted))


def rmse(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Calculate Root Mean Square Error.
    
    Args:
        actual: Actual values
        predicted: Predicted values
        
    Returns:
        RMSE value
    """
    actual = np.asarray(actual)
    predicted = np.asarray(predicted)
    
    return np.sqrt(np.mean((actual - predicted) ** 2))


def wape(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Calculate Weighted Absolute Percentage Error.
    
    Args:
        actual: Actual values
        predicted: Predicted values
        
    Returns:
        WAPE value
    """
    actual = np.asarray(actual)
    predicted = np.asarray(predicted)
    
    numerator = np.sum(np.abs(actual - predicted))
    denominator = np.sum(actual)
    
    if denominator == 0:
        return 0.0
    
    return numerator / denominator


def smape(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Calculate Symmetric Mean Absolute Percentage Error.
    
    Args:
        actual: Actual values
        predicted: Predicted values
        
    Returns:
        SMAPE value
    """
    actual = np.asarray(actual)
    predicted = np.asarray(predicted)
    
    numerator = np.sum(np.abs(actual - predicted))
    denominator = np.sum(np.abs(actual) + np.abs(predicted))
    
    if denominator == 0:
        return 0.0
    
    return 2 * numerator / denominator


def mape(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Calculate Mean Absolute Percentage Error.
    
    Args:
        actual: Actual values
        predicted: Predicted values
        
    Returns:
        MAPE value (as percentage)
    """
    actual = np.asarray(actual)
    predicted = np.asarray(predicted)
    
    # Avoid division by zero
    mask = actual != 0
    if not np.any(mask):
        return 0.0
    
    return np.mean(np.abs((actual[mask] - predicted[mask]) / actual[mask])) * 100


def mase(
    actual: np.ndarray,
    predicted: np.ndarray,
    naive_forecast: np.ndarray
) -> float:
    """Calculate Mean Absolute Scaled Error.
    
    Args:
        actual: Actual values
        predicted: Predicted values
        naive_forecast: Naive forecast values
        
    Returns:
        MASE value
    """
    actual = np.asarray(actual)
    predicted = np.asarray(predicted)
    naive_forecast = np.asarray(naive_forecast)
    
    # Mean absolute error of the forecast
    mae_forecast = np.mean(np.abs(actual - predicted))
    
    # Mean absolute error of naive forecast
    mae_naive = np.mean(np.abs(actual - naive_forecast))
    
    if mae_naive == 0:
        return 0.0
    
    return mae_forecast / mae_naive


def calculate_all_metrics(
    actual: np.ndarray,
    predicted: np.ndarray,
    naive_forecast: Optional[np.ndarray] = None
) -> Dict[str, float]:
    """Calculate all evaluation metrics.
    
    Args:
        actual: Actual values
        predicted: Predicted values
        naive_forecast: Optional naive forecast for MASE
        
    Returns:
        Dictionary of metrics
    """
    metrics = {
        "mae": mae(actual, predicted),
        "rmse": rmse(actual, predicted),
        "wape": wape(actual, predicted),
        "smape": smape(actual, predicted),
        "mape": mape(actual, predicted),
    }
    
    if naive_forecast is not None:
        metrics["mase"] = mase(actual, predicted, naive_forecast)
    
    return metrics


def _time_cv(X, y, model_factory, initial_train_size, horizon, step_size, metric, expanding):
    """Use distinct dates as CV units when dates are supplied, otherwise rows."""
    if min(initial_train_size, horizon, step_size) <= 0:
        raise ValueError("CV window sizes must be positive")
    if len(X) != len(y):
        raise ValueError("Features and targets must have equal lengths")
    if "date" in X:
        dates = pd.to_datetime(X["date"], errors="raise")
        if dates.isna().any():
            raise ValueError("CV dates cannot be missing")
        units = np.sort(dates.unique())
        groups = [np.flatnonzero(dates.to_numpy() == date) for date in units]
    else:
        units = np.arange(len(X))
        groups = [np.array([i]) for i in range(len(X))]
    scorers = {"mae": mae, "rmse": rmse, "wape": wape}
    if metric not in scorers:
        raise ValueError(f"Unknown CV metric: {metric}")
    results = []
    for end in range(initial_train_size, len(groups) - horizon + 1, step_size):
        start = 0 if expanding else end - initial_train_size
        train_rows = np.concatenate(groups[start:end])
        val_rows = np.concatenate(groups[end:end+horizon])
        model = model_factory()
        model.fit(X.iloc[train_rows].copy(), y.iloc[train_rows].copy())
        predictions = model.predict(X.iloc[val_rows].copy())
        results.append({"start_date": units[end], "end_date": units[end+horizon-1],
                        "metric": scorers[metric](y.iloc[val_rows].to_numpy(), predictions),
                        "horizon": horizon})
    return results


def rolling_origin_cv(X, y, model_factory, initial_train_size=100, horizon=28,
                      step_size=7, metric="rmse"):
    """Fixed training window moving over distinct dates (or rows without dates)."""
    return _time_cv(X, y, model_factory, initial_train_size, horizon, step_size, metric, False)


def expanding_window_cv(X, y, model_factory, initial_train_size=100, horizon=28,
                        step_size=7, metric="rmse"):
    """Growing training window over distinct dates (or rows without dates)."""
    return _time_cv(X, y, model_factory, initial_train_size, horizon, step_size, metric, True)


def segment_metrics(
    actual: np.ndarray,
    predicted: np.ndarray,
    segments: Optional[np.ndarray] = None
) -> Dict[str, Dict[str, float]]:
    """Calculate metrics by segment.
    
    Args:
        actual: Actual values
        predicted: Predicted values
        segments: Segment labels
        
    Returns:
        Dictionary of segment metrics
    """
    if segments is None:
        segments = np.zeros(len(actual), dtype=int)
    
    unique_segments = np.unique(segments)
    results = {}
    
    for seg in unique_segments:
        mask = segments == seg
        seg_actual = actual[mask]
        seg_predicted = predicted[mask]
        
        if len(seg_actual) > 0:
            results[f"segment_{seg}"] = {
                "mae": mae(seg_actual, seg_predicted),
                "rmse": rmse(seg_actual, seg_predicted),
                "wape": wape(seg_actual, seg_predicted),
                "count": int(len(seg_actual))
            }
    
    return results
