"""Helper functions for ForecastOps."""

import random
from typing import Dict, Any, List

import numpy as np
import pandas as pd
import torch


def set_seed(seed: int = 42) -> None:
    """Set random seeds for reproducibility.
    
    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def format_metrics(metrics: Dict[str, float]) -> str:
    """Format metrics dictionary into a readable string.
    
    Args:
        metrics: Dictionary of metric names and values
        
    Returns:
        Formatted string representation
    """
    return " | ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])


def rolling_window(
    data: np.ndarray,
    window_size: int,
    step: int = 1
) -> np.ndarray:
    """Create rolling window view of array.
    
    Args:
        data: Input array
        window_size: Size of window
        step: Step size between windows
        
    Returns:
        2D array of windows
    """
    if window_size > len(data):
        return np.array([])
    
    n_windows = (len(data) - window_size) // step + 1
    windows = np.zeros((n_windows, window_size))
    
    for i in range(n_windows):
        start = i * step
        windows[i] = data[start:start + window_size]
    
    return windows


def train_val_test_split(
    data: pd.DataFrame,
    test_ratio: float = 0.1,
    val_ratio: float = 0.2,
    time_column: str = "date"
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split data into train, validation, and test sets.
    
    Args:
        data: Input dataframe
        test_ratio: Ratio for test set
        val_ratio: Ratio for validation set
        time_column: Column to sort by for time-based splitting
        
    Returns:
        Tuple of (train, val, test) dataframes
    """
    if not (0 < test_ratio < 1 and 0 < val_ratio < 1 and test_ratio + val_ratio < 1):
        raise ValueError("Validation/test ratios must be positive and sum to less than one")
    data = data.copy()
    data[time_column] = pd.to_datetime(data[time_column], errors="raise")
    if data[time_column].isna().any():
        raise ValueError("Split dates cannot be missing")
    data = data.sort_values(time_column, kind="stable")
    dates = data[time_column].drop_duplicates().to_numpy()
    n_test = max(1, int(len(dates) * test_ratio))
    n_val = max(1, int(len(dates) * val_ratio))
    if len(dates) <= n_test + n_val:
        raise ValueError("At least three distinct dates are required for nonempty splits")
    val_start = dates[-n_test - n_val]
    test_start = dates[-n_test]
    return (
        data[data[time_column] < val_start].copy(),
        data[(data[time_column] >= val_start) & (data[time_column] < test_start)].copy(),
        data[data[time_column] >= test_start].copy(),
    )


def split_features(features, config):
    """Shared one-step-ahead evaluation split; observed past sales form lags."""
    splits = train_val_test_split(features, config.training.test_split,
                                  config.training.validation_split)
    target = config.model.target_column
    columns = [c for c in features.columns if c not in {target, "id"}]
    return tuple(part for frame in splits for part in (frame[columns].copy(), frame[target].copy()))


def calculate_wape(actual: np.ndarray, predicted: np.ndarray) -> float:
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


def calculate_smape(actual: np.ndarray, predicted: np.ndarray) -> float:
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


def calculate_mape(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Calculate Mean Absolute Percentage Error.
    
    Args:
        actual: Actual values
        predicted: Predicted values
        
    Returns:
        MAPE value
    """
    actual = np.asarray(actual)
    predicted = np.asarray(predicted)
    
    # Avoid division by zero
    mask = actual != 0
    if not np.any(mask):
        return 0.0
    
    return np.mean(np.abs((actual[mask] - predicted[mask]) / actual[mask])) * 100
