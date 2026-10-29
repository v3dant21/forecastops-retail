"""Tests for evaluation metrics."""

import pytest
import numpy as np
import pandas as pd

from forecastops.evaluation.metrics import (
    mae, rmse, wape, smape, mape, mase,
    calculate_all_metrics, rolling_origin_cv, expanding_window_cv,
    segment_metrics
)


class TestMetrics:
    """Test evaluation metrics."""
    
    def test_mae(self):
        """Test MAE calculation."""
        actual = np.array([1, 2, 3, 4, 5])
        predicted = np.array([1, 2, 3, 4, 5])
        
        result = mae(actual, predicted)
        assert result == 0.0
    
    def test_mae_with_errors(self):
        """Test MAE with prediction errors."""
        actual = np.array([1, 2, 3, 4, 5])
        predicted = np.array([1, 2, 3, 4, 6])
        
        result = mae(actual, predicted)
        assert result == 0.2
    
    def test_rmse(self):
        """Test RMSE calculation."""
        actual = np.array([1, 2, 3, 4, 5])
        predicted = np.array([1, 2, 3, 4, 5])
        
        result = rmse(actual, predicted)
        assert result == 0.0
    
    def test_wape(self):
        """Test WAPE calculation."""
        actual = np.array([10, 20, 30])
        predicted = np.array([10, 20, 30])
        
        result = wape(actual, predicted)
        assert result == 0.0
    
    def test_wape_with_errors(self):
        """Test WAPE with prediction errors."""
        actual = np.array([10, 20, 30])
        predicted = np.array([10, 20, 35])
        
        result = wape(actual, predicted)
        expected = 5 / 60
        assert abs(result - expected) < 0.001
    
    def test_smape(self):
        """Test SMAPE calculation."""
        actual = np.array([10, 20, 30])
        predicted = np.array([10, 20, 30])
        
        result = smape(actual, predicted)
        assert result == 0.0
    
    def test_mape(self):
        """Test MAPE calculation."""
        actual = np.array([10, 20, 30])
        predicted = np.array([10, 20, 30])
        
        result = mape(actual, predicted)
        assert result == 0.0
    
    def test_mase(self):
        """Test MASE calculation."""
        actual = np.array([10, 20, 30, 40])
        predicted = np.array([12, 22, 32, 42])
        naive = np.array([0, 10, 20, 30])
        
        result = mase(actual, predicted, naive)
        assert abs(result - 0.2) < 0.01
    
    def test_calculate_all_metrics(self):
        """Test all metrics calculation."""
        actual = np.array([10, 20, 30])
        predicted = np.array([10, 20, 30])
        
        metrics = calculate_all_metrics(actual, predicted)
        
        assert "mae" in metrics
        assert "rmse" in metrics
        assert "wape" in metrics
        assert "smape" in metrics
        assert "mape" in metrics
        
        assert metrics["mae"] == 0.0
        assert metrics["rmse"] == 0.0


class TestTimeSeriesValidation:
    """Test time series validation methods."""
    
    def test_rolling_origin_cv(self):
        """Test rolling origin cross-validation."""
        df = pd.DataFrame({
            "date": pd.date_range("2021-01-01", periods=100),
            "feature": list(range(100))
        })
        y = pd.Series(list(range(100)))
        
        def model_factory():
            from forecastops.models.baseline import LinearRegressor
            return LinearRegressor()
        
        results = rolling_origin_cv(
            df, y, model_factory,
            initial_train_size=30,
            horizon=10,
            step_size=5,
            metric="rmse"
        )
        
        assert len(results) > 0
        assert "start_date" in results[0]
        assert "end_date" in results[0]
        assert "metric" in results[0]
    
    def test_expanding_window_cv(self):
        """Test expanding window cross-validation."""
        df = pd.DataFrame({
            "date": pd.date_range("2021-01-01", periods=100),
            "feature": list(range(100))
        })
        y = pd.Series(list(range(100)))
        
        def model_factory():
            from forecastops.models.baseline import LinearRegressor
            return LinearRegressor()
        
        results = expanding_window_cv(
            df, y, model_factory,
            initial_train_size=30,
            horizon=10,
            step_size=5,
            metric="rmse"
        )
        
        assert len(results) > 0
        assert "start_date" in results[0]


class TestSegmentMetrics:
    """Test segment-based metrics."""
    
    def test_segment_metrics(self):
        """Test metrics by segment."""
        actual = np.array([10, 20, 30, 40, 50])
        predicted = np.array([10, 20, 30, 40, 50])
        segments = np.array([0, 0, 1, 1, 1])
        
        results = segment_metrics(actual, predicted, segments)
        
        assert "segment_0" in results
        assert "segment_1" in results
        assert results["segment_0"]["mae"] == 0.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
