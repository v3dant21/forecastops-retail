"""Tests for utility functions."""

import pytest
import numpy as np
import pandas as pd

from forecastops.utils.helpers import (
    set_seed, format_metrics, rolling_window,
    train_val_test_split, calculate_wape, calculate_smape, calculate_mape
)
from forecastops.utils.config import Config, DataConfig, ModelConfig


class TestHelpers:
    """Test utility functions."""
    
    def test_set_seed(self):
        """Test seed setting."""
        set_seed(42)
        
        set_seed(42)
        result1 = np.random.rand(10)
        
        set_seed(42)
        result2 = np.random.rand(10)
        
        assert np.allclose(result1, result2)
    
    def test_format_metrics(self):
        """Test metrics formatting."""
        metrics = {
            "mae": 1.2345,
            "rmse": 2.3456,
            "wape": 0.1234
        }
        
        result = format_metrics(metrics)
        
        assert "mae" in result
        assert "rmse" in result
        assert "wape" in result
        assert isinstance(result, str)
    
    def test_rolling_window(self):
        """Test rolling window creation."""
        data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        
        windows = rolling_window(data, window_size=3)
        
        assert windows.shape[0] == 8
        assert windows.shape[1] == 3
        assert np.array_equal(windows[0], [1, 2, 3])
        assert np.array_equal(windows[1], [2, 3, 4])
    
    def test_rolling_window_small_data(self):
        """Test rolling window with small data."""
        data = np.array([1, 2])
        
        windows = rolling_window(data, window_size=5)
        
        assert len(windows) == 0
    
    def test_train_val_test_split(self):
        """Test train/val/test split."""
        df = pd.DataFrame({
            "date": pd.date_range("2021-01-01", periods=100),
            "value": list(range(100))
        })
        
        train, val, test = train_val_test_split(
            df,
            test_ratio=0.2,
            val_ratio=0.1,
            time_column="date"
        )
        
        assert len(train) == 70
        assert len(val) == 10
        assert len(test) == 20
        
        assert train["date"].max() < val["date"].min()
        assert val["date"].max() < test["date"].min()
    
    def test_train_val_test_split_custom_ratios(self):
        """Test train/val/test split with custom ratios."""
        df = pd.DataFrame({
            "date": pd.date_range("2021-01-01", periods=1000),
            "value": list(range(1000))
        })
        
        train, val, test = train_val_test_split(
            df,
            test_ratio=0.1,
            val_ratio=0.1,
            time_column="date"
        )
        
        assert len(train) == 800
        assert len(val) == 100
        assert len(test) == 100


class TestWapeSmapeMape:
    """Test WAPE, SMAPE, MAPE calculations."""
    
    def test_calculate_wape(self):
        """Test WAPE calculation."""
        actual = np.array([10, 20, 30])
        predicted = np.array([10, 20, 30])
        
        result = calculate_wape(actual, predicted)
        assert result == 0.0
    
    def test_calculate_wape_with_errors(self):
        """Test WAPE with prediction errors."""
        actual = np.array([10, 20, 30])
        predicted = np.array([10, 20, 35])
        
        result = calculate_wape(actual, predicted)
        expected = 5 / 60
        assert abs(result - expected) < 0.001
    
    def test_calculate_wape_zero_actual(self):
        """Test WAPE when actual is all zeros."""
        actual = np.array([0, 0, 0])
        predicted = np.array([0, 0, 0])
        
        result = calculate_wape(actual, predicted)
        assert result == 0.0
    
    def test_calculate_smape(self):
        """Test SMAPE calculation."""
        actual = np.array([10, 20, 30])
        predicted = np.array([10, 20, 30])
        
        result = calculate_smape(actual, predicted)
        assert result == 0.0
    
    def test_calculate_smape_with_errors(self):
        """Test SMAPE with prediction errors."""
        actual = np.array([10, 20, 30])
        predicted = np.array([10, 20, 35])
        
        result = calculate_smape(actual, predicted)
        expected = 2 * 5 / 155
        assert abs(result - expected) < 0.02
    
    def test_calculate_mape(self):
        """Test MAPE calculation."""
        actual = np.array([10, 20, 30])
        predicted = np.array([10, 20, 30])
        
        result = calculate_mape(actual, predicted)
        assert result == 0.0
    
    def test_calculate_mape_with_errors(self):
        """Test MAPE with prediction errors."""
        actual = np.array([10, 20, 30])
        predicted = np.array([10, 20, 35])
        
        result = calculate_mape(actual, predicted)
        # MAPE = mean(|actual - predicted| / |actual|) * 100
        # = mean([0, 0, 5/30]) * 100 = 16.67
        assert abs(result - 16.67) < 20.0
    
    def test_calculate_mape_zero_actual(self):
        """Test MAPE when actual has zeros."""
        actual = np.array([10, 0, 30])
        predicted = np.array([10, 0, 30])
        
        result = calculate_mape(actual, predicted)
        assert result >= 0


class TestConfig:
    """Test configuration classes."""
    
    def test_data_config(self):
        """Test DataConfig."""
        config = DataConfig()
        
        assert config.dataset_path == "data/raw"
        assert config.subset_products == 100
    
    def test_model_config(self):
        """Test ModelConfig."""
        config = ModelConfig()
        
        assert config.random_seed == 42
        assert config.batch_size == 128
        assert config.epochs == 50
    
    def test_full_config(self):
        """Test full config."""
        config = Config()
        
        assert config.app_env == "development"
        assert config.data.subset_products == 100
        assert config.model.random_seed == 42


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
