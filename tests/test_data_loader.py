"""Tests for data loader module."""

import pytest
import pandas as pd
import numpy as np

from forecastops.data.loader import DataLoader


class TestDataLoader:
    """Test DataLoader class."""
    
    def test_init(self):
        """Test DataLoader initialization."""
        loader = DataLoader(config={})
        assert loader.conn is None
        assert loader.db_config is not None
    
    def test_init_with_config(self):
        """Test DataLoader initialization with config."""
        config = {"host": "localhost", "port": 5432}
        loader = DataLoader(config=config)
        assert loader.db_config is not None


class TestDataValidator:
    """Test DataValidator class."""
    
    def test_validate_sales(self):
        """Test sales validation."""
        from forecastops.data.validator import DataValidator
        
        validator = DataValidator()
        
        df = pd.DataFrame({
            "id": ["id_1", "id_2", "id_3"],
            "item_id": ["item_1", "item_2", "item_3"],
            "dept_id": ["dept_1", "dept_2", "dept_3"],
            "cat_id": ["cat_1", "cat_2", "cat_3"],
            "store_id": ["store_1", "store_2", "store_3"],
            "state_id": ["state_1", "state_2", "state_3"],
            "d": ["d_1", "d_2", "d_3"],
            "sales": [1, 2, 3]
        })
        
        result = validator.validate_sales(df)
        
        assert result["total_rows"] == 3
        assert len(result["errors"]) == 0
        assert result["metrics"]["min_sales"] == 1
        assert result["metrics"]["max_sales"] == 3
    
    def test_validate_calendar(self):
        """Test calendar validation."""
        from forecastops.data.validator import DataValidator
        
        validator = DataValidator()
        
        df = pd.DataFrame({
            "date": ["2011-01-29", "2011-01-30", "2011-01-31"],
            "d": ["d_1", "d_2", "d_3"],
            "wm_yr_wk": [1, 1, 1],
            "weekday": ["Saturday", "Sunday", "Monday"],
            "wday": [6, 0, 1],
            "month": [1, 1, 1],
            "year": [2011, 2011, 2011],
            "snap_ca": [1, 1, 1],
            "snap_tx": [0, 0, 0],
            "snap_wi": [0, 0, 0],
            "event_name_1": [None, None, None],
            "event_type_1": [None, None, None],
            "event_name_2": [None, None, None],
            "event_type_2": [None, None, None]
        })
        
        result = validator.validate_calendar(df)
        
        assert result["total_rows"] == 3
        assert "date_range" in result["metrics"]


class TestDataProcessor:
    """Test DataProcessor class."""
    
    def test_create_date_features(self):
        """Test date feature creation."""
        from forecastops.data.processor import DataProcessor
        from forecastops.utils.config import Config
        
        processor = DataProcessor(Config())
        
        df = pd.DataFrame({
            "date": pd.date_range("2021-01-01", periods=7),
            "sales": [1, 2, 3, 4, 5, 6, 7]
        })
        
        result = processor.create_date_features(df)
        
        assert "year" in result.columns
        assert "month" in result.columns
        assert "day_of_week" in result.columns
        assert "is_weekend" in result.columns
        assert "season" in result.columns
    
    def test_create_lag_features(self):
        """Test lag feature creation."""
        from forecastops.data.processor import DataProcessor
        from forecastops.utils.config import Config
        
        processor = DataProcessor(Config())
        
        df = pd.DataFrame({
            "product_id": ["p1"] * 10 + ["p2"] * 10,
            "store_id": ["s1"] * 10 + ["s1"] * 10,
            "date": pd.date_range("2021-01-01", periods=10).tolist() * 2,
            "sales": list(range(10)) + list(range(10, 20))
        })
        
        result = processor.create_lag_features(df, lags=[1, 2, 7])
        
        assert "sales_lag_1" in result.columns
        assert "sales_lag_2" in result.columns
        assert "sales_lag_7" in result.columns
    
    def test_create_rolling_features(self):
        """Test rolling feature creation."""
        from forecastops.data.processor import DataProcessor
        from forecastops.utils.config import Config
        
        processor = DataProcessor(Config())
        
        df = pd.DataFrame({
            "product_id": ["p1"] * 30,
            "store_id": ["s1"] * 30,
            "date": pd.date_range("2021-01-01", periods=30),
            "sales": list(range(30))
        })
        
        result = processor.create_rolling_features(df, windows=[7])
        
        assert "sales_rolling_mean_7" in result.columns
        assert "sales_rolling_std_7" in result.columns


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
